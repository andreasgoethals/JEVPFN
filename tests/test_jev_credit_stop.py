"""Synthetic exhausted-credit responses must stop safely without duplicate paid work."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import pytest

from src.cluster.gpu_contracts import synthetic_datasets
from src.jev.cache import ResponseCache
from src.jev.extraction import (
    AccountCreditError,
    DatasetExtractor,
    ExtractionLimits,
    RateGate,
    TypeSafeTransport,
)
from src.jev.requests import build_requests
from src.utils.config import load_config


@pytest.fixture
def workload(tmp_path, isolated_output):
    dataset = next(synthetic_datasets())
    for n in range(3):
        dataset.frame.loc[n, dataset.metadata.text_columns[0]] = f"Unique synthetic example {n}"
    requests = [build_requests(dataset.features, dataset.metadata, n, load_config())[0]
                for n in range(3)]
    assert len({r.cache_key for r in requests}) == 3
    cache = ResponseCache(tmp_path / "credit.sqlite3", dataset.metadata.dataset_id)
    limits = ExtractionLimits(1, .042, 64000, 1, 100000)
    return requests, cache, limits


def make_extractor(cache, client, limits, **kw):
    return DatasetExtractor(cache, client, limits, validate_context=lambda _: None,
                            gate=kw.pop("gate", SimpleNamespace(acquire=lambda: None)), **kw)


def successful_response():
    return {"model": "jev-1.14.0", "usage": {"input_tokens": 10},
            "answers": {"semantic": {"type": "noul", "noul": .4}}}


@pytest.mark.parametrize("status,body", [
    (402, {"message": "private-key-and-row-text-must-not-be-logged"}),
    (429, {"error": {"code": "insufficient_credits"}}),
    (403, {"code": "insufficient_balance"}),
])
def test_real_sdk_mock_classifies_payment_rejection_without_retry(monkeypatch, status, body, workload):
    sdk = pytest.importorskip("typesafe_sdk")
    httpx = pytest.importorskip("httpx2")
    requests, _, _ = workload
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json=body, headers={"retry-after": "2"})

    cfg = load_config("feature_creation")
    cfg["live_extraction"].update(
        enabled=True, pilot_authorized=True, account_limits_verified=True,
        billing_bound_verified=True, context_policy_reviewed=True, price_verified=True,
        owner_budget_eur=None, campaign_spending_limit_usd=1,
        dataset_budget_allocations_usd={"fixture": 1},
        max_billed_input_tokens_per_attempt=64000, requests_per_second=1,
        input_tokens_per_second=100000,
    )
    real_client = sdk.TypeSafeClient
    monkeypatch.setattr(sdk, "TypeSafeClient",
                        lambda **kw: real_client(**kw, transport=httpx.MockTransport(handler)))
    monkeypatch.setenv("TYPESAFE_API_KEY", "synthetic-key-no-network")
    transport = TypeSafeTransport(cfg, allow_network=True)
    try:
        with pytest.raises(AccountCreditError) as exc:
            transport.evaluate(requests[0])
        assert exc.value.http_status == status and len(calls) == 1
        assert "private-key" not in str(exc.value)
    finally:
        transport.close()


@pytest.mark.parametrize("body", [None, [], "balance", {"error": "insufficient credits"},
                                   {"code": ["insufficient_credits"]}])
def test_unknown_bodies_are_not_assumed_unbilled_or_credit_errors(body):
    assert AccountCreditError.from_api_error(SimpleNamespace(status=500, body=body)) is None


def test_balance_stop_preserves_charge_reservation_cache_and_restart(workload):
    requests, cache, limits = workload
    calls = []

    def evaluate(request):
        calls.append(request.cache_key)
        if len(calls) == 2:
            raise AccountCreditError(http_status=402)
        return successful_response()

    client = SimpleNamespace(evaluate=evaluate)
    extractor = make_extractor(cache, client, limits)
    assert extractor.run(requests) == {"completed": 1, "ambiguous": 1}
    assert calls == [r.cache_key for r in requests[:2]]
    assert extractor.stop_details["http_status"] == 402
    with cache._connect() as db:
        jobs = dict(db.execute("SELECT request_key,status FROM extraction_jobs"))
        spent, reserved = db.execute("SELECT SUM(spent_usd),SUM(reserved_usd) FROM extraction_jobs").fetchone()
    assert jobs == {requests[0].cache_key: "completed", requests[1].cache_key: "ambiguous"}
    assert spent == pytest.approx(10 * .042 / 1e6)
    assert reserved == pytest.approx(64000 * .042 / 1e6)
    # Simulate a funded account: an unresolved old attempt still cannot be paid again.
    restarted = make_extractor(cache, client, limits)
    assert restarted.run(requests) == {"cache_hit": 1, "ambiguous": 1}
    assert len(calls) == 2
    assert restarted.stop_details["reason"] == "account_credit_or_payment_required"
    events = [json.loads(line) for line in extractor.log.read_text().splitlines()]
    assert any(e["event"] == "account_stopped" and e["http_status"] == 402 for e in events)


def test_waiting_parallel_worker_releases_only_its_unsent_reservation(workload):
    requests, cache, limits = workload
    waiting, release = Event(), Event()
    calls = []

    def evaluate(request):
        calls.append(request.cache_key)
        raise AccountCreditError(http_status=402, code="insufficient_credits")

    def acquire():
        waiting.set()
        assert release.wait(5)

    extractor = make_extractor(cache, SimpleNamespace(evaluate=evaluate), replace(limits, workers=2),
                               gate=SimpleNamespace(acquire=acquire))
    with ThreadPoolExecutor(max_workers=1) as pool:
        unsent = pool.submit(extractor.evaluate, requests[1])
        try:
            assert waiting.wait(5)
            extractor.gate = SimpleNamespace(acquire=lambda: None)
            assert extractor.evaluate(requests[0]) == "ambiguous"
        finally:
            release.set()
        assert unsent.result(timeout=5) == "account_stopped"
    assert calls == [requests[0].cache_key]
    with cache._connect() as db:
        assert db.execute("SELECT status,attempts,reserved_usd FROM extraction_jobs WHERE request_key=?",
                          (requests[1].cache_key,)).fetchone() == ("pending", 0, 0)
        assert db.execute("SELECT reserved_usd FROM extraction_jobs WHERE request_key=?",
                          (requests[0].cache_key,)).fetchone()[0] > 0


def test_unknown_api_failure_retains_status_without_response_message(workload):
    requests, cache, limits = workload

    class UnknownAPIError(RuntimeError):
        status = 403

    def evaluate(_):
        raise UnknownAPIError("private-key-and-row-text-must-not-be-logged")

    extractor = make_extractor(cache, SimpleNamespace(evaluate=evaluate), limits)
    assert extractor.run(requests) == {"ambiguous": 1}
    log = extractor.log.read_text()
    assert "private-key" not in log
    assert any(e.get("http_status") == 403 for e in map(json.loads, log.splitlines()))


def test_hosted_receipt_explains_credit_stop_and_preserves_safe_restart(isolated_output, monkeypatch):
    from src.jev import hosted
    from src.jev.inputs import prepare_dataset_inputs
    from src.jev.question_bank import build_question_bank, save_question_bank
    from src.utils import paths
    from src.utils.config import load_yaml

    dataset = next(synthetic_datasets())
    bank = build_question_bank([dataset.metadata], load_yaml("question_design"))
    save_question_bank(bank)
    prepared = prepare_dataset_inputs(dataset, bank, variant="accepted")
    identifier = dataset.metadata.dataset_id
    monkeypatch.setattr(paths, "on_vsc", lambda: False)
    monkeypatch.setattr(hosted, "selected_inputs", lambda: [
        {"dataset_id": identifier, "input_set_id": prepared["input_set_id"]}
    ])
    monkeypatch.setattr(hosted, "RateGate", lambda _: SimpleNamespace(acquire=lambda: None))
    cfg = load_config("feature_creation")
    cfg["live_extraction"].update(
        enabled=True, owner_budget_eur=None, campaign_spending_limit_usd=1,
        dataset_budget_allocations_usd={identifier: 1}, workers=1,
        requests_per_second=1, input_tokens_per_second=100000,
        max_billed_input_tokens_per_attempt=64000,
        context_envelope={"verified": True, "model": cfg["jev"]["model"],
            "mode": "provider_enforced", "max_request_utf8_bytes": 32 * 1024 * 1024,
            "max_billed_input_tokens": 64000, "per_question_context_tokens": 32000,
            "evidence": "synthetic"},
    )
    calls = []

    def evaluate(request):
        calls.append(request.cache_key)
        raise AccountCreditError(http_status=402)

    def factory(*a, **kw):
        return SimpleNamespace(evaluate=evaluate)
    first = hosted.run(cfg, execute=True, transport_factory=factory)
    second = hosted.run(cfg, execute=True, transport_factory=factory)
    assert first["status"] == second["status"] == "stopped_requires_review"
    assert len(calls) == 1
    assert first["stop_details"]["http_status"] == 402
    assert first["stop_details"]["dataset_id"] == identifier
    manifest = paths.manifests_dir("experiment_0") / "hosted" / f"{first['run_id']}.json"
    assert json.loads(manifest.read_text())["stop_details"] == first["stop_details"]


def test_faster_gate_caps_rps_and_shares_backoff_without_changing_money_bound(workload):
    _, _, limits = workload
    limits = replace(limits, requests_per_second=20, workers=8)
    assert RateGate(limits).interval == .64
    now = [0.0]
    waits = []

    def sleep(delay):
        waits.append(delay)
        now[0] += delay

    gate = RateGate(limits, provider_enforced_tokens=True, clock=lambda: now[0], sleep=sleep)
    gate.acquire()
    gate.acquire()
    assert now[0] == pytest.approx(.05)
    gate.defer(3)
    gate.acquire()
    assert waits == pytest.approx([.05, 3])
    assert limits.max_billed_input_tokens == 64000


@pytest.mark.parametrize("changes", [
    {"account_limits_verified": False}, {"requests_per_second": 81},
    {"input_tokens_per_second": 100001}, {"verified_account_input_tokens_per_second": None},
    {"token_rate_policy": "unbounded"},
])
def test_faster_gate_requires_explicit_verified_limits(changes, workload):
    from src.jev.hosted import rate_gate

    _, _, limits = workload
    review = dict(token_rate_policy="provider_enforced", account_limits_verified=True,
                  requests_per_second=20, input_tokens_per_second=100000,
                  verified_account_requests_per_second=80, verified_account_input_tokens_per_second=100000)
    assert rate_gate(review, replace(limits, requests_per_second=20)).interval == .05
    review.update(changes)
    with pytest.raises(ValueError):
        rate_gate(review, limits)


def test_allocated_budget_stops_can_continue_without_materializing_partial_datasets(isolated_output, monkeypatch):
    from src.jev import hosted
    from src.jev.inputs import prepare_dataset_inputs
    from src.jev.question_bank import build_question_bank, save_question_bank
    from src.utils import paths
    from src.utils.config import load_yaml

    datasets = list(synthetic_datasets())[:2]
    bank = build_question_bank([d.metadata for d in datasets], load_yaml("question_design"))
    save_question_bank(bank)
    items = []
    for dataset in datasets:
        prepared = prepare_dataset_inputs(dataset, bank, variant="accepted")
        items.append({"dataset_id": dataset.metadata.dataset_id, "input_set_id": prepared["input_set_id"]})
    monkeypatch.setattr(paths, "on_vsc", lambda: False)
    monkeypatch.setattr(hosted, "selected_inputs", lambda: items)
    monkeypatch.setattr(hosted, "reviewed_acceptance", lambda *a, **kw: {"accepted": True})
    cfg = load_config("feature_creation")
    cfg["live_extraction"].update(
        enabled=True, owner_budget_eur=None, campaign_spending_limit_usd=.000002,
        dataset_budget_allocations_usd={i["dataset_id"]: .000001 for i in items},
        workers=2, full_feature_creation_enabled=True, pilot_accepted=True,
        continue_on_dataset_budget_stop=True,
        requests_per_second=1, input_tokens_per_second=100000,
        max_billed_input_tokens_per_attempt=64000,
        context_envelope={"verified": True, "model": cfg["jev"]["model"],
            "mode": "provider_enforced", "max_request_utf8_bytes": 32 * 1024 * 1024,
            "max_billed_input_tokens": 64000, "per_question_context_tokens": 32000,
            "evidence": "synthetic"},
    )

    def forbidden(*_):
        raise AssertionError("No request fits these budgets; no paid call or materialization is allowed")

    monkeypatch.setattr("src.jev.self_hosted._materialize", forbidden)
    report = hosted.run(cfg, execute=True, full=True,
                        transport_factory=lambda *a, **kw: SimpleNamespace(evaluate=forbidden))
    assert report["status"] == "stopped_requires_budget"
    assert len(report["datasets"]) == len(report["budget_stopped_datasets"]) == 2
    assert all(set(d["counts"]) == {"budget_stop"} for d in report["datasets"])
