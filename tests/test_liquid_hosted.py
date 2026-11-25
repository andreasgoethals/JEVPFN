"""Hosted Liquid d1.2-ultra adapter: offline mapping, decoding, gates and dispatch (fake HTTP only)."""

import copy
import json

import pandas as pd
import pytest

from src.data.metadata import extract_metadata
from src.evaluation import jev_provenance as prov
from src.jev import liquid_hosted as liquid
from src.jev.cache import ResponseCache
from src.jev.extraction import (
    AccountCreditError,
    DatasetExtractor,
    ExtractionLimits,
    RejectedRetryable,
)
from src.jev.provider import api_preview, response_features
from src.jev.requests import IntendedRequest, build_requests
from src.utils.config import load_config
from src.utils.serialization import canonical_json

PINNED = "d1.2-ultra@2026-11-01"


@pytest.fixture(autouse=True)
def _no_real_liquid_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("LIQUID_API_KEY", raising=False)
    monkeypatch.setattr(liquid, "LOCAL_CREDENTIAL_PATH", tmp_path / "absent_liquid.json")


def _request(task, targets):
    frame = pd.DataFrame({"text": ["Example evidence", "More text", "Third text"], "y": targets})
    metadata = extract_metadata(frame, {"target": "y", "task_type": task}, dataset_id="fixture",
                                dataset_name="Fixture", text_columns=["text"])
    return build_requests(frame.drop(columns="y"), metadata, 0, load_config())[0]


TASKS = {"binary": ("cls", [0, 1, 0]), "multiclass": ("cls", [0, 1, 2]), "regression": ("reg", [0.1, 2.0, 4.3])}


def _candidate(**changes):
    cand = copy.deepcopy(liquid.candidate())
    cand["version"] = PINNED
    for key, value in changes.items():
        cand[key] = value
    return cand


def _ready_candidate():
    cand = _candidate(enabled=True, account_created=True, budget_authorised=True)
    review = cand["live_extraction"]
    review.update(enabled=True, pilot_authorization_manifest="signed.json", max_pilot_requests=4,
                  dataset_budget_allocations_usd={"fixture": 0.01}, campaign_spending_limit_usd=0.01,
                  max_billed_input_tokens_per_attempt=65536, requests_per_second=1,
                  input_tokens_per_second=100000)
    review.update({flag: True for flag in liquid.READINESS_FLAGS})
    return cand


def _bound(kind):
    return liquid.bind(_request(*TASKS[kind]), _candidate())


def _answer(request, *, shuffle=False):
    question = liquid.api_preview(request)["questions"]["semantic"]
    if question["type"] == "noul":
        return {"type": "noul", "noul": 0.73}
    if question["type"] == "choice":
        keys = list(question["criteria"])
        values = [0.6, 0.3, 0.1][: len(keys)] if len(keys) == 3 else [0.2, 0.8]
        probabilities = dict(zip(keys, values))
        if shuffle:
            probabilities = dict(reversed(list(probabilities.items())))
        return {"type": "choice", "choice": max(probabilities, key=probabilities.get),
                "probabilities": probabilities, "confidence": 0.5}
    probabilities = {str(i): (1.0 if i == 6 else 0.0) for i in range(9)}
    return {"type": "score", "score": 6.0, "probabilities": probabilities, "confidence": 1.0,
            "legend": {str(i): c for i, c in enumerate(question["criteria"])}}


def _response(request, **overrides):
    body = {"model": "d1.2-ultra", "answers": {"semantic": _answer(request)},
            "usage": {"input_tokens": 120, "output_tokens": 0}}
    body.update(overrides)
    return body


class FakeHttp:
    def __init__(self, post=None, release="2026-11-01"):
        self.calls, self.post, self.release = [], post, release

    def request(self, method, url, headers, body=None):
        self.calls.append((method, url, headers, body))
        if method == "GET":
            return 200, {}, json.dumps({"models": [{"name": "d1.2", "release_date": "2026-10-01"},
                                                   {"name": "d1.2-ultra", "release_date": self.release}]}).encode()
        return self.post(json.loads(body))

    def close(self):
        pass


@pytest.mark.parametrize("kind", TASKS)
def test_request_mapping_is_the_shared_semantic_question(kind):
    original = _request(*TASKS[kind])
    bound = liquid.bind(original, _candidate())
    body = liquid.api_preview(bound)
    jev = api_preview(original)
    assert body == {"model": "d1.2-ultra", "state": jev["state"], "questions": jev["questions"]}
    assert api_preview(bound) == body  # The shared dispatcher routes to the Liquid adapter.
    assert bound.payload_json == original.payload_json and bound.cache_key != original.cache_key
    assert {"binary": "noul", "multiclass": "choice", "regression": "score"}[kind] == body["questions"]["semantic"]["type"]
    if kind == "regression":
        assert len(body["questions"]["semantic"]["criteria"]) == 9


def test_unpinned_aliases_and_empty_text_are_refused():
    original = _request(*TASKS["binary"])
    for version in (None, "d1.2-ultra@latest", "d1.2-ultra@2026-02-31", "d1.2-ultra@20261101"):
        with pytest.raises(ValueError, match="Pin"):
            liquid.api_preview(liquid.bind(original, _candidate(version=version)))
    with pytest.raises(ValueError, match="aliases"):
        liquid.api_preview(liquid.bind(original, _candidate(model="d1.2-ultra:latest")))
    payload = json.loads(original.payload_json)
    payload["state"]["text"] = {"text": ""}
    with pytest.raises(ValueError, match="Empty text"):
        liquid.bind(IntendedRequest(canonical_json(payload), original.settings_json, original.provenance_json), _candidate())
    jev_only = json.loads(original.settings_json) | {"provider": "other"}
    with pytest.raises(ValueError, match="Jev only"):
        api_preview(IntendedRequest(original.payload_json, canonical_json(jev_only), original.provenance_json))


@pytest.mark.parametrize("kind", TASKS)
def test_response_decoding_keeps_canonical_class_order(kind):
    request = _bound(kind)
    body = _response(request)
    if kind == "multiclass":
        body["answers"]["semantic"] = _answer(request, shuffle=True)
    features = response_features(request, body)
    if kind == "binary":
        assert features == {"p_positive": 0.73}
    elif kind == "multiclass":
        assert list(features) == ["p_c000", "p_c001", "p_c002", "confidence"]
        assert [features[k] for k in ("p_c000", "p_c001", "p_c002")] == [0.6, 0.3, 0.1]
    else:
        assert [features[f"p_{i}"] for i in range(9)] == [0.0] * 6 + [1.0, 0.0, 0.0]
        assert features["expected_direction"] == 2.0


def test_decoding_is_strict_without_documented_rounding():
    request = _bound("multiclass")
    body = _response(request)
    body["answers"]["semantic"]["probabilities"]["c002"] = 0.11  # Sum 1.01; no envelope for Liquid.
    with pytest.raises(ValueError, match="sum to one"):
        response_features(request, body)
    with pytest.raises(ValueError, match="model"):
        response_features(request, _response(request, model="d1.2"))


def test_shipped_configuration_blocks_every_gate_without_network_or_key(monkeypatch):
    cand = liquid.candidate()
    gates = liquid.readiness(cand, full=True)
    assert cand["enabled"] is False and cand["budget_authorised"] is False and cand["account_required"] is True
    assert not any(gates.values())
    monkeypatch.setattr(liquid, "read_api_key", lambda *a: pytest.fail("key read"))
    http = FakeHttp()
    with pytest.raises(RuntimeError, match="Network extraction is disabled"):
        liquid.LiquidTransport(cand, http=http)
    with pytest.raises(RuntimeError, match="budget_authorised"):
        liquid.LiquidTransport(cand, allow_network=True, http=http)
    with pytest.raises(RuntimeError, match="--execute"):
        liquid.run(execute=False)
    with pytest.raises(RuntimeError, match="unmet gates"):
        liquid.run(execute=True)
    assert http.calls == []


def test_each_gate_alone_blocks_transport(monkeypatch):
    monkeypatch.setenv("LIQUID_API_KEY", "liquid_fake")
    ready = _ready_candidate()
    assert all(liquid.readiness(ready).values())
    for gate in liquid.readiness(ready):
        cand = copy.deepcopy(ready)
        if gate == "credential_present":
            monkeypatch.delenv("LIQUID_API_KEY")
        elif gate in {"candidate_enabled", "account_created", "budget_authorised"}:
            cand[{"candidate_enabled": "enabled"}.get(gate, gate)] = False
        elif gate == "version_pinned":
            cand["version"] = None
        elif gate == "live_extraction_enabled":
            cand["live_extraction"]["enabled"] = False
        elif gate == "dataset_allocations":
            cand["live_extraction"]["dataset_budget_allocations_usd"] = {}
        else:
            cand["live_extraction"][gate] = None if gate.endswith("manifest") else False
        http = FakeHttp()
        with pytest.raises(RuntimeError, match=gate):
            liquid.LiquidTransport(cand, allow_network=True, http=http)
        assert http.calls == []
        monkeypatch.setenv("LIQUID_API_KEY", "liquid_fake")


def test_transport_checks_served_release_before_any_paid_call(monkeypatch):
    monkeypatch.setenv("LIQUID_API_KEY", "liquid_fake")
    http = FakeHttp(post=lambda body: pytest.fail("paid call"), release="2026-12-01")
    with pytest.raises(ValueError, match="pinned version"):
        liquid.LiquidTransport(_ready_candidate(), allow_network=True, http=http)
    assert [c[0] for c in http.calls] == ["GET"]


def test_transport_maps_errors_without_provider_messages(monkeypatch):
    monkeypatch.setenv("LIQUID_API_KEY", "liquid_fake")
    request = _bound("binary")

    def status(code, error_code, headers=None):
        body = json.dumps({"error": {"message": "secret text", "code": error_code}}).encode()
        return FakeHttp(post=lambda _: (code, headers or {}, body))

    for http, expected in ((status(402, "insufficient_quota"), AccountCreditError),
                           (status(429, "rate_limit_exceeded", {"retry-after": "3"}), liquid.LiquidAPIError),
                           (status(422, None), liquid.LiquidAPIError)):
        client = liquid.LiquidTransport(_ready_candidate(), allow_network=True, http=http)
        with pytest.raises(expected) as info:
            client.evaluate(request)
        assert "secret" not in str(info.value)
    verified = _ready_candidate()
    verified["live_extraction"]["overload_rejections_unbilled_verified"] = True
    client = liquid.LiquidTransport(verified, allow_network=True,
                                    http=status(503, "server_is_overloaded", {"retry-after": "3"}))
    with pytest.raises(RejectedRetryable) as info:
        client.evaluate(request)
    assert info.value.retry_after_seconds == 3.0


class NoWait:
    def acquire(self):
        pass


def _extractor(tmp_path, client, budget=1.0):
    limits = ExtractionLimits(budget, 0.042, 65536, 10, 1e6)
    return DatasetExtractor(ResponseCache(tmp_path / "liquid.sqlite3", "fixture"), client, limits,
                            gate=NoWait(), validate_context=liquid.LiquidContextEnvelope(_ready_candidate()),
                            log=tmp_path / "liquid.log")


def test_dispatch_records_version_usage_and_raw_responses(tmp_path, monkeypatch):
    monkeypatch.setenv("LIQUID_API_KEY", "liquid_fake")
    request = _bound("regression")
    http = FakeHttp(post=lambda body: (200, {}, json.dumps(_response(request)).encode()))
    client = liquid.LiquidTransport(_ready_candidate(), allow_network=True, http=http)
    assert client.release == "2026-11-01"
    extractor = _extractor(tmp_path, client)
    assert extractor.run([request]) == {"completed": 1}
    assert [c[0] for c in http.calls] == ["GET", "GET", "POST", "GET"]
    method, url, headers, body = http.calls[2]
    assert (method, url) == ("POST", "https://api.liquid.ai/decisions/v1/systemone")
    assert headers["Authorization"] == "Bearer liquid_fake" and json.loads(body) == liquid.api_preview(request)
    assert extractor.cache.get(request, kind="verified_response")["model"] == "d1.2-ultra"
    with extractor.cache._connect() as db:
        spent, tokens = db.execute("SELECT spent_usd,input_tokens FROM extraction_jobs").fetchone()
    assert tokens == 120 and spent == pytest.approx(120 * 0.042 / 1e6)
    ledger = prov.version_ledger(tmp_path, provider="liquid_ultra")
    assert ledger.requested_version.tolist() == [PINNED] and not ledger.version_mismatch.any()
    assert ledger.version_status.tolist() == ["model_matches_release_unreported"]
    evidence, = ledger.catalog_release_evidence
    assert evidence["before_release"] == evidence["after_release"] == client.release
    assert evidence["matches_pin"] and evidence["source"] == "models_endpoint_release_date"
    assert extractor.evaluate(request) == "cache_hit" and len(http.calls) == 4
    assert prov.version_ledger(tmp_path).empty  # Jev ledger ignores Liquid rows.


def test_changed_catalog_before_decision_releases_unspent_reservation(tmp_path, monkeypatch):
    monkeypatch.setenv("LIQUID_API_KEY", "liquid_fake")
    http = FakeHttp(post=lambda body: pytest.fail("decision was dispatched"))
    client = liquid.LiquidTransport(_ready_candidate(), allow_network=True, http=http)
    http.release = "2026-12-01"
    extractor = _extractor(tmp_path, client)
    assert extractor.run([_bound("binary")]) == {"release_unverified": 1}
    with extractor.cache._connect() as db:
        assert db.execute("SELECT status,attempts,reserved_usd,spent_usd FROM extraction_jobs").fetchone() == (
            "pending", 0, 0, 0)
    assert [c[0] for c in http.calls] == ["GET", "GET"]


def test_unbilled_retry_rechecks_release_before_another_decision(tmp_path, monkeypatch):
    monkeypatch.setenv("LIQUID_API_KEY", "liquid_fake")
    cand, http = _ready_candidate(), FakeHttp()
    cand["live_extraction"]["overload_rejections_unbilled_verified"] = True

    def reject(body):
        http.release = "2026-12-01"
        return 429, {}, b'{"error":{"code":"rate_limit_exceeded"}}'

    http.post = reject
    extractor = _extractor(tmp_path, liquid.LiquidTransport(cand, allow_network=True, http=http))
    extractor.sleep = lambda _: None
    assert extractor.run([_bound("binary")]) == {"release_unverified": 1}
    with extractor.cache._connect() as db:
        assert db.execute("SELECT status,attempts,reserved_usd,spent_usd FROM extraction_jobs").fetchone() == (
            "pending", 1, 0, 0)
    assert [c[0] for c in http.calls] == ["GET", "GET", "POST", "GET"]


@pytest.mark.parametrize("after", ["2026-12-01", None])
def test_paid_decision_with_changed_or_unavailable_catalog_keeps_raw_without_replay(tmp_path, monkeypatch, after):
    monkeypatch.setenv("LIQUID_API_KEY", "liquid_fake")
    request = _bound("binary")
    raw = _response(request)
    http = FakeHttp()

    def decision(body):
        http.release = after
        return 200, {}, json.dumps(raw).encode()

    http.post = decision
    client = liquid.LiquidTransport(_ready_candidate(), allow_network=True, http=http)
    extractor = _extractor(tmp_path, client)
    assert extractor.run([request]) == {"release_unverified": 1}
    assert _extractor(tmp_path, client).evaluate(request) == "release_unverified"
    assert extractor.cache.get(request, kind="verified_response") is None
    assert [c[0] for c in http.calls] == ["GET", "GET", "POST", "GET"]
    with extractor.cache._connect() as db:
        saved, spent, reserved, evidence = db.execute(
            "SELECT raw_response_json,spent_usd,reserved_usd,release_evidence_json FROM extraction_jobs"
        ).fetchone()
    assert json.loads(saved) == raw and spent == pytest.approx(120 * 0.042 / 1e6) and reserved == 0
    evidence = json.loads(evidence)
    assert evidence["before_release"] == "2026-11-01" and evidence["after_release"] == after
    assert evidence["matches_pin"] is False
    ledger = prov.version_ledger(tmp_path, provider="liquid_ultra")
    assert ledger.kind.tolist() == ["saved_release_unverified"]
    assert ledger.catalog_release_evidence.tolist() == [evidence]
    # A separately reconciled usage hold may restore the raw reply to 'received'.
    with extractor.cache._connect() as db:
        db.execute("UPDATE extraction_jobs SET status='received'")
    assert extractor.evaluate(request) == "release_unverified"
    assert extractor.cache.get(request, kind="verified_response") is None and len(http.calls) == 4


def test_old_mapping_remains_decodable_and_cannot_satisfy_or_dispatch_new_work(tmp_path, monkeypatch):
    monkeypatch.setenv("LIQUID_API_KEY", "liquid_fake")
    current = _bound("binary")
    settings = json.loads(current.settings_json) | {"api_mapping_version": liquid.LEGACY_MAPPING_VERSION}
    old = IntendedRequest(current.payload_json, canonical_json(settings), current.provenance_json)
    raw = _response(current)
    assert response_features(old, raw) == {"p_positive": 0.73}
    http = FakeHttp(post=lambda _: pytest.fail("decision was dispatched"))
    extractor = _extractor(tmp_path, liquid.LiquidTransport(_ready_candidate(), allow_network=True, http=http))
    assert extractor.evaluate(old) == "release_unverified"
    extractor.cache.put(old, raw, kind="verified_response")
    assert extractor.cache.get(current, kind="verified_response") is None
    assert [c[0] for c in http.calls] == ["GET"]


def test_ambiguous_usage_keeps_raw_response_and_stops(tmp_path, monkeypatch):
    monkeypatch.setenv("LIQUID_API_KEY", "liquid_fake")
    request = _bound("binary")
    body = _response(request, usage={"output_tokens": 0})
    http = FakeHttp(post=lambda _: (200, {}, json.dumps(body).encode()))
    extractor = _extractor(tmp_path, liquid.LiquidTransport(_ready_candidate(), allow_network=True, http=http))
    assert extractor.run([request]) == {"ambiguous_usage": 1}
    with extractor.cache._connect() as db:
        raw, reserved = db.execute("SELECT raw_response_json,reserved_usd FROM extraction_jobs").fetchone()
    assert json.loads(raw) == body and reserved > 0


def test_budget_refusal_precedes_dispatch(tmp_path, monkeypatch):
    monkeypatch.setenv("LIQUID_API_KEY", "liquid_fake")
    http = FakeHttp(post=lambda _: pytest.fail("paid call"))
    client = liquid.LiquidTransport(_ready_candidate(), allow_network=True, http=http)
    extractor = _extractor(tmp_path, client, budget=0.001)  # Reservation 65536 tokens = 0.00275 USD.
    assert extractor.run([_bound("binary")]) == {"budget_stop": 1}
    assert [c[0] for c in http.calls] == ["GET"]


def test_context_envelope_refuses_oversized_text_without_truncation():
    cand = _ready_candidate()
    cand["max_text_bytes_per_question"] = 50
    with pytest.raises(ValueError, match="no truncation"):
        liquid.LiquidContextEnvelope(cand)(_bound("binary"))
