"""Failure/recovery properties identified during the independent review."""

import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.cluster.gpu_contracts import synthetic_datasets
from src.cluster.semantic_contracts import exercise_semantics
from src.data.loaders import LoadedDataset
from src.data.metadata import extract_metadata
from src.evaluation import protocol
from src.evaluation.models import FoldInput
from src.jev.cache import ResponseCache
from src.jev.open_runtimes import probability_answer
from src.jev.provider import semantic_question
from src.jev.requests import build_requests
from src.jev.self_hosted import cached_evaluate, locked_datasets
from src.utils import paths
from src.utils.config import load_config
from src.utils.serialization import digest


def test_cleanup_preserves_durable_outputs(isolated_output):
    from src.utils.clean_run import main, roots

    protected = []
    for root in roots():
        for name in (
            "transfers/cache_backups/x.sqlite3",
            "3. experiment_0/results/token_audit/counts.sqlite3",
            "4. experiment_1/results/predictions/fold/record.json",
            "4. experiment_1/logs/run.log",
        ):
            file = root / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("protected")
            protected.append(file)
        figure = root / "figures/old.pdf"
        figure.parent.mkdir(parents=True, exist_ok=True)
        figure.write_bytes(b"pdf")
    main(["--clean"])
    assert all(p.read_text() == "protected" for p in protected)
    assert all(not (r / "figures/old.pdf").exists() for r in roots())


def test_predictors_cannot_start_on_a_vsc_login_node(isolated_output):
    from src.evaluation.models import estimator_factory

    with pytest.raises(RuntimeError, match="compute allocation"):
        estimator_factory("linear", next(synthetic_datasets()).metadata, {}, allow_model_runs=True)


def test_dates_keep_unseen_values_and_missingness_without_target_information():
    train = pd.DataFrame({"date": ["2025-01-01", "2025-01-02", None], "nominal": ["a", "b", "c"]})
    test = pd.DataFrame({"date": ["2025-02-01", None], "nominal": ["unseen", "a"]})
    encoded = FoldInput().fit(train).transform(test)
    assert np.isfinite(encoded.date.iloc[0])
    assert np.isnan(encoded.date.iloc[1])
    assert np.isnan(encoded.nominal.iloc[0])  # Unknown nominal categories remain missing.
    nanos = pd.DataFrame(
        {"date": ["1970-01-01 00:00:01.410000001", "1970-01-01 00:00:01.420000001"]}
    )
    assert FoldInput().fit(nanos).transform(nanos).date.nunique() == 2


def test_failed_input_persists_without_becoming_a_missing_text_response(tmp_path):
    dataset = next(synthetic_datasets())
    request = build_requests(dataset.features, dataset.metadata, 0, load_config())[0]
    cache = ResponseCache(tmp_path / "cache.sqlite3", dataset.metadata.dataset_id)
    calls = []

    def fail(request):
        calls.append(request.cache_key)
        raise ValueError("private error must not be written")

    with pytest.raises(ValueError):
        cached_evaluate(cache, SimpleNamespace(evaluate=fail), request)
    reopened = ResponseCache(cache.path, dataset.metadata.dataset_id)
    assert reopened.failure_count(request) == 1
    assert (
        cached_evaluate(reopened, SimpleNamespace(evaluate=fail), request, skip_failed=True)
        == "failed_recorded"
    )
    assert len(calls) == 1
    assert reopened.get(request, kind="verified_response") is None
    with reopened._connect() as db:
        row = db.execute("SELECT error_type FROM failures").fetchone()
    assert row == ("ValueError",)


def test_two_provider_workers_wait_and_both_complete(isolated_output):
    items = [{"dataset_id": "only"}]
    entered, completed, errors = threading.Event(), [], []

    def worker(provider, first):
        try:
            for item in locked_datasets(items, provider, timeout=3):
                if first:
                    entered.set()
                    time.sleep(0.25)
                completed.append((provider, item["dataset_id"]))
        except Exception as exc:
            errors.append(exc)

    a = threading.Thread(target=worker, args=("cloudflare", True))
    b = threading.Thread(target=worker, args=("intern", False))
    a.start()
    assert entered.wait(2)
    b.start()
    a.join(5)
    b.join(5)
    assert not errors and len(completed) == 2


@pytest.mark.parametrize("fault", [None, "binary_flip", "option_position", "score_reverse"])
def test_semantic_diagnostics_detect_mapping_faults(tmp_path, fault):
    settings = {
        "provider": "intern",
        "model": "fixture",
        "version": "pin",
        "api_mapping_version": "intern-self-hosted-v1",
    }

    def evaluate(request):
        payload = json.loads(request.payload_json)
        q = semantic_question(payload["question"])
        text = payload["state"]["text"]["body"]
        if q["type"] == "noul":
            p = 0.9 if " on." in text else 0.1
            probs = {"true": 1 - p if fault == "binary_flip" else p}
        elif q["type"] == "choice":
            key = next(k for k, v in q["criteria"].items() if "green" in v)
            if fault == "option_position":
                key = "c001"
            probs = {k: float(k == key) for k in q["criteria"]}
        else:
            integer = int(text.removesuffix(".").split()[-1])
            key = str(4 + (-integer if fault == "score_reverse" else integer))
            probs = {str(i): float(str(i) == key) for i in range(9)}
        return {
            "provider": "intern",
            "model": "fixture",
            "version": "pin",
            "settings_sha256": digest(settings),
            "answers": {"semantic": probability_answer(q, probs)},
        }

    cache = ResponseCache(tmp_path / "semantic.sqlite3", "EXP0_SEMANTICS")
    result = exercise_semantics(SimpleNamespace(settings=settings, evaluate=evaluate), cache)
    assert result["passed"] == (fault is None)
    assert len(result["probes"]) == 7


@pytest.mark.parametrize("binary", [False, True])
def test_cv_resumes_only_completed_folds_and_detects_corruption(isolated_output, binary):
    frame = pd.DataFrame(
        {
            "number": np.arange(30),
            "text": ["fixture"] * 30,
            "target": np.arange(30) % 2 if binary else np.arange(30.0),
        }
    )
    official = {"target": "target", "task_type": "cls" if binary else "reg"}
    meta = extract_metadata(
        frame, official, dataset_id="resume", dataset_name="Resume", text_columns=["text"]
    )
    dataset = LoadedDataset(frame, meta, official, {"data_identity": "fixed"}, [])
    matrix = frame[["number"]].set_axis([f"csv:{i}" for i in range(30)])
    calls = []
    state = {"stop": True}

    def factory():
        if state["stop"] and len(calls) == (8 if binary else 2):
            raise RuntimeError("interrupt after two completed folds")
        return SimpleNamespace(
            classes_=[0, 1],
            fit=lambda x, y: calls.append(len(x)),
            predict=lambda x: x.number.to_numpy(),
            predict_proba=lambda x: np.tile([0.4, 0.6], (len(x), 1)),
        )

    kwargs = dict(
        specification=protocol.shared_splits(dataset),
        model_id="fixture",
        checkpoint_id="pinned",
        feature_set="non_text",
        allow_model_runs=True,
        phase="experiment_0",
        estimator_settings={},
        feature_artifact_ids=[],
    )
    with pytest.raises(RuntimeError, match="interrupt"):
        protocol.evaluate_cv(dataset, matrix, factory, **kwargs)
    state["stop"] = False
    records = protocol.evaluate_cv(dataset, matrix, factory, **kwargs)
    assert len(calls) == (20 if binary else 5)
    assert len(records) == (10 if binary else 5)
    assert protocol.evaluate_cv(dataset, matrix, factory, **kwargs) == records
    assert len(calls) == (20 if binary else 5)
    body = json.loads(records[0].read_text())
    assert body["resources"]["peak_rss_bytes"] > 0
    from src.evaluation.results import collect

    summary, rows, means, _ = collect(("experiment_0",))
    assert summary["complete_arms"] == 1 and not means.empty
    assert set(rows.outer_fold) == set(range(5))
    assert any(name.startswith("text.unseen_joint.") for name in rows.metric)
    from src.evaluation.store import connect

    with connect(records[0].database) as db:
        db.execute("UPDATE records SET predictions=? WHERE id=?", (b"corrupt", records[0].key))
    with pytest.raises(ValueError, match="[Cc]orrupt"):
        protocol.evaluate_cv(dataset, matrix, factory, **kwargs)




def test_provenance_urls_are_not_sent_but_urls_in_actual_text_are_preserved():
    from src.jev.question_bank import build_question_bank, payload_from_text
    from src.utils.config import load_yaml

    dataset = next(synthetic_datasets())
    bank = build_question_bank([dataset.metadata], load_yaml("question_design"))
    entry = bank["datasets"][dataset.metadata.dataset_id]
    entry["variants"]["documented"] = {
        "version": "documented-task-v2",
        "question": entry["variants"]["accepted"]["question"],
        "documentary_definition": {
            "source_reference": "https://source.example/target-bound",
            "observation_unit": "one item",
        },
    }
    payload = payload_from_text(entry, "documented", {"text_a": "Visit https://real-text.example/"})
    assert "source_reference" not in payload["state"]["task"]["documentary_definition"]
    assert "https://real-text.example/" in payload["state"]["text"]["text_a"]
    assert "source_reference" in entry["variants"]["documented"]["documentary_definition"]


def test_request_rejection_has_bounded_exponential_backoff(isolated_output):
    from src.jev.extraction import DatasetExtractor, ExtractionLimits, RejectedRetryable

    dataset = next(synthetic_datasets())
    request = build_requests(dataset.features, dataset.metadata, 0, load_config())[0]
    cache = ResponseCache(isolated_output / "retry.sqlite3", dataset.metadata.dataset_id)
    delays, calls = [], []

    def reject(request):
        calls.append(request)
        raise RejectedRetryable()

    extractor = DatasetExtractor(
        cache,
        SimpleNamespace(evaluate=reject),
        ExtractionLimits(1, 1, 100, 100, 10000, max_attempts=3),
        gate=SimpleNamespace(acquire=lambda: None),
        validate_context=lambda r: None,
        sleep=delays.append,
    )
    assert extractor.evaluate(request) == "attempt_limit"
    assert delays == [1, 2] and len(calls) == 3
    assert extractor.evaluate(request) == "attempt_limit"
    assert len(calls) == 3
    assert extractor.log.is_relative_to(isolated_output)
    assert extractor.log.is_file()


@pytest.mark.parametrize("status", [200, 400, 413, 429])
def test_installed_typesafe_sdk_mock_transport_keeps_pins_and_no_hidden_retries(
    monkeypatch, status
):
    sdk = pytest.importorskip("typesafe_sdk")
    httpx = pytest.importorskip("httpx2")
    from src.jev.extraction import RejectedRetryable, TypeSafeTransport

    calls = []
    raw = {
        "model": "jev-1.13.0",
        "answers": {"semantic": {"type": "noul", "noul": 0.4}},
        "usage": {"input_tokens": 10},
    }

    def handler(request):
        assert "TypeSafe-Danger-Zone-Long-Context" not in request.headers
        calls.append(json.loads(request.content))
        return httpx.Response(
            status,
            json=raw if status == 200 else {"error": "rate limit"},
            headers={"retry-after": "3"},
        )

    cfg = load_config("feature_creation")
    live = cfg["live_extraction"]
    live.update(
        enabled=True,
        pilot_authorized=True,
        account_limits_verified=True,
        billing_bound_verified=True,
        context_policy_reviewed=True,
        price_verified=True,
        campaign_spending_limit_usd=1,
        usd_per_eur=1.0,
        exchange_rate_evidence="synthetic fixture, not a real exchange rate",
        dataset_budget_allocations_usd={"fixture": 1},
        max_billed_input_tokens_per_attempt=100,
        requests_per_second=1,
        input_tokens_per_second=100,
    )
    real_client = sdk.TypeSafeClient
    monkeypatch.setattr(
        sdk,
        "TypeSafeClient",
        lambda **kw: real_client(**kw, transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setenv("TYPESAFE_API_KEY", "local-mock-no-network")
    transport = TypeSafeTransport(cfg, allow_network=True)
    dataset = next(synthetic_datasets())
    request = build_requests(dataset.features, dataset.metadata, 0, cfg)[0]
    if status == 429:
        with pytest.raises(RejectedRetryable) as error:
            transport.evaluate(request)
        assert error.value.retry_after_seconds == 3
    elif status in {400, 413}:
        with pytest.raises(sdk.TypeSafeAPIError) as error:
            transport.evaluate(request)
        assert error.value.status == status  # No truncation, retry or new long-context request.
    else:
        assert transport.evaluate(request) == raw
    assert len(calls) == 1 and calls[0]["model"] == cfg["jev"]["model"]
    transport.client.close()


def test_acceptance_requires_existing_unchanged_evidence(tmp_path):
    from src.utils.acceptance import reviewed_acceptance
    from src.utils.serialization import file_sha256, write_json

    evidence = tmp_path / "test.json"
    evidence.write_text('{"status":"passed"}')
    acceptance = tmp_path / "acceptance.json"
    write_json(
        acceptance,
        {
            "accepted": True,
            "reviewed_by": "fixture",
            "stage": "after_extraction",
            "evidence": [{"path": "test.json", "sha256": file_sha256(evidence)}],
        },
    )
    assert reviewed_acceptance(acceptance, stage="after_extraction")["accepted"]
    evidence.write_text('{"status":"failed"}')
    with pytest.raises(ValueError, match="missing or changed"):
        reviewed_acceptance(acceptance, stage="after_extraction")


def test_runtime_changes_create_distinct_feature_artifacts_and_reads_do_not_grow_origins(
    isolated_output,
):
    from dataclasses import replace

    from src.jev.features import materialize_cached_block
    from src.utils.serialization import canonical_json

    dataset = next(synthetic_datasets())
    cfg = load_config("feature_creation")
    cfg["jev"]["representation"] = "per_column"
    cache = ResponseCache.for_dataset(dataset.metadata.dataset_id)
    requests = [
        r
        for i in range(len(dataset.frame))
        for r in build_requests(dataset.features, dataset.metadata, i, cfg)
    ]
    original = json.loads(requests[0].settings_json)
    identities = []
    for dtype in ("float32", "bfloat16"):
        settings = {**original, "runtime_dtype": dtype}
        for request in requests:
            cache.put(
                replace(request, settings_json=canonical_json(settings)),
                {
                    "model": settings["model"],
                    "answers": {"semantic": {"type": "noul", "noul": 0.5}},
                },
                kind="verified_response",
            )
        with cache._connect() as db:
            before = db.execute("SELECT COUNT(*) FROM origins").fetchone()[0]
        for _ in range(2):
            block = materialize_cached_block(dataset, cfg, cache, provider_settings=settings)
        with cache._connect() as db:
            assert db.execute("SELECT COUNT(*) FROM origins").fetchone()[0] == before
        identities.append(block.manifest["artifact_id"])
        assert block.manifest["identity"]["settings_sha256"] == digest(settings)
    assert identities[0] != identities[1]


def test_hosted_driver_resumes_prepared_unique_inputs_without_network(isolated_output, monkeypatch):
    from src.jev import hosted
    from src.jev.inputs import prepare_dataset_inputs
    from src.jev.question_bank import build_question_bank, save_question_bank
    from src.utils.config import load_yaml
    from src.utils.serialization import file_sha256

    dataset = next(synthetic_datasets())
    bank = build_question_bank([dataset.metadata], load_yaml("question_design"))
    save_question_bank(bank)
    prepared = prepare_dataset_inputs(dataset, bank, variant="accepted")
    from pathlib import Path

    source = Path(prepared["path"])
    identifier = dataset.metadata.dataset_id
    monkeypatch.setattr(
        hosted,
        "selected_inputs",
        lambda: [
            {
                "dataset_id": identifier,
                "input_set_id": prepared["input_set_id"],
                "input_file_sha256": file_sha256(source),
            }
        ],
    )
    monkeypatch.setattr(paths, "on_vsc", lambda: False)
    monkeypatch.setattr(paths, "dataset_question_path", lambda _: source)
    monkeypatch.setattr(paths, "dataset_cache_path", lambda _: isolated_output / "cache.sqlite3")
    monkeypatch.setattr(hosted, "RateGate", lambda _: SimpleNamespace(acquire=lambda: None))
    cfg = load_config("feature_creation")
    live = cfg["live_extraction"]
    live.update(
        enabled=True,
        campaign_spending_limit_usd=1,
        usd_per_eur=1.0,
        exchange_rate_evidence="synthetic fixture, not a real exchange rate",
        dataset_budget_allocations_usd={identifier: 1},
        max_billed_input_tokens_per_attempt=100,
        requests_per_second=10,
        input_tokens_per_second=1000,
        context_envelope={
            "verified": True,
            "model": cfg["jev"]["model"],
            "max_request_utf8_bytes": 100000,
            "max_billed_input_tokens": 100,
            "evidence": "synthetic fixture only",
        },
    )
    calls = []

    def evaluate(request):
        calls.append(request.cache_key)
        return {
            "model": cfg["jev"]["model"],
            "answers": {"semantic": {"type": "noul", "noul": 0.4}},
            "usage": {"input_tokens": 10},
        }

    def factory(*a, **kw):
        return SimpleNamespace(evaluate=evaluate)

    first = hosted.run(cfg, execute=True, transport_factory=factory)
    second = hosted.run(cfg, execute=True, transport_factory=factory)
    assert first["status"] == second["status"] == "completed_requires_review"
    assert len(calls) == len(set(calls))
    assert set(second["datasets"][0]["counts"]) == {"cache_hit"}
