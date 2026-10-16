"""Reviewed recovery never loses spending, repeats ambiguous calls, or repays for decoding."""

import json

import pandas as pd
import pytest

from src.data.metadata import extract_metadata
from src.jev.cache import ResponseCache
from src.jev.extraction import DatasetExtractor, ExtractionLimits, RejectedRetryable
from src.jev.merge import merge_cache, validate_cache
from src.jev.recovery import inspect, resolve
from src.jev.requests import build_requests
from src.utils.config import load_config
from src.utils.serialization import file_sha256, write_json


class Client:
    def __init__(self):
        self.calls, self.mode = 0, "ok"

    def evaluate(self, request):
        self.calls += 1
        if self.mode == "429":
            raise RejectedRetryable()
        if self.mode == "timeout":
            raise TimeoutError()
        response = {"model": "jev-1.13.0", "answers": {"semantic": {"type": "noul", "noul": 0.4}}}
        if self.mode != "missing_usage":
            response["usage"] = {"input_tokens": 10}
        return response


class Gate:
    def acquire(self):
        pass


@pytest.fixture
def extraction(tmp_path, isolated_output):
    frame = pd.DataFrame({"text": ["one", "two"], "y": [0, 1]})
    meta = extract_metadata(
        frame,
        {"task_type": "cls", "target": "y"},
        dataset_id="fixture",
        dataset_name="Fixture",
        text_columns=["text"],
    )
    request = build_requests(frame.drop(columns="y"), meta, 0, load_config())[0]
    client = Client()
    cache = ResponseCache(tmp_path / "fixture.sqlite3", "fixture")
    extractor = DatasetExtractor(
        cache,
        client,
        ExtractionLimits(1, 1, 100, 10, 1000),
        gate=Gate(),
        validate_context=lambda _: None,
        sleep=lambda _: None,
    )
    return extractor, client, request


def approval(extractor, tmp_path, action, **values):
    evidence = write_json(
        tmp_path / "billing_evidence.json", {"fixture": "provider reconciliation"}
    )
    job = inspect(extractor.cache.path)[0]
    body = {
        "stage": "jev_request_recovery",
        "accepted": True,
        "reviewed_by": "test reviewer",
        "evidence": [{"path": str(evidence), "sha256": file_sha256(evidence)}],
        "dataset_id": "fixture",
        "request_key": job["request_key"],
        "expected_job_sha256": job["expected_job_sha256"],
        "action": action,
        **values,
    }
    return write_json(tmp_path / f"review_{action}.json", body)


def test_three_unbilled_rejections_require_review_then_can_resume(extraction, tmp_path):
    extractor, client, request = extraction
    client.mode = "429"
    assert extractor.evaluate(request) == "attempt_limit" and client.calls == 3
    assert extractor.evaluate(request) == "attempt_limit" and client.calls == 3
    review = approval(extractor, tmp_path, "renew_rejected_attempts")
    before = file_sha256(extractor.cache.path)
    assert resolve(extractor.cache.path, review)["status"] == "preview"
    assert file_sha256(extractor.cache.path) == before
    resolve(extractor.cache.path, review, apply=True, writers_stopped=True)
    client.mode = "ok"
    assert extractor.evaluate(request) == "completed" and client.calls == 4
    with extractor.cache._connect() as db:
        assert db.execute("SELECT spent_usd FROM extraction_jobs").fetchone()[0] == pytest.approx(
            0.00001
        )
        record = json.loads(
            db.execute("SELECT record_json FROM extraction_recoveries").fetchone()[0]
        )
        assert record["before"]["attempts"] == 3 and record["after"]["attempts"] == 0


def test_decoder_repair_recovers_paid_response_without_another_call(extraction, monkeypatch):
    import src.jev.extraction as module

    extractor, client, request = extraction
    correct = module.response_features

    def broken(*args):
        raise ValueError("synthetic decoder defect")

    monkeypatch.setattr(module, "response_features", broken)
    assert extractor.evaluate(request) == "invalid_response"
    assert extractor.evaluate(request) == "invalid_response" and client.calls == 1
    monkeypatch.setattr(module, "response_features", correct)
    assert extractor.evaluate(request) == "recovered_response" and client.calls == 1
    with extractor.cache._connect() as db:
        assert db.execute("SELECT spent_usd FROM extraction_jobs").fetchone()[0] == pytest.approx(
            0.00001
        )


@pytest.mark.parametrize("charge", [0, 0.00003])
def test_ambiguous_billing_settlement_and_explicit_later_retry(extraction, tmp_path, charge):
    extractor, client, request = extraction
    client.mode = "timeout"
    assert extractor.evaluate(request) == "ambiguous"
    review = approval(
        extractor,
        tmp_path,
        "reconcile_billing",
        charged_usd=charge,
        input_tokens=None,
        retry_if_response_missing=False,
    )
    outcome = resolve(extractor.cache.path, review, apply=True, writers_stopped=True)
    assert outcome["spent_usd"] == charge and outcome["reserved_usd"] == 0
    assert extractor.run([request]) == {"resolved_no_response": 1} and client.calls == 1
    assert (
        resolve(extractor.cache.path, review, apply=True, writers_stopped=True)["status"]
        == "already_applied"
    )
    retry = approval(extractor, tmp_path, "retry_resolved_without_response")
    resolve(extractor.cache.path, retry, apply=True, writers_stopped=True)
    client.mode = "ok"
    assert extractor.evaluate(request) == "completed" and client.calls == 2
    with extractor.cache._connect() as db:
        assert db.execute("SELECT spent_usd FROM extraction_jobs").fetchone()[0] == pytest.approx(
            charge + 0.00001
        )


def test_missing_usage_is_reconciled_without_repeating_the_response(extraction, tmp_path):
    extractor, client, request = extraction
    client.mode = "missing_usage"
    assert extractor.evaluate(request) == "ambiguous_usage"
    review = approval(
        extractor,
        tmp_path,
        "reconcile_billing",
        charged_usd=0.00001,
        input_tokens=10,
        retry_if_response_missing=False,
    )
    resolve(extractor.cache.path, review, apply=True, writers_stopped=True)
    assert extractor.evaluate(request) == "recovered_response" and client.calls == 1
    with extractor.cache._connect() as db:
        assert db.execute(
            "SELECT spent_usd,input_tokens,reserved_usd FROM extraction_jobs"
        ).fetchone() == (0.00001, 10, 0)


@pytest.mark.parametrize(
    "change,match",
    [
        ({"dataset_id": "other"}, "different dataset"),
        ({"expected_job_sha256": "stale"}, "state changed"),
        ({"accepted": False}, "explicitly approve"),
        ({"action": "renew_rejected_attempts"}, "verified unbilled"),
        ({"charged_usd": -1}, "charged_usd"),
        ({"charged_usd": True}, "charged_usd"),
        ({"input_tokens": True}, "input_tokens"),
        ({"retry_if_response_missing": None}, "Explicit"),
    ],
)
def test_recovery_rejects_unsafe_or_stale_approval_without_changes(
    extraction, tmp_path, change, match
):
    extractor, client, request = extraction
    client.mode = "timeout"
    extractor.evaluate(request)
    review = approval(
        extractor,
        tmp_path,
        "reconcile_billing",
        charged_usd=0,
        input_tokens=0,
        retry_if_response_missing=False,
    )
    body = json.loads(review.read_text(encoding="utf-8"))
    write_json(review, {**body, **change})
    before = file_sha256(extractor.cache.path)
    with pytest.raises(ValueError, match=match):
        resolve(extractor.cache.path, review, apply=True, writers_stopped=True)
    assert file_sha256(extractor.cache.path) == before


def test_recovery_requires_stopped_writers_and_preserves_mergeable_history(extraction, tmp_path):
    extractor, client, request = extraction
    client.mode = "429"
    extractor.evaluate(request)
    review = approval(extractor, tmp_path, "renew_rejected_attempts")
    with pytest.raises(ValueError, match="writers-stopped"):
        resolve(extractor.cache.path, review, apply=True)
    resolve(extractor.cache.path, review, apply=True, writers_stopped=True)
    assert validate_cache(extractor.cache.path) == "fixture"
    from src.utils import paths

    destination = paths.dataset_cache_path("fixture")
    merged = merge_cache(extractor.cache.path, destination)
    assert merged["ledger_records_added"] >= 1
    with ResponseCache(destination, "fixture")._connect() as db:
        assert (
            db.execute(
                "SELECT COUNT(*) FROM imported_extraction_records WHERE table_name='extraction_recoveries'"
            ).fetchone()[0]
            == 1
        )
        assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='extraction_jobs'").fetchone()
