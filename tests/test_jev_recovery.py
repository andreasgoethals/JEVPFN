"""Reviewed recovery never loses spending, repeats ambiguous calls, or repays for decoding."""

import json
from dataclasses import replace

import pandas as pd
import pytest

from src.data.metadata import extract_metadata
from src.jev.cache import ResponseCache
from src.jev.extraction import DatasetExtractor, ExtractionLimits, RejectedRetryable
from src.jev.merge import merge_cache, validate_cache
from src.jev.recovery import inspect, resolve
from src.jev.requests import build_requests
from src.utils import paths
from src.utils.config import load_config
from src.utils.serialization import file_sha256, write_json


def test_paid_rounded_multiclass_response_recovers_without_redispatch(extraction, monkeypatch):
    from src.jev import extraction as module
    from src.jev.provider import response_features
    from src.utils.serialization import canonical_json

    extractor, client, request = extraction
    payload = json.loads(request.payload_json)
    payload["question"] = {
        "kind": "choice",
        "text": "Which class is supported?",
        "allowed_answers": [1, 2, 3, 4],
    }
    request = replace(request, payload_json=canonical_json(payload))
    response = {
        "model": "jev-1.14.0",
        "usage": {"input_tokens": 10},
        "answers": {
            "semantic": {
                "type": "choice",
                "choice": "c002",
                "confidence": 0.12,
                "probabilities": dict(
                    zip([f"c{i:03d}" for i in range(4)], [0.05, 0.33, 0.34, 0.27])
                ),
            }
        },
    }

    def evaluate(_):
        client.calls += 1
        return response

    client.evaluate = evaluate

    def legacy_decoder(*_):
        raise ValueError("Legacy strict sum check")

    monkeypatch.setattr(module, "response_features", legacy_decoder)
    assert extractor.evaluate(request) == "invalid_response"
    monkeypatch.setattr(module, "response_features", response_features)
    assert extractor.evaluate(request) == "recovered_response"
    assert extractor.evaluate(request) == "cache_hit" and client.calls == 1
    with extractor.cache._connect() as db:
        status, spent, raw = db.execute(
            "SELECT status,spent_usd,raw_response_json FROM extraction_jobs"
        ).fetchone()
    assert status == "completed" and spent == pytest.approx(0.00001)
    assert json.loads(raw) == response


class Client:
    def __init__(self):
        self.calls, self.mode = 0, "ok"

    def evaluate(self, request):
        self.calls += 1
        if self.mode == "429":
            raise RejectedRetryable()
        if self.mode == "timeout":
            raise TimeoutError()
        response = {"model": "jev-1.14.0", "answers": {"semantic": {"type": "noul", "noul": 0.4}}}
        if self.mode != "missing_usage":
            response["usage"] = {"input_tokens": 10}
        return response


def test_mismatched_server_release_is_charged_recorded_and_never_reused(extraction, monkeypatch):
    from src.evaluation import jev_provenance

    extractor, client, request = extraction
    original = client.evaluate

    def wrong_release(record):
        return {**original(record), "model": "jev-1.15.0"}

    monkeypatch.setattr(client, "evaluate", wrong_release)
    assert extractor.evaluate(request) == extractor.evaluate(request) == "invalid_response"
    assert client.calls == 1 and extractor.cache.get(request, kind="verified_response") is None
    with extractor.cache._connect() as db:
        spent, held, billed, raw = db.execute(
            "SELECT spent_usd,reserved_usd,input_tokens,raw_response_json FROM extraction_jobs"
        ).fetchone()
    assert json.loads(raw)["model"] == "jev-1.15.0" and (held, billed) == (0, 10)
    assert spent == pytest.approx(.00001)
    ledger = jev_provenance.version_ledger(extractor.cache.path.parent)
    row, = ledger.to_dict("records")
    assert (row["requested_model"], row["requested_version"], row["reported_model"]) == (
        "jev-1.14.0", "1.14.0", "jev-1.15.0")
    assert row["kind"] == "saved_invalid_response" and row["version_mismatch"] and row["responses"] == 1
    assert row["first_response_utc"] is None
    upgraded = replace(request, settings_json=json.dumps({
        **json.loads(request.settings_json), "model": "jev-1.15.0", "version": "1.15.0",
    }))
    assert upgraded.cache_key != request.cache_key
    assert extractor.evaluate(upgraded) == "completed"
    assert extractor.evaluate(upgraded) == "cache_hit" and client.calls == 2
    assert extractor.observed_response_models() == [
        {"source": "api_reply", "reported_model": "jev-1.15.0", "count": 2},
        {"source": "cache_hit", "reported_model": "jev-1.15.0", "count": 1},
        {"source": "saved_reply", "reported_model": "jev-1.15.0", "count": 1},
    ]
    ledger = jev_provenance.version_ledger(extractor.cache.path.parent)
    assert dict(zip(ledger.kind, ledger.responses)) == {"verified_response": 1, "saved_invalid_response": 1}


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
    cache = ResponseCache(paths.dataset_cache_path("fixture").parent / "authoritative" / "fixture.sqlite3", "fixture")
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


def test_credit_stop_requires_review_then_new_session_can_resume(extraction, tmp_path):
    from src.jev.extraction import AccountCreditError

    extractor, client, request = extraction
    normal = client.evaluate

    def no_credit(_):
        client.calls += 1
        raise AccountCreditError(http_status=402)

    client.evaluate = no_credit
    assert extractor.evaluate(request) == "ambiguous" and client.calls == 1
    review = approval(extractor, tmp_path, "reconcile_billing", charged_usd=0,
                      input_tokens=0, retry_if_response_missing=True)
    resolve(extractor.cache.path, review, apply=True, writers_stopped=True)
    client.evaluate = normal
    resumed = DatasetExtractor(extractor.cache, client, extractor.limits, gate=Gate(),
                               validate_context=lambda _: None)
    assert resumed.evaluate(request) == "completed" and client.calls == 2
    assert resumed.evaluate(request) == "cache_hit" and client.calls == 2
    validate_cache(extractor.cache.path)


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
    from src.jev.merge import snapshot
    from src.jev.prefix_pilot import ledger_total
    from src.jev.spending import register_policy

    extractor, client, request = extraction
    client.mode = "timeout"
    assert extractor.evaluate(request) == "ambiguous"
    archived = snapshot(extractor.cache.path, tmp_path / "before-settlement.sqlite3")
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
    merge_cache(archived, extractor.cache.path, backup_destination=False)
    assert ledger_total(extractor.cache.path) == pytest.approx(charge)
    assert register_policy(extractor.cache, extractor.limits) == extractor.policy_sha256
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


def test_backup_after_review_cannot_erase_a_later_ambiguous_dispatch(extraction, tmp_path):
    from src.evaluation.study_audit import jev_spend
    from src.jev.merge import snapshot
    from src.jev.prefix_pilot import ledger_total

    first, client, request = extraction
    client.mode = "timeout"
    assert first.evaluate(request) == "ambiguous"
    first_hold = ledger_total(first.cache.path)
    signed = approval(first, tmp_path, "reconcile_billing", charged_usd=0, input_tokens=0,
                      retry_if_response_missing=True)
    resolve(first.cache.path, signed, apply=True, writers_stopped=True)
    restored = snapshot(first.cache.path, paths.dataset_cache_path("fixture"))
    assert ledger_total(restored) == 0
    second = DatasetExtractor(first.cache, client, first.limits, gate=Gate(), validate_context=lambda _: None)
    assert second.evaluate(request) == "ambiguous"
    merge_cache(first.cache.path, restored, backup_destination=False)
    assert ledger_total(restored) == pytest.approx(first_hold)
    assert jev_spend(restored.parent).reserved_usd.sum() == pytest.approx(first_hold)
    with pytest.raises(ValueError, match="authoritative spending ledger"):
        DatasetExtractor(ResponseCache(restored, "fixture"), client, first.limits, gate=Gate(),
                         validate_context=lambda _: None)
    assert client.calls == 2


@pytest.mark.parametrize("charge", [0, 0.00003])
def test_billing_review_in_archive_frees_only_its_hold_for_other_datasets(extraction, tmp_path, charge):
    from src.evaluation.study_audit import jev_spend
    from src.jev.prefix_pilot import ledger_total
    from src.jev.spending import campaign_committed, register_policy

    original, client, request = extraction
    archive = paths.dataset_cache_path("fixture")
    client.mode = "timeout"
    assert original.evaluate(request) == "ambiguous"
    merge_cache(original.cache.path, archive, backup_destination=False)
    assert ledger_total(archive) > charge
    signed = approval(original, tmp_path, "reconcile_billing", charged_usd=charge, input_tokens=0,
                      retry_if_response_missing=False)
    resolve(original.cache.path, signed, apply=True, writers_stopped=True)
    merge_cache(original.cache.path, archive, backup_destination=False)
    assert ledger_total(archive) == campaign_committed(archive.parent) == pytest.approx(charge)
    row = jev_spend(archive.parent).iloc[0]
    assert row.spent_usd == pytest.approx(charge) and row.reserved_usd == 0
    with pytest.raises(ValueError, match="authoritative spending ledger"):
        register_policy(ResponseCache(archive, "fixture"), original.limits)
    assert client.calls == 1


@pytest.mark.parametrize("over_token_bound", [False, True])
def test_distinct_billed_sessions_survive_merge_of_independent_restores(extraction, over_token_bound):
    from src.evaluation.study_audit import jev_spend
    from src.jev.merge import snapshot
    from src.jev.prefix_pilot import ledger_total
    from src.jev.spending import register_policy

    original, client, request = extraction
    restored = snapshot(original.cache.path, paths.dataset_cache_path("fixture"))
    other = DatasetExtractor(ResponseCache(restored, "fixture"), client, original.limits,
                             gate=Gate(), validate_context=lambda _: None)
    if over_token_bound:
        get_response = client.evaluate

        def too_many_tokens(request):
            response = get_response(request)
            response["usage"]["input_tokens"] = original.limits.max_billed_input_tokens + 1
            return response

        client.evaluate = too_many_tokens
        for worker in (original, other):
            with pytest.raises(RuntimeError, match="billing-token bound"):
                worker.evaluate(request)
    else:
        assert original.evaluate(request) == other.evaluate(request) == "completed"
    assert client.calls == 2
    one_charge = ledger_total(original.cache.path)
    merge_cache(original.cache.path, restored, backup_destination=False)
    merge_cache(original.cache.path, restored, backup_destination=False)
    assert ledger_total(restored) == pytest.approx(2 * one_charge)
    assert jev_spend(restored.parent).spent_usd.sum() == pytest.approx(2 * one_charge)
    with pytest.raises(ValueError, match="authoritative spending ledger"):
        register_policy(other.cache, original.limits)


def test_reservation_counts_separate_reviewed_charges_and_diagnostic_holds(extraction, tmp_path):
    from src.jev.merge import snapshot
    from src.jev.prefix_pilot import ledger_total
    from src.jev.spending import campaign_committed, policy_settings, register_policy

    first, client, request = extraction
    destination = paths.dataset_cache_path("fixture")
    snapshot(first.cache.path, destination)
    second = DatasetExtractor(ResponseCache(destination, "fixture"), client, first.limits,
                              gate=Gate(), validate_context=lambda _: None)
    charge = 0.00003
    client.mode = "timeout"
    for i, worker in enumerate((first, second)):
        assert worker.evaluate(request) == "ambiguous"
        review = approval(worker, tmp_path / str(i), "reconcile_billing", charged_usd=charge,
                          input_tokens=30, retry_if_response_missing=True)
        resolve(worker.cache.path, review, apply=True, writers_stopped=True)
    merge_cache(first.cache.path, destination, backup_destination=False)
    write_json(destination.parent / "_control/jev_smoke/fixture/state.json",
               {"attempts": [{"charged_usd": 0.00001, "reserved_usd": 0.00002}]})
    total = campaign_committed(destination.parent)
    assert ledger_total(destination) == pytest.approx(2 * charge)
    assert total == pytest.approx(3 * charge)
    reduced = replace(first.limits, dataset_budget_usd=1.5 * charge)
    evidence = tmp_path / "0" / "billing_evidence.json"
    revision = write_json(tmp_path / "revision.json", {
        "stage": "jev_spending_policy", "accepted": True, "reviewed_by": "fixture",
        "evidence": [{"path": str(evidence), "sha256": file_sha256(evidence)}],
        "policies": {"fixture": {"previous_policy_sha256": second.policy_sha256, "policy": policy_settings(reduced)}},
    })
    with pytest.raises(ValueError, match="below spending already recorded"):
        register_policy(second.cache, reduced, revision_manifest=revision)
    # Each approved retry needs room for its entire token reservation.
    reservation = first.limits.max_billed_input_tokens * first.limits.input_price_per_million / 1e6
    ceiling = total + reservation - charge / 2
    worker = DatasetExtractor(second.cache, client, first.limits, gate=Gate(), validate_context=lambda _: None,
                              campaign_available_usd=ceiling - (total - ledger_total(destination)))
    assert worker.evaluate(request) == "budget_stop"
    assert client.calls == 2 and campaign_committed(destination.parent) == pytest.approx(total)
    # At the exact boundary, one more dispatch is possible and its hold consumes all remaining room.
    worker.available_usd = 2 * charge + reservation
    assert worker.evaluate(request) == "ambiguous"
    assert client.calls == 3 and ledger_total(destination) == pytest.approx(2 * charge + reservation)


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
