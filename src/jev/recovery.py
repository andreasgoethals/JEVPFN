"""Offline, reviewed reconciliation of stopped Jev requests; never call a provider.

Inspect with ``python -m src.jev.recovery DATASET_ID``. A review file binds one exact
job-state hash and checksummed evidence. ``--review FILE`` previews; add ``--apply
--writers-stopped`` to record the resolution atomically. Cumulative charges and raw
responses are preserved, including the previous state and reviewer in permanent history.
"""

import argparse
import json
import math
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from src.jev.requests import IntendedRequest
from src.utils import paths
from src.utils.acceptance import reviewed_acceptance
from src.utils.locking import file_lock
from src.utils.serialization import canonical_json, digest, file_sha256

RECOVERY_SQL = """CREATE TABLE IF NOT EXISTS extraction_recoveries (
    recovery_id TEXT PRIMARY KEY, request_key TEXT NOT NULL, record_json TEXT NOT NULL,
    record_sha256 TEXT NOT NULL, created_utc TEXT NOT NULL)"""


def _job(db, key):
    cursor = db.execute("SELECT * FROM extraction_jobs WHERE request_key=?", (key,))
    row = cursor.fetchone()
    if row is None:
        raise ValueError("Unknown extraction request")
    return dict(zip([c[0] for c in cursor.description], row))


def inspect(path):
    """Return stopped job identities/accounting only; no raw text, secrets or writes."""
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='extraction_jobs'").fetchone():
            return []
        result = []
        for (key,) in db.execute(
            "SELECT request_key FROM extraction_jobs WHERE status!='completed'"
        ):
            job = _job(db, key)
            result.append(
                {
                    k: job[k]
                    for k in (
                        "request_key",
                        "status",
                        "attempts",
                        "reserved_usd",
                        "spent_usd",
                        "error_type",
                    )
                }
                | {
                    "expected_job_sha256": digest(job),
                    "raw_response_available": bool(job["raw_response_json"]),
                }
            )
        return result


def _resolved_state(before, approval):
    after = dict(before)
    action = approval.get("action")
    if action == "renew_rejected_attempts":
        if not (
            before["status"] == "pending"
            and before["error_type"] == "RejectedRetryable"
            and before["reserved_usd"] == 0
            and not before["raw_response_json"]
            and before["attempts"] > 0
        ):
            raise ValueError("Only verified unbilled rejections can renew attempts")
        after.update(attempts=0, owner_session=None, error_type=None)
    elif action == "reconcile_billing":
        if before["status"] not in {"sending", "ambiguous", "ambiguous_usage"}:
            raise ValueError("Only unresolved dispatch/usage states need billing reconciliation")
        charge, tokens = approval.get("charged_usd"), approval.get("input_tokens")
        if (
            isinstance(charge, bool)
            or not isinstance(charge, (int, float))
            or not math.isfinite(charge)
            or charge < 0
        ):
            raise ValueError("Provider-confirmed nonnegative charged_usd is required")
        if tokens is not None and (type(tokens) is not int or tokens < 0):
            raise ValueError("Confirmed input_tokens must be nonnegative or null")
        retry = approval.get("retry_if_response_missing")
        if type(retry) is not bool:
            raise ValueError("Explicit retry_if_response_missing true/false is required")
        after.update(
            reserved_usd=0,
            spent_usd=before["spent_usd"] + charge,
            input_tokens=((before["input_tokens"] or 0) + tokens)
            if tokens is not None
            else before["input_tokens"],
            status="received"
            if before["raw_response_json"]
            else ("pending" if retry else "resolved_no_response"),
            attempts=0,
            owner_session=None,
            error_type=None,
        )
    elif action == "retry_resolved_without_response":
        if (
            before["status"] != "resolved_no_response"
            or before["reserved_usd"]
            or before["raw_response_json"]
        ):
            raise ValueError("Only a settled request without a response can be reopened")
        after.update(status="pending", attempts=0, owner_session=None, error_type=None)
    else:
        raise ValueError("Unknown recovery action")
    return after


def resolve(path, review, *, apply=False, writers_stopped=False):
    """Compare-and-swap one reviewed job state; repeat approval is idempotent."""
    path = Path(path).resolve(strict=True)
    if apply and not writers_stopped:
        raise ValueError("Stop writers and supply --writers-stopped before applying recovery")
    # Same lock as hosted extraction and cache merging. No recovery during a paid call.
    with file_lock(path.with_suffix(".extraction.lock"), timeout=1):
        approval = reviewed_acceptance(review, stage="jev_request_recovery")
        recovery_id = file_sha256(Path(review))
        uri = path.as_uri() + ("?mode=rw" if apply else "?mode=ro")
        with closing(sqlite3.connect(uri, uri=True)) as db, db:
            db.execute("BEGIN IMMEDIATE" if apply else "BEGIN")
            dataset = db.execute("SELECT dataset_id FROM dataset_identity").fetchall()
            if dataset != [(approval.get("dataset_id"),)]:
                raise ValueError("Recovery approval belongs to a different dataset")
            if db.execute(
                "SELECT 1 FROM sqlite_master WHERE name='extraction_recoveries'"
            ).fetchone():
                prior = db.execute(
                    "SELECT record_json,record_sha256 FROM extraction_recoveries WHERE recovery_id=?",
                    (recovery_id,),
                ).fetchone()
                if prior:
                    if digest(json.loads(prior[0])) != prior[1]:
                        raise ValueError("Recovery history checksum mismatch")
                    return {"status": "already_applied", "recovery_id": recovery_id}
            key = approval.get("request_key")
            before = _job(db, key)
            request = IntendedRequest.from_json(before["request_json"])
            if (
                request.cache_key != key
                or json.loads(request.provenance_json)["dataset_id"] != dataset[0][0]
            ):
                raise ValueError("Recovery job request identity differs")
            if digest(before) != approval.get("expected_job_sha256"):
                raise ValueError("Job state changed; inspect and review again")
            after = _resolved_state(before, approval)
            if apply:
                record = {
                    "before": before,
                    "after": after,
                    "approval": approval,
                    "approval_sha256": recovery_id,
                    "created_utc": datetime.now(UTC).isoformat(),
                }
                db.execute(RECOVERY_SQL)
                db.execute(
                    "INSERT INTO extraction_recoveries VALUES (?,?,?,?,?)",
                    (
                        recovery_id,
                        key,
                        canonical_json(record),
                        digest(record),
                        record["created_utc"],
                    ),
                )
                columns = (
                    "status",
                    "attempts",
                    "reserved_usd",
                    "spent_usd",
                    "input_tokens",
                    "error_type",
                    "owner_session",
                )
                db.execute(
                    "UPDATE extraction_jobs SET "
                    + ",".join(f"{c}=?" for c in columns)
                    + " WHERE request_key=?",
                    (*(after[c] for c in columns), key),
                )
    return {
        "status": "applied" if apply else "preview",
        "recovery_id": recovery_id,
        "request_key": key,
        "before_status": before["status"],
        "after_status": after["status"],
        "spent_usd": after["spent_usd"],
        "reserved_usd": after["reserved_usd"],
        "inference_performed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_id")
    parser.add_argument("--review", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--writers-stopped", action="store_true")
    args = parser.parse_args()
    if args.apply and not args.review:
        parser.error("--apply requires --review")
    path = paths.dataset_cache_path(args.dataset_id)
    result = (
        resolve(path, args.review, apply=args.apply, writers_stopped=args.writers_stopped)
        if args.review
        else inspect(path)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
