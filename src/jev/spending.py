"""Reviewed revisions to cumulative Jev allocations; never reset paid usage."""

import json
from datetime import UTC, datetime
from pathlib import Path

from src.utils.acceptance import reviewed_acceptance
from src.utils.serialization import canonical_json, digest, file_sha256


def policy_settings(limits):
    return {
        "budget": limits.dataset_budget_usd,
        "price": limits.input_price_per_million,
        "token_bound": limits.max_billed_input_tokens,
    }


def register_policy(cache, limits, *, revision_manifest=None, stage="feature_creation"):
    """Pin first use; require checksummed review to change an existing allocation.

    Budgets are cumulative over this dataset's existing jobs, including pilot calls.
    History and attempt policy identities migrate transactionally from the original
    single-policy schema. No response, spending total or unresolved reservation is reset.
    """
    settings = policy_settings(limits)
    encoded, policy_hash = canonical_json(settings), digest(settings)
    with cache._connect() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS extraction_policy (
                id INTEGER PRIMARY KEY CHECK(id=1), settings_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS extraction_jobs (
                request_key TEXT PRIMARY KEY, request_json TEXT NOT NULL,
                status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                reserved_usd REAL NOT NULL DEFAULT 0, spent_usd REAL NOT NULL DEFAULT 0,
                input_tokens INTEGER, raw_response_json TEXT, error_type TEXT, owner_session TEXT);
            CREATE TABLE IF NOT EXISTS extraction_policy_history (
                revision INTEGER PRIMARY KEY, policy_sha256 TEXT NOT NULL,
                previous_policy_sha256 TEXT, settings_json TEXT NOT NULL,
                authorization_json TEXT NOT NULL, created_utc TEXT NOT NULL);
        """)
        db.execute("BEGIN IMMEDIATE")
        prior = db.execute("SELECT settings_json FROM extraction_policy WHERE id=1").fetchone()
        previous = json.loads(prior[0]) if prior else None
        previous_hash = digest(previous) if previous is not None else None
        if "policy_sha256" not in {
            row[1] for row in db.execute("PRAGMA table_info(extraction_jobs)")
        }:
            db.execute("ALTER TABLE extraction_jobs ADD COLUMN policy_sha256 TEXT")
        latest = db.execute(
            "SELECT policy_sha256 FROM extraction_policy_history ORDER BY revision DESC LIMIT 1"
        ).fetchone()
        if latest and latest[0] != previous_hash:
            raise ValueError("Active spending policy disagrees with its revision history")
        if prior and not latest:
            db.execute(
                "INSERT INTO extraction_policy_history(policy_sha256,previous_policy_sha256,settings_json,authorization_json,created_utc) VALUES (?,NULL,?,?,?)",
                (
                    previous_hash,
                    canonical_json(previous),
                    canonical_json({"kind": "legacy_allocation"}),
                    datetime.now(UTC).isoformat(),
                ),
            )
            db.execute(
                "UPDATE extraction_jobs SET policy_sha256=? WHERE policy_sha256 IS NULL",
                (previous_hash,),
            )
        if previous == settings:
            # Equivalent YAML/JSON numbers (1 vs 1.0) must reuse the stored identity.
            return previous_hash
        authorization = {"kind": "initial_allocation", "execution_stage": stage}
        if previous is not None:
            if not revision_manifest:
                raise ValueError(
                    "Extraction budget policy differs from its permanent allocation; "
                    "supply a reviewed spending_policy_revision_manifest"
                )
            approval = reviewed_acceptance(revision_manifest, stage="jev_spending_policy")
            expected = {"previous_policy_sha256": previous_hash, "policy": settings}
            if approval.get("policies", {}).get(cache.dataset_id) != expected:
                raise ValueError("Spending revision does not match this dataset's old/new policy")
            if db.execute(
                "SELECT 1 FROM extraction_jobs WHERE reserved_usd>0 OR status='sending' LIMIT 1"
            ).fetchone():
                raise ValueError(
                    "Resolve in-flight or ambiguous reservations before revising spending"
                )
            authorization = {
                "kind": "reviewed_revision",
                "execution_stage": stage,
                "reviewed_by": approval["reviewed_by"],
                "acceptance_sha256": file_sha256(Path(revision_manifest)),
                "evidence": approval["evidence"],
            }
        total = db.execute(
            "SELECT COALESCE(SUM(reserved_usd+spent_usd),0) FROM extraction_jobs"
        ).fetchone()[0]
        if total > limits.dataset_budget_usd + 1e-12:
            raise ValueError("Revised cumulative budget is below spending already recorded")
        db.execute(
            "INSERT INTO extraction_policy_history(policy_sha256,previous_policy_sha256,settings_json,authorization_json,created_utc) VALUES (?,?,?,?,?)",
            (
                policy_hash,
                previous_hash,
                encoded,
                canonical_json(authorization),
                datetime.now(UTC).isoformat(),
            ),
        )
        db.execute(
            "INSERT INTO extraction_policy VALUES (1,?) ON CONFLICT(id) DO UPDATE SET settings_json=excluded.settings_json",
            (encoded,),
        )
    return policy_hash
