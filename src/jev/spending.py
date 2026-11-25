"""Reviewed revisions to cumulative Jev allocations; never reset paid usage."""

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

from src.utils.acceptance import reviewed_acceptance
from src.utils.serialization import canonical_json, digest, file_sha256

# Owner's cumulative Jev allowance approved on 5 November 2026 (including diagnostics).
JEV_CEILING_USD = 5.50


class _Job(NamedTuple):
    status: str
    attempts: int
    spent: float
    reserved: float
    owner: str | None
    error: str | None
    policy: str | None
    billed_tokens: int | None
    provider: str | None


class LedgerUsage(NamedTuple):
    request_key: str
    provider: str | None
    completed: bool
    spent: float
    reserved: float


_TERMINAL = {"completed", "received", "invalid_response", "billing_bound_exceeded", "release_unverified"}
_FIELDS = ("status", "attempts", "spent_usd", "reserved_usd", "owner_session", "error_type",
           "policy_sha256", "raw_response_json", "request_json")


def _has_table(db, name):
    return db.execute("SELECT 1 FROM sqlite_master WHERE name=? AND type='table'", (name,)).fetchone()


def _jobs(db, *, archived):
    """Fetch the billing fields with SQLite. Full prompts and responses stay on disk."""
    table = "imported_extraction_records" if archived else "extraction_jobs"
    if not _has_table(db, table):
        return
    if archived:
        fields = [f"json_extract(record_json, '$.{name}')" for name in _FIELDS]
        key, where = "record_key", "WHERE table_name='extraction_jobs'"
    else:
        columns = {row[1] for row in db.execute("PRAGMA table_info(extraction_jobs)")}
        fields = [name if name in columns else "NULL" for name in _FIELDS]
        key, where = "request_key", ""
    for i in (1, 2, 3):
        fields[i] = f"coalesce({fields[i]}, 0)"
    fields[-2] = f"json_extract({fields[-2]}, '$.usage.input_tokens')"
    fields[-1] = f"json_extract({fields[-1]}, '$.settings.provider')"
    for row in db.execute(f"SELECT {key}, {','.join(fields)} FROM {table} {where}"):
        yield row[0], _Job(*row[1:])


def _job_record(row):
    raw = json.loads(row["raw_response_json"]) if row.get("raw_response_json") else {}
    request = json.loads(row["request_json"]) if row.get("request_json") else {}
    return _Job(row["status"], row["attempts"], row["spent_usd"], row["reserved_usd"],
                row.get("owner_session"), row.get("error_type"), row.get("policy_sha256"),
                (raw.get("usage") or {}).get("input_tokens"), request.get("settings", {}).get("provider"))


def _recoveries(db, *, include_archived):
    """Import a review for accounting only after verifying its before-state, action and result."""
    from src.jev.recovery import _resolved_state

    queries = []
    if _has_table(db, "extraction_recoveries"):
        queries.append("SELECT recovery_id,request_key,record_json,record_sha256 FROM extraction_recoveries")
    if include_archived and _has_table(db, "imported_extraction_records"):
        queries.append("""SELECT record_key,json_extract(record_json,'$.request_key'),
            json_extract(record_json,'$.record_json'),json_extract(record_json,'$.record_sha256')
            FROM imported_extraction_records WHERE table_name='extraction_recoveries'""")
    out = defaultdict(dict)
    for query in queries:
        for identifier, key, raw, sha in db.execute(query):
            record = json.loads(raw)
            before, after, approval = record["before"], record["after"], record["approval"]
            dataset = json.loads(before["request_json"]).get("provenance", {}).get("dataset_id")
            if (digest(record) != sha or record["approval_sha256"] != identifier
                    or before["request_key"] != key or after["request_key"] != key
                    or approval.get("stage") != "jev_request_recovery" or approval.get("accepted") is not True
                    or not approval.get("reviewed_by") or not approval.get("evidence")
                    or approval.get("dataset_id") != dataset or approval.get("request_key") != key
                    or approval.get("expected_job_sha256") != digest(before)
                    or after != _resolved_state(before, approval)):
                raise ValueError("Stored billing recovery differs from its approved job state")
            old, new = _job_record(before), _job_record(after)
            if old in out[key] and out[key][old] != new:
                raise ValueError("Conflicting reviews of the same Jev dispatch")
            out[key][old] = new
    return out


def _same_progress(earlier, later):
    """Can this job state account for the earlier observation of the same physical dispatch?"""
    if earlier == later:
        return True
    if (earlier.owner is None or earlier.owner != later.owner or earlier.attempts > later.attempts
            or earlier.spent > later.spent + 1e-12):
        return False
    if earlier.status in _TERMINAL:
        return later.status in _TERMINAL and earlier.attempts == later.attempts
    if earlier.status == "sending":
        return True  # a later attempt under the same owner follows an unbilled rejection
    return earlier.status == later.status or (earlier.status == "pending" and earlier.reserved == 0)


def _prices(db):
    prices = {}
    if _has_table(db, "extraction_policy_history"):
        for sha, raw in db.execute("SELECT policy_sha256,settings_json FROM extraction_policy_history"):
            prices[sha] = json.loads(raw)["price"]
    if _has_table(db, "imported_extraction_records"):
        for sha, raw in db.execute("""SELECT json_extract(record_json,'$.policy_sha256'),
                json_extract(record_json,'$.settings_json') FROM imported_extraction_records
                WHERE table_name='extraction_policy_history'"""):
            prices[sha] = json.loads(raw)["price"]
    return prices


def _has_billing_archives(db):
    return (_has_table(db, "imported_extraction_records")
            and db.execute("SELECT 1 FROM imported_extraction_records WHERE table_name IN "
                           "('extraction_jobs', 'extraction_recoveries') LIMIT 1").fetchone())


def ledger_jobs(db):
    """Yield per-request provider, status, known charges and remaining holds over all snapshots."""
    if not _has_billing_archives(db):
        for key, state in _jobs(db, archived=False):
            yield LedgerUsage(key, state.provider, state.status == "completed", state.spent, state.reserved)
        return
    states = defaultdict(set)
    for archived in (True, False):
        for key, state in _jobs(db, archived=archived):
            states[key].add(state)
    recoveries = _recoveries(db, include_archived=True)
    prices = _prices(db)
    for key in states.keys() | recoveries.keys():
        observed = states[key]
        reviewed = recoveries.get(key, {})
        observed.update(reviewed)
        observed.update(reviewed.values())
        # Every settled attempt has an exact owner/before-state. The same reviewed
        # recovery may have been archived in many snapshots.
        confirmed = sum(after.spent - before.spent for before, after in reviewed.items())
        charges, holds = {}, {}
        for state in observed:
            if (state.reserved and not any(_same_progress(state, before) for before in reviewed)
                    and not any(_same_progress(state, other) and not other.reserved for other in observed)):
                holds[state.owner] = max(holds.get(state.owner, 0), state.reserved)
            if state.owner and state.status in _TERMINAL:
                if state.policy in prices and type(state.billed_tokens) is int and state.billed_tokens >= 0:
                    charges[state.owner] = state.billed_tokens * prices[state.policy] / 1e6
                else:
                    charges.setdefault(state.owner, None)
        # Successful responses from independent owners prove independent calls,
        # even when restoring their backups produced an identical semantic key.
        if len(charges) > 1 and None in charges.values():
            raise ValueError("Cannot reconcile charges of independently completed Jev dispatches without billing evidence")
        spent = max(max(s.spent for s in observed), confirmed + sum(v for v in charges.values() if v is not None))
        reserved = sum(holds.values())
        terminal = any(s.status == "completed" for s in observed)
        provider = next((s.provider for s in observed if s.provider), None)
        yield LedgerUsage(key, provider, terminal and not reserved, spent, reserved)


def committed_for_admission(db):
    """Read the cumulative dataset balance inside the transaction that grants the reservation."""
    if _has_billing_archives(db):
        return sum(row.spent + row.reserved for row in ledger_jobs(db))
    return db.execute("SELECT COALESCE(SUM(spent_usd + reserved_usd), 0) FROM extraction_jobs").fetchone()[0]


def require_authoritative_jobs(db):
    """Importing a snapshot must never authorize repeating a possibly billed call."""
    if not _has_table(db, "imported_extraction_records"):
        return
    recoveries = _recoveries(db, include_archived=False)
    live = dict(_jobs(db, archived=False))
    for key, prior in _jobs(db, archived=True):
        if not (prior.attempts or prior.spent or prior.reserved or prior.status not in {"pending", "completed"}):
            continue
        current = live.get(key)
        if current is not None:
            if _same_progress(prior, current):
                continue
            if any((_same_progress(prior, before) or prior == after) and current.spent + 1e-12 >= after.spent
                   for before, after in recoveries.get(key, {}).items()):
                continue
            if (prior.status == "pending" and not prior.reserved
                    and current.attempts >= prior.attempts and current.spent + 1e-12 >= prior.spent):
                continue
        raise ValueError("Imported paid job lacks a reconciled authoritative spending ledger. "
                         "Restore the dataset's original extraction ledger before paid dispatch.")


def diagnostic_spending(cache_folder):
    """Count each permanently recorded smoke or robustness round once, including holds."""
    for name in ("jev_smoke", "jev_robustness", "jev_dataset_robustness"):
        for state in sorted((Path(cache_folder) / "_control" / name).rglob("state.json")):
            record = json.loads(state.read_text(encoding="utf-8"))
            attempts = record["attempts"]
            spent = sum(row.get("charged_usd", 0) for row in attempts)
            held = record.get("historical_reserved_usd", 0) + sum(
                row.get("reserved_usd", 0) for row in attempts)
            yield state, spent, held


def replay_spending(folder):
    from src.jev.prefix_pilot import ledger_balances

    for path in sorted(folder.glob("*/*/*.sqlite3")) if folder is not None else ():
        yield ledger_balances(path)


def campaign_committed(cache_folder, *, replay_folder=None):
    from src.jev.prefix_pilot import ledger_total

    return (sum(ledger_total(path) for path in cache_folder.glob("*.sqlite3"))
            + sum(spent + held for spent, held in replay_spending(replay_folder))
            + sum(spent + held for _, spent, held in diagnostic_spending(cache_folder)))


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
        require_authoritative_jobs(db)
        prior = db.execute("SELECT settings_json FROM extraction_policy WHERE id=1").fetchone()
        previous = json.loads(prior[0]) if prior else None
        previous_hash = digest(previous) if previous is not None else None
        job_columns = {row[1] for row in db.execute("PRAGMA table_info(extraction_jobs)")}
        if "policy_sha256" not in job_columns:
            db.execute("ALTER TABLE extraction_jobs ADD COLUMN policy_sha256 TEXT")
        if "release_evidence_json" not in job_columns:
            db.execute("ALTER TABLE extraction_jobs ADD COLUMN release_evidence_json TEXT")
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
        if committed_for_admission(db) > limits.dataset_budget_usd + 1e-12:
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
