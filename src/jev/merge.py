"""Lossless, transactional union of dataset caches; never replace a paid response.

Snapshots include SQLite WAL contents. Operational spending/reservation tables are
archived as evidence, never adopted as a new machine's extraction queue or budget.
Conflicts roll back the entire dataset merge. Source files are opened read-only.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from src.jev.cache import ResponseCache
from src.jev.features import load_block
from src.jev.requests import IntendedRequest
from src.utils import paths
from src.utils.locking import file_lock
from src.utils.serialization import canonical_json, digest, file_sha256

ARTIFACT_SQL = """CREATE TABLE IF NOT EXISTS feature_artifacts (
    artifact_id TEXT PRIMARY KEY, manifest_json TEXT NOT NULL,
    features_parquet BLOB NOT NULL, requests_parquet BLOB NOT NULL)"""
LEDGER_SQL = """CREATE TABLE IF NOT EXISTS imported_extraction_records (
    snapshot_sha256 TEXT NOT NULL, table_name TEXT NOT NULL,
    record_key TEXT NOT NULL, record_json TEXT NOT NULL,
    PRIMARY KEY(snapshot_sha256,table_name,record_key))"""
TABLES = {
    "dataset_identity",
    "responses",
    "origins",
    "feature_artifacts",
    "extraction_policy",
    "extraction_policy_history",
    "extraction_jobs",
    "imported_extraction_records",
    "failures",
}


def snapshot(source: Path, destination: Path) -> Path:
    """SQLite's backup API includes committed WAL pages, unlike copying the main file."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_file() or destination.exists() or source == destination:
        raise ValueError("Snapshot needs an existing source and a new destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with (
            closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as src,
            closing(sqlite3.connect(destination)) as dst,
        ):
            src.backup(dst)
            if dst.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise ValueError("SQLite snapshot integrity check failed")
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination


def validate_cache(source: Path) -> str:
    """Check identities, provenance, blobs and referenced responses before any merge."""
    with closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("PRAGMA trusted_schema=OFF")
        if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise ValueError("Invalid SQLite cache")
        schema = db.execute(
            "SELECT type,name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
        ).fetchall()
        tables = {name for kind, name in schema if kind == "table"}
        if tables - TABLES or any(kind != "table" for kind, _ in schema):
            raise ValueError("Unknown cache schema; refusing automatic merge")
        expected = {
            "failures": [
                "event_id",
                "request_key",
                "payload_json",
                "settings_json",
                "error_type",
                "created_at",
            ],
            "dataset_identity": ["dataset_id"],
            "responses": [
                "request_key",
                "kind",
                "payload_json",
                "settings_json",
                "response_json",
                "created_at",
            ],
            "origins": [
                "request_key",
                "kind",
                "provenance_hash",
                "provenance_json",
                "first_seen_at",
            ],
            "feature_artifacts": [
                "artifact_id",
                "manifest_json",
                "features_parquet",
                "requests_parquet",
            ],
            "imported_extraction_records": [
                "snapshot_sha256",
                "table_name",
                "record_key",
                "record_json",
            ],
            "extraction_policy_history": [
                "revision",
                "policy_sha256",
                "previous_policy_sha256",
                "settings_json",
                "authorization_json",
                "created_utc",
            ],
        }
        if not {"dataset_identity", "responses", "origins"} <= tables:
            raise ValueError("Incomplete cache schema")
        for table in tables & expected.keys():
            if [r[1] for r in db.execute(f"PRAGMA table_info({table})")] != expected[table]:
                raise ValueError("Unknown cache column order/schema")
        bound = db.execute("SELECT dataset_id FROM dataset_identity").fetchall()
        if len(bound) != 1:
            raise ValueError("Merge requires exactly one dataset identity")
        dataset_id = paths.output_component(bound[0][0])
        for key, kind, payload, settings, response in db.execute(
            "SELECT request_key,kind,payload_json,settings_json,response_json FROM responses"
        ):
            request = IntendedRequest(payload, settings, canonical_json({"dataset_id": dataset_id}))
            ResponseCache._validate_kind(request, kind)
            if request.cache_key != key or not isinstance(json.loads(response), dict):
                raise ValueError("Invalid cached request identity/response")
        for _key, kind, hashed, encoded in db.execute(
            "SELECT request_key,kind,provenance_hash,provenance_json FROM origins"
        ):
            origin = json.loads(encoded)
            if digest(origin) != hashed or origin.get("dataset_id") != dataset_id:
                raise ValueError("Invalid cache provenance")
            # In-flight reservations may have origins before a response exists.
            if kind not in {"mock", "verified_response"}:
                raise ValueError("Invalid origin namespace")
        if "feature_artifacts" in tables:
            for (artifact_id,) in db.execute("SELECT artifact_id FROM feature_artifacts"):
                block = load_block(source, artifact_id)
                if block.manifest["identity"]["dataset_id"] != dataset_id:
                    raise ValueError("Feature artifact belongs to another dataset")
                for key in block.requests.request_key.dropna().unique():
                    if not db.execute(
                        "SELECT 1 FROM responses WHERE request_key=? AND kind='verified_response'",
                        (key,),
                    ).fetchone():
                        raise ValueError("Feature artifact has missing response references")
        if "failures" in tables:
            for _, key, payload, settings, error, _ in db.execute("SELECT * FROM failures"):
                request = IntendedRequest(
                    payload, settings, canonical_json({"dataset_id": dataset_id})
                )
                if request.cache_key != key or not isinstance(error, str) or not error:
                    raise ValueError("Invalid failed-request record")
    return dataset_id


def merge_cache(source: Path, destination: Path, *, backup_dir: Path | None = None) -> dict:
    """One dataset transaction, bounded-memory inserts and idempotent exact reuse.

    The incoming snapshot must be quiescent (e.g. exported transfer bundle). Existing
    local extraction budgets are never reset. Backups are made before any write.
    """
    source, destination = Path(source).resolve(), paths.require_project_output(Path(destination))
    destination = destination.resolve()
    if source == destination:
        raise ValueError("Cannot merge a cache into itself")
    dataset_id = validate_cache(source)
    source_hash = file_sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(destination.with_suffix(".extraction.lock"), timeout=1):
        backup = None
        if destination.exists():
            if validate_cache(destination) != dataset_id:
                raise ValueError("Caches belong to different datasets")
            backup_root = paths.require_project_output(
                backup_dir or destination.parent / "merge_backups"
            )
            # A fresh snapshot also captures WAL; its name is assigned from its content.
            from uuid import uuid4

            temporary = backup_root / f"{dataset_id}_{uuid4().hex}.sqlite3"
            snapshot(destination, temporary)
            backup = backup_root / f"{dataset_id}_{file_sha256(temporary)}.sqlite3"
            if backup.exists():
                temporary.unlink()
            else:
                temporary.rename(backup)
        cache = ResponseCache(destination, dataset_id)
        counts = {
            "responses_added": 0,
            "responses_reused": 0,
            "origins_added": 0,
            "artifacts_added": 0,
            "artifacts_reused": 0,
            "ledger_records_added": 0,
            "failures_added": 0,
        }
        with (
            closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as incoming,
            cache._connect() as db,
        ):
            db.execute("BEGIN IMMEDIATE")
            db.execute(ARTIFACT_SQL)
            db.execute(LEDGER_SQL)
            for row in incoming.execute("SELECT * FROM responses"):
                key, kind, payload, settings, response, _ = row
                prior = db.execute(
                    "SELECT payload_json,settings_json,response_json FROM responses WHERE request_key=? AND kind=?",
                    (key, kind),
                ).fetchone()
                if prior and [json.loads(s) for s in prior] != [
                    json.loads(s) for s in (payload, settings, response)
                ]:
                    raise ValueError(f"Conflicting cached response: {key}; no rows merged")
                if prior:
                    counts["responses_reused"] += 1
                else:
                    db.execute("INSERT INTO responses VALUES (?,?,?,?,?,?)", row)
                    counts["responses_added"] += 1
            for row in incoming.execute("SELECT * FROM origins"):
                counts["origins_added"] += db.execute(
                    "INSERT OR IGNORE INTO origins VALUES (?,?,?,?,?)", row
                ).rowcount
            tables = {
                r[0] for r in incoming.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if "failures" in tables:
                for row in incoming.execute("SELECT * FROM failures"):
                    prior = db.execute(
                        "SELECT * FROM failures WHERE event_id=?", (row[0],)
                    ).fetchone()
                    if prior is not None and prior != row:
                        raise ValueError("Conflicting failure evidence; no rows merged")
                    counts["failures_added"] += db.execute(
                        "INSERT OR IGNORE INTO failures VALUES (?,?,?,?,?,?)", row
                    ).rowcount
            if "feature_artifacts" in tables:
                for row in incoming.execute("SELECT * FROM feature_artifacts"):
                    artifact_id = row[0]
                    prior = db.execute(
                        "SELECT manifest_json FROM feature_artifacts WHERE artifact_id=?",
                        (artifact_id,),
                    ).fetchone()
                    if prior:
                        old, new = (
                            load_block(destination, artifact_id),
                            load_block(source, artifact_id),
                        )
                        if (
                            old.manifest["identity"] != new.manifest["identity"]
                            or not old.values.equals(new.values)
                            or not old.requests.equals(new.requests)
                        ):
                            raise ValueError(
                                f"Conflicting feature artifact: {artifact_id}; no rows merged"
                            )
                        counts["artifacts_reused"] += 1
                    else:
                        db.execute("INSERT INTO feature_artifacts VALUES (?,?,?,?)", row)
                        counts["artifacts_added"] += 1
            for table in ("extraction_policy", "extraction_policy_history", "extraction_jobs"):
                if table not in tables:
                    continue
                cursor = incoming.execute(f"SELECT * FROM {table}")
                columns = [item[0] for item in cursor.description]
                for row in cursor:
                    counts["ledger_records_added"] += db.execute(
                        "INSERT OR IGNORE INTO imported_extraction_records VALUES (?,?,?,?)",
                        (
                            source_hash,
                            table,
                            str(row[0]),
                            canonical_json(dict(zip(columns, row))),
                        ),
                    ).rowcount
            if "imported_extraction_records" in tables:
                for row in incoming.execute("SELECT * FROM imported_extraction_records"):
                    counts["ledger_records_added"] += db.execute(
                        "INSERT OR IGNORE INTO imported_extraction_records VALUES (?,?,?,?)",
                        row,
                    ).rowcount
        return {
            "dataset_id": dataset_id,
            "source_sha256": source_hash,
            "destination": str(destination),
            "backup": str(backup) if backup else None,
            **counts,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(merge_cache(args.source, args.destination), indent=2))


if __name__ == "__main__":
    main()
