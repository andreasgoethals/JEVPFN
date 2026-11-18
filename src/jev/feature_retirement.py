"""Verified equivalent producer revisions for previously published numeric blocks.

Keep compatibility with existing SQLite readers. Historical receipts retain the
original identities; these two mappings only apply to byte-identical payloads.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing

REPLACEMENTS = {
    "177104d245f29857c1e3d8fdf1cf7c516be1ba3090203bac1e144fcb26e1591d":
        "70bf81c3b07118c7cf68934f2800eb22b5b8187637f89eb5f5fdee59157c8d57",
    "62c719aca2cbd39e15f03fb5d8bec6f9ef4c0376f842b87f315bbf911c6fe57c":
        "fc29e6bbe75bafe7e7cc072aaeb684de83770298a36b4ed17d227767285d75e9",
}


def scientific_manifest(encoded):
    item = json.loads(encoded)
    item.pop("created_at", None)
    item.pop("artifact_id")
    item["identity"].pop("code_sha256", None)
    return item


def verify_equivalent(original, retained):
    if scientific_manifest(original[1]) != scientific_manifest(retained[1]):
        raise ValueError("The published feature definitions differ")
    for row in (original, retained):
        manifest = json.loads(row[1])
        if manifest["artifact_id"] != row[0]:
            raise ValueError("Embedded and indexed artifact IDs differ")
        for name, blob in zip(("features.parquet", "requests.parquet"), row[2:], strict=True):
            if hashlib.sha256(blob).hexdigest() != manifest["sha256"][name]:
                raise ValueError("Stored Parquet checksum differs from the manifest")
    if original[2:] != retained[2:]:
        raise ValueError("Feature or request-link bytes differ")


def canonical_equivalent(db, row):
    """Choose a stored replacement only after checking the actual incoming payload."""
    target = REPLACEMENTS.get(row[0])
    if target:
        saved = db.execute("SELECT * FROM feature_artifacts WHERE artifact_id=?", (target,)).fetchone()
        if saved:
            verify_equivalent(row, saved)
            return target
    # When replaying an old snapshot into an empty database, the earlier revision
    # may arrive first. Replace it transactionally when the reviewed revision arrives.
    superseded = [a for a, b in REPLACEMENTS.items() if b == row[0]]
    for old in superseded:
        previous = db.execute("SELECT * FROM feature_artifacts WHERE artifact_id=?", (old,)).fetchone()
        if previous:
            verify_equivalent(previous, row)
            db.execute("DELETE FROM feature_artifacts WHERE artifact_id=?", (old,))
    if target or superseded:
        return None
    # A new producer fingerprint needs an explicit canonical choice. Do not publish
    # a second identical result or discard provenance by guessing which ID to keep.
    manifest = json.loads(row[1])
    if "code_sha256" not in manifest.get("identity", {}):
        return None
    for artifact_id, encoded in db.execute(
        "SELECT artifact_id,manifest_json FROM feature_artifacts WHERE artifact_id<>?", (row[0],)
    ):
        if scientific_manifest(encoded) == scientific_manifest(row[1]):
            candidate = db.execute("SELECT * FROM feature_artifacts WHERE artifact_id=?", (artifact_id,)).fetchone()
            verify_equivalent(candidate, row)
            raise ValueError(f"Equivalent feature publication already exists as {artifact_id}; review both producer revisions")
    return None


def retire_pairs(path):
    from src.utils.locking import file_lock

    if path.name != "MUL_TEXT_US_ACCIDENTS.sqlite3" or not path.is_file() or path.is_symlink():
        raise ValueError("Expected the US Accidents database")
    with file_lock(path.with_suffix(".extraction.lock"), timeout=5), closing(
        sqlite3.connect(path.resolve().as_uri() + "?mode=rw", uri=True, timeout=5)
    ) as db, db:
        db.execute("BEGIN IMMEDIATE")
        if db.execute("SELECT dataset_id FROM dataset_identity").fetchall() != [("MUL_TEXT_US_ACCIDENTS",)]:
            raise ValueError("Unexpected dataset")
        released = 0
        for old, new in REPLACEMENTS.items():
            keep = db.execute("SELECT manifest_json FROM feature_artifacts WHERE artifact_id=?", (new,)).fetchone()
            if not keep or json.loads(keep[0])["identity"]["provider"] != "nimble":
                raise ValueError("Missing retained Nimble revision")
            row = db.execute("SELECT * FROM feature_artifacts WHERE artifact_id=?", (old,)).fetchone()
            if row:
                if canonical_equivalent(db, row) != new:
                    raise ValueError("No verified replacement")
                db.execute("DELETE FROM feature_artifacts WHERE artifact_id=?", (old,))
                released += sum(map(len, row[2:]))
        inventory = db.execute("SELECT artifact_id FROM feature_artifacts WHERE json_extract(manifest_json,'$.identity.provider')='nimble'").fetchall()
        if {r[0] for r in inventory} != set(REPLACEMENTS.values()):
            raise ValueError("Unexpected Nimble inventory; rolling back")
        return {"retained": sorted(REPLACEMENTS.values()), "retired": sorted(REPLACEMENTS),
                "retired_blob_bytes_in_this_run": released, "database_bytes": path.stat().st_size}
