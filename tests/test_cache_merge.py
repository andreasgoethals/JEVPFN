"""Synthetic paid-response identities; no live calls, weights or real cache mutations."""

import json
import sqlite3

import pandas as pd
import pytest

from src.jev.cache import ResponseCache
from src.jev.features import FeatureBlock, load_block, save_block
from src.jev.merge import merge_cache, snapshot
from src.jev.requests import IntendedRequest
from src.utils.serialization import canonical_json, digest, file_sha256


def request(provider="jev", text="same", dataset="fixture", row="csv:0"):
    settings = {
        "provider": provider,
        "model": "jev-1.14.0",
        "version": "1.14.0",
        "api_mapping_version": "typesafe-systemone-v1",
        "request_configuration": {},
    }
    return IntendedRequest(
        canonical_json(
            {
                "state": {"text": {"t": text}},
                "question": {"kind": "binary_probability", "text": "Is the text positive?"},
            }
        ),
        canonical_json(settings),
        canonical_json({"dataset_id": dataset, "row_id": row}),
    )


def response(value=0.7):
    return {"model": "jev-1.14.0", "answers": {"semantic": {"type": "noul", "noul": value}}}


def test_failed_attempts_merge_once_without_creating_responses(tmp_path):
    remote = ResponseCache(tmp_path / "remote.sqlite3", "fixture")
    req = request()
    remote.record_failure(req, ValueError("not retained"))
    target = tmp_path / "local.sqlite3"
    assert merge_cache(remote.path, target)["failures_added"] == 1
    assert merge_cache(remote.path, target)["failures_added"] == 0
    local = ResponseCache(target, "fixture")
    assert local.failure_count(req) == 1
    assert local.get(req, kind="verified_response") is None


def cache(path, req=None, value=0.7):
    store = ResponseCache(path, "fixture")
    store.put(req or request(), response(value), kind="verified_response")
    return store


def feature(store, provider="jev"):
    req = request(provider)
    store.put(req, response(), kind="verified_response")
    identity = {
        "dataset_id": "fixture",
        "provider": provider,
        "representation": "joint",
        "data_identity": "data",
        "model": "pinned",
        "version": "one",
    }
    values = pd.DataFrame(
        {"row_id": ["csv:0", "csv:1"], f"{provider}__all_text__p_positive": [0.7, float("nan")]}
    )
    refs = pd.DataFrame(
        {
            "row_id": ["csv:0", "csv:1"],
            "group": ["all_text"] * 2,
            "request_key": [req.cache_key, None],
            "status": ["verified_response", "missing_text"],
        }
    )
    manifest = {
        "identity": identity,
        "artifact_id": digest(identity),
        "status": "complete",
        "namespace": "verified_response",
        "row_count": 2,
        "row_ids_sha256": digest(values.row_id.tolist()),
        "feature_columns": list(values.columns[1:]),
        "text_columns": ["t"],
        "response_store": str(store.path),
    }
    save_block(FeatureBlock(values, refs, manifest), store.path)
    return manifest["artifact_id"]


def test_union_providers_origins_artifacts_and_repeated_import(tmp_path, monkeypatch):
    from src.utils import paths

    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    local = cache(tmp_path / "local.sqlite3")
    local_id = feature(local)
    remote = cache(tmp_path / "remote.sqlite3", request("liquid"))
    remote_id = feature(remote, "liquid")
    remote.put(request(row="csv:99"), response(), kind="verified_response")
    before = file_sha256(remote.path)
    result = merge_cache(remote.path, local.path)
    assert result["responses_added"] == 1 and result["responses_reused"] == 1
    assert result["artifacts_added"] == 1 and result["origins_added"] == 2
    assert load_block(local.path, local_id).values.shape == (2, 2)
    assert load_block(local.path, remote_id).requests.iloc[1].status == "missing_text"
    assert file_sha256(remote.path) == before
    again = merge_cache(remote.path, local.path)
    assert again["responses_added"] == again["artifacts_added"] == again["origins_added"] == 0
    assert again["responses_reused"] == 2 and again["artifacts_reused"] == 1
    backup = load_block(__import__("pathlib").Path(result["backup"]), local_id)
    assert backup.values.equals(load_block(local.path, local_id).values)


@pytest.mark.parametrize("backup_destination", [True, False])
def test_conflict_rolls_back_new_responses_too(tmp_path, backup_destination):
    local = cache(tmp_path / "local.sqlite3")
    remote = cache(tmp_path / "remote.sqlite3", request(text="new"))
    remote.put(request(), response(0.2), kind="verified_response")
    with pytest.raises(ValueError, match="Conflicting"):
        merge_cache(remote.path, local.path, backup_destination=backup_destination)
    assert local.get(request(text="new"), kind="verified_response") is None
    assert local.get(request(), kind="verified_response") == response()
    assert (tmp_path / "merge_backups").exists() == backup_destination


def test_mirror_union_can_preserve_both_stores_without_extra_backups(tmp_path):
    local = cache(tmp_path / "local.sqlite3")
    remote = cache(tmp_path / "remote.sqlite3", request("intern"))
    before = file_sha256(remote.path)
    result = merge_cache(remote.path, local.path, backup_destination=False)
    assert result["responses_added"] == 1
    assert result["backup"] is None
    assert local.get(request(), kind="verified_response") == response()
    assert local.get(request("intern"), kind="verified_response") == response()
    assert file_sha256(remote.path) == before
    assert not (tmp_path / "merge_backups").exists()


def test_mirror_ledger_reuses_unchanged_versions_across_snapshots_and_feedback(tmp_path):
    local = cache(tmp_path / "local.sqlite3")
    remote = cache(tmp_path / "remote.sqlite3", request("intern"))
    with remote._connect() as db:
        db.execute(
            "CREATE TABLE extraction_jobs (request_key TEXT PRIMARY KEY,status TEXT,cost REAL)"
        )
        db.execute("INSERT INTO extraction_jobs VALUES ('job','attempting',0.1)")
    first_hash = file_sha256(remote.path)
    options = {"backup_destination": False, "deduplicate_ledger": True}
    assert merge_cache(remote.path, local.path, **options)["ledger_records_added"] == 1
    remote.put(request("intern", text="new result"), response(), kind="verified_response")
    assert file_sha256(remote.path) != first_hash
    assert merge_cache(remote.path, local.path, **options)["ledger_records_added"] == 0
    with remote._connect() as db:
        db.execute("UPDATE extraction_jobs SET status='completed',cost=0.2")
    assert merge_cache(remote.path, local.path, **options)["ledger_records_added"] == 1
    with local._connect() as db:
        versions = db.execute(
            "SELECT snapshot_sha256,record_json FROM imported_extraction_records"
        ).fetchall()
        assert len(versions) == 2
        assert (
            first_hash,
            canonical_json({"request_key": "job", "status": "attempting", "cost": 0.1}),
        ) in versions
        assert {json.loads(v[1])["cost"] for v in versions} == {0.1, 0.2}
        assert (
            db.execute("SELECT name FROM sqlite_master WHERE name='mirror_ledger_seen'").fetchone()
            is None
        )
    # Returning the durable union to a working set must not multiply its history.
    merge_cache(local.path, remote.path, **options)
    assert merge_cache(remote.path, local.path, **options)["ledger_records_added"] == 0
    with local._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM imported_extraction_records").fetchone() == (2,)


@pytest.mark.parametrize("deduplicate_ledger", [False, True])
def test_conflicting_ledger_identity_rolls_back_other_imported_changes(
    tmp_path, deduplicate_ledger
):
    from src.jev.merge import LEDGER_SQL

    local = cache(tmp_path / "local.sqlite3")
    remote = cache(tmp_path / "remote.sqlite3", request(text="must roll back"))
    for store, status in ((local, "completed"), (remote, "failed")):
        with store._connect() as db:
            db.execute(LEDGER_SQL)
            db.execute(
                "INSERT INTO imported_extraction_records VALUES ('snapshot','extraction_jobs','job',?)",
                (canonical_json({"status": status}),),
            )
    with pytest.raises(ValueError, match="Conflicting imported extraction record"):
        merge_cache(
            remote.path, local.path, backup_destination=False, deduplicate_ledger=deduplicate_ledger
        )
    assert local.get(request(text="must roll back"), kind="verified_response") is None
    with local._connect() as db:
        assert json.loads(
            db.execute("SELECT record_json FROM imported_extraction_records").fetchone()[0]
        ) == {"status": "completed"}


@pytest.mark.parametrize("damage", ["identity", "request", "origin", "trigger"])
def test_corruption_refused_before_local_write(tmp_path, damage):
    local, remote = cache(tmp_path / "local.sqlite3"), cache(tmp_path / "remote.sqlite3")
    with remote._connect() as db:
        if damage == "identity":
            db.execute("UPDATE dataset_identity SET dataset_id='other'")
        elif damage == "request":
            db.execute("UPDATE responses SET request_key='bad'")
        elif damage == "origin":
            db.execute("UPDATE origins SET provenance_hash='bad'")
        else:
            db.execute(
                "CREATE TRIGGER bad AFTER INSERT ON responses BEGIN DELETE FROM origins; END"
            )
    before = file_sha256(local.path)
    with pytest.raises(ValueError):
        merge_cache(remote.path, local.path)
    assert file_sha256(local.path) == before


def test_snapshot_includes_committed_wal(tmp_path):
    source = tmp_path / "wal.sqlite3"
    with sqlite3.connect(source) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE example(value)")
        db.execute("INSERT INTO example VALUES (42)")
        db.commit()
        copied = snapshot(source, tmp_path / "snapshot.sqlite3")
        with sqlite3.connect(copied) as other:
            assert other.execute("SELECT value FROM example").fetchone() == (42,)


def test_reservations_archived_without_resetting_local_budget(tmp_path):
    local, remote = cache(tmp_path / "local.sqlite3"), cache(tmp_path / "remote.sqlite3")
    for store, amount in [(local, 10), (remote, 20)]:
        with store._connect() as db:
            db.execute(
                "CREATE TABLE extraction_policy (id INTEGER PRIMARY KEY, settings_json TEXT)"
            )
            db.execute(
                "INSERT INTO extraction_policy VALUES (1,?)", (json.dumps({"budget": amount}),)
            )
    result = merge_cache(remote.path, local.path)
    assert result["ledger_records_added"] == 1
    with local._connect() as db:
        assert (
            json.loads(db.execute("SELECT settings_json FROM extraction_policy").fetchone()[0])[
                "budget"
            ]
            == 10
        )
        archived = db.execute("SELECT record_json FROM imported_extraction_records").fetchone()[0]
        assert json.loads(json.loads(archived)["settings_json"])["budget"] == 20


def test_corrupt_feature_blob_and_valid_but_conflicting_features(tmp_path, monkeypatch):
    import hashlib
    import io

    from src.utils import paths

    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    local, remote = cache(tmp_path / "local.sqlite3"), cache(tmp_path / "remote.sqlite3")
    artifact = feature(local)
    feature(remote)
    original = load_block(remote.path, artifact)
    altered = original.values.copy()
    altered.iloc[0, 1] = 0.2
    buffer = io.BytesIO()
    altered.to_parquet(buffer, index=False)
    blob = buffer.getvalue()
    with remote._connect() as db:
        db.execute("UPDATE feature_artifacts SET features_parquet=?", (blob,))
    with pytest.raises(ValueError, match="checksum"):
        merge_cache(remote.path, local.path)
    manifest = original.manifest
    manifest["sha256"]["features.parquet"] = hashlib.sha256(blob).hexdigest()
    with remote._connect() as db:
        db.execute("UPDATE feature_artifacts SET manifest_json=?", (json.dumps(manifest),))
    remote.put(request(text="must roll back"), response(), kind="verified_response")
    with pytest.raises(ValueError, match="Conflicting feature"):
        merge_cache(remote.path, local.path)
    assert local.get(request(text="must roll back"), kind="verified_response") is None
    assert load_block(local.path, artifact).values.iloc[0, 1] == 0.7


def test_merge_on_vsc_requires_project_storage(isolated_output):
    from src.utils import paths

    remote = cache(paths.jev_cache_path() / "incoming.sqlite3")
    with pytest.raises(ValueError, match="PROJECT|project|DATA"):
        merge_cache(remote.path, paths.outputs_dir() / "wrong.sqlite3")
    result = merge_cache(remote.path, paths.dataset_cache_path("fixture"))
    assert result["responses_added"] == 1
