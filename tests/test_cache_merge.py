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
        "model": "jev-1.13.0",
        "version": "1.13.0",
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
    return {"model": "jev-1.13.0", "answers": {"semantic": {"type": "noul", "noul": value}}}


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


def test_conflict_rolls_back_new_responses_too(tmp_path):
    local = cache(tmp_path / "local.sqlite3")
    remote = cache(tmp_path / "remote.sqlite3", request(text="new"))
    remote.put(request(), response(0.2), kind="verified_response")
    with pytest.raises(ValueError, match="Conflicting"):
        merge_cache(remote.path, local.path)
    assert local.get(request(text="new"), kind="verified_response") is None
    assert local.get(request(), kind="verified_response") == response()


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
