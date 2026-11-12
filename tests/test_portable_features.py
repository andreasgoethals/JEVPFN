import json
import sqlite3
import zipfile
from contextlib import closing

import pytest

from src.evaluation import portable_features
from src.jev.features import feature_inventory
from src.jev.merge import ARTIFACT_SQL
from src.utils.serialization import digest, file_sha256


def bundle(tmp_path, provider, version="first"):
    db_path = tmp_path / f"{provider}-{version}.sqlite3"
    artifacts = []
    with closing(sqlite3.connect(db_path)) as db, db:
        db.execute(ARTIFACT_SQL)
        db.execute("CREATE TABLE payload_hashes (request_key TEXT PRIMARY KEY, payload_sha256 BLOB)")
        db.execute("INSERT INTO payload_hashes VALUES (?,?)", (provider, bytes(32)))
        for rep in ("joint", "per_column"):
            identity = {"dataset_id": "fixture", "provider": provider, "representation": rep, "version": version}
            aid = digest(identity)
            manifest = {"identity": identity, "artifact_id": aid, "row_count": 1, "feature_columns": []}
            db.execute("INSERT INTO feature_artifacts VALUES (?,?,?,?)", (aid, json.dumps(manifest), b"x", b"y"))
            artifacts.append(aid)
    path = db_path.with_suffix(".zip")
    with zipfile.ZipFile(path, "w") as archive:
        archive.write(db_path, "fixture.sqlite3")
        archive.writestr("manifest.json", json.dumps({
            "schema": "jevpfn.decision-features.v1", "providers": [provider], "jobs": {provider: ["12"]},
            "datasets": [{"dataset_id": "fixture", "path": "fixture.sqlite3", "bytes": db_path.stat().st_size,
                          "sha256": file_sha256(db_path), "artifacts": artifacts}],
        }))
    return path


def test_portable_import_can_add_a_verified_provider_without_dropping_or_mixing_versions(tmp_path, monkeypatch):
    root = tmp_path / "installed"
    monkeypatch.setattr(portable_features.paths, "on_vsc", lambda: False)
    monkeypatch.setattr(portable_features.paths, "jev_cache_path", lambda _: root)
    monkeypatch.setattr(portable_features, "load_catalog", lambda: None)
    monkeypatch.setattr(portable_features, "load_yaml", lambda _: None)
    monkeypatch.setattr(portable_features, "selected_entries", lambda *_: [{"id": "fixture"}])
    first = bundle(tmp_path, "liquid")
    second = bundle(tmp_path, "decision3")
    conflicting = bundle(tmp_path, "decision3", version="different")
    portable_features.import_bundle(first, file_sha256(first))
    with pytest.raises(ValueError, match="would drop"):
        portable_features.import_bundle(second, file_sha256(second))
    for _ in range(2):
        portable_features.import_bundle(second, file_sha256(second), add_providers=True)
    installed = root / "fixture.sqlite3"
    assert feature_inventory(installed).provider.value_counts().to_dict() == {"decision3": 2, "liquid": 2}
    assert portable_features.verified_payload_hashes(installed, ["decision3", "liquid"]) == {
        "decision3": "00" * 32, "liquid": "00" * 32,
    }
    previous = file_sha256(installed)
    with pytest.raises(ValueError, match="mix versions"):
        portable_features.import_bundle(conflicting, file_sha256(conflicting), add_providers=True)
    assert file_sha256(installed) == previous
