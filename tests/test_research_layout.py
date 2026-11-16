import sqlite3

import pytest

from src.cluster.research_layout import migrate
from src.utils import paths
from src.utils.serialization import file_sha256


def test_idle_result_move_preserves_database_and_relative_payloads(tmp_path, monkeypatch):
    monkeypatch.delenv("VSC_DATA", raising=False)
    old = tmp_path / "5. experiment_2/results/pca"
    old.mkdir(parents=True)
    db = old / "a.sqlite3"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE features (x INTEGER)")
        con.execute("INSERT INTO features VALUES (3)")
    con.close()
    expected = file_sha256(db)
    with pytest.raises(ValueError):
        migrate(tmp_path)
    report = migrate(tmp_path, writers_stopped=True)
    new = tmp_path / paths.result_relative("experiment_2") / "pca/a.sqlite3"
    assert new.exists() and not db.exists() and file_sha256(new) == expected
    assert report["deleted_files"] == 0


def test_migration_refuses_two_populated_layouts_or_database_sidecars(tmp_path, monkeypatch):
    monkeypatch.delenv("VSC_DATA", raising=False)
    old = tmp_path / "4. experiment_1/results"
    old.mkdir(parents=True)
    (old / "store.sqlite3-wal").write_bytes(b"live")
    with pytest.raises(ValueError, match="sidecar"):
        migrate(tmp_path, writers_stopped=True)
    new = tmp_path / paths.result_relative("experiment_1")
    new.mkdir(parents=True)
    with pytest.raises(ValueError, match="Both layouts"):
        migrate(tmp_path, writers_stopped=True)
