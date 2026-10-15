"""Local regression checks for the probe; only its VSC run validates the actual mount."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from src.cluster import storage_contracts as probe
from src.utils.locking import file_lock


def test_worker_requires_visible_database_and_exclusive_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(probe.paths, "require_project_output", Path)
    with pytest.raises(FileNotFoundError):
        probe.worker(tmp_path, expect_busy=True)
    assert not (tmp_path / "probe.sqlite3").exists()
    with closing(sqlite3.connect(tmp_path / "probe.sqlite3")) as db, db:
        db.execute("CREATE TABLE probe (id INTEGER PRIMARY KEY,value TEXT)")
        db.execute("INSERT INTO probe VALUES (1,'original')")
    with file_lock(tmp_path / "writer.lock"):
        assert probe.worker(tmp_path, expect_busy=True)["status"] == "blocked_as_expected"
    with pytest.raises(AssertionError, match="Both nodes acquired"):
        probe.worker(tmp_path, expect_busy=True)
    assert probe.worker(tmp_path, expect_busy=False)["status"] == "acquired_after_release"


def test_storage_probe_saves_evidence_and_restores_snapshot(isolated_output, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-test")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "node[1-2]")
    monkeypatch.setenv("SLURMD_NODENAME", "node1")
    monkeypatch.setattr(probe.paths, "on_vsc", lambda: True)
    monkeypatch.setattr(probe.paths, "require_project_output", Path)
    monkeypatch.setattr(
        probe.paths, "results_dir", lambda *args, **kw: isolated_output.joinpath(*args)
    )
    monkeypatch.setattr(probe.paths, "manifests_dir", lambda *args: isolated_output / "manifests")
    monkeypatch.setattr(probe, "storage_policy_failures", lambda _: [])

    def invoke(command, **kwargs):
        if command[0] == "scontrol":
            return "node1\nnode2\n"
        assert "--nodelist=node2" in command
        with monkeypatch.context() as context:
            context.setenv("SLURMD_NODENAME", "node2")
            result = probe.worker(
                command[command.index("--worker") + 1], expect_busy="--expect-busy" in command
            )
        return json.dumps(result)

    monkeypatch.setattr(probe.subprocess, "check_output", invoke)
    report = probe.run(execute=True)
    assert report["snapshot_restore"]
    assert report["cross_node_exclusion"]["node"] == "node2"
    assert report["quota_verified"] is report["independent_backup_verified"] is False
    manifests = list((isolated_output / "manifests").rglob("*.json"))
    assert len(manifests) == 1
    assert json.loads(manifests[0].read_text())["status"].startswith("storage_contracts_passed")


def test_storage_probe_refuses_login_node(monkeypatch):
    monkeypatch.setattr(probe.paths, "on_vsc", lambda: True)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="allocation"):
        probe.run(execute=True)
