"""Local regression checks for the probe; only its VSC run validates the actual mount."""

import json
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest

from src.cluster import storage_contracts as probe
from src.utils.locking import file_lock
from src.utils.serialization import write_json


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


@pytest.fixture
def storage_run(isolated_output, monkeypatch):
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

    simulation = {"stdout": "", "failure": None, "damage": None, "calls": []}

    def invoke(command, **kwargs):
        simulation["calls"].append(command)
        assert "--nodelist=node2" in command
        assert kwargs["check"] is True and kwargs["timeout"] == 60
        kwargs["stdout"].write(simulation["stdout"])
        kwargs["stderr"].write("Synthetic launcher diagnostic\n")
        if simulation["failure"] == "exit":
            raise subprocess.CalledProcessError(1, command)
        if simulation["failure"] == "timeout":
            raise subprocess.TimeoutExpired(command, 60)
        if simulation["failure"] == "missing_receipt":
            return subprocess.CompletedProcess(command, 0)
        with monkeypatch.context() as context:
            context.setenv("SLURMD_NODENAME", "node2")
            context.setattr(
                sys, "argv", ["storage_contracts", *command[command.index("--worker") :]]
            )
            probe.main()
        if simulation["damage"]:
            folder = Path(command[command.index("--worker") + 1])
            receipt = probe.worker_receipt(folder.name, expect_busy="--expect-busy" in command)
            if simulation["damage"] == "json":
                receipt.write_text("invalid JSON")
            else:
                value = json.loads(receipt.read_text())
                value[simulation["damage"]] = "wrong"
                write_json(receipt, value)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(probe.subprocess, "check_output", lambda *a, **kw: "node1\nnode2\n")
    monkeypatch.setattr(probe.subprocess, "run", invoke)
    return simulation


@pytest.mark.parametrize("launcher_output", ["", "Cluster launch notice\n{unrelated message}\n"])
def test_storage_probe_saves_evidence_and_restores_snapshot(
    isolated_output, storage_run, launcher_output
):
    storage_run["stdout"] = launcher_output
    report = probe.run(execute=True)
    assert report["snapshot_restore"]
    assert report["cross_node_exclusion"]["node"] == "node2"
    assert report["quota_verified"] is report["independent_backup_verified"] is False
    manifests = list(
        (isolated_output / "manifests" / "before_extraction" / "storage_contracts").glob("*.json")
    )
    assert len(manifests) == 1
    assert json.loads(manifests[0].read_text())["status"].startswith("storage_contracts_passed")
    assert len(storage_run["calls"]) == 2
    for phase in ("blocked", "released"):
        receipt = json.loads(Path(report["worker_receipts"][phase]).read_text())
        assert receipt["job_id"] == "synthetic-test"
        assert receipt["expect_busy"] is (phase == "blocked")
        assert (Path(report["worker_logs"]) / f"{phase}.stdout.log").read_text() == launcher_output
        assert (
            Path(report["worker_logs"]) / f"{phase}.stderr.log"
        ).read_text() == "Synthetic launcher diagnostic\n"
    assert report["cross_node_exclusion"]["probe_id"] != report["release_and_reacquire"]["probe_id"]


@pytest.mark.parametrize("failure", ["exit", "timeout", "missing_receipt"])
def test_storage_probe_keeps_failed_launch_evidence(isolated_output, storage_run, failure):
    storage_run.update(failure=failure, stdout="diagnostic, not a JSON result\n")
    expected = {
        "exit": subprocess.CalledProcessError,
        "timeout": subprocess.TimeoutExpired,
        "missing_receipt": RuntimeError,
    }[failure]
    with pytest.raises(expected):
        probe.run(execute=True)
    report_path = next(
        (isolated_output / "manifests" / "before_extraction" / "storage_contracts").glob("*.json")
    )
    report = json.loads(report_path.read_text())
    assert report["status"] == "failed" and report["error_type"] == expected.__name__
    assert report["error_message"]
    assert (Path(report["worker_logs"]) / "blocked.stdout.log").read_text() == storage_run["stdout"]
    assert (Path(report["worker_logs"]) / "blocked.stderr.log").is_file()
    assert len(storage_run["calls"]) == 1  # No release phase after a failed exclusion check.


@pytest.mark.parametrize("damage", ["json", "probe_id", "job_id", "expect_busy", "node", "status"])
def test_storage_probe_rejects_bad_or_foreign_receipts(isolated_output, storage_run, damage):
    storage_run["damage"] = damage
    with pytest.raises((ValueError, AssertionError)):
        probe.run(execute=True)
    report_path = next(
        (isolated_output / "manifests" / "before_extraction" / "storage_contracts").glob("*.json")
    )
    assert json.loads(report_path.read_text())["status"] == "failed"


def test_storage_probe_refuses_preexisting_receipt(storage_run):
    path = probe.worker_receipt("stale-run", expect_busy=True)
    write_json(path, {"status": "old evidence"})
    with pytest.raises(FileExistsError, match="existing worker receipt"):
        probe.remote_worker([], run_id="stale-run", expect_busy=True)
    assert not storage_run["calls"]
    assert json.loads(path.read_text()) == {"status": "old evidence"}


def test_storage_probe_refuses_login_node(monkeypatch):
    monkeypatch.setattr(probe.paths, "on_vsc", lambda: True)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="allocation"):
        probe.run(execute=True)
