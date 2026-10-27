"""Local regression checks for the probe; only its VSC run validates the actual mount."""

import json
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

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
def receipt_clock(monkeypatch):
    clock = {"seconds": 0, "sleeps": []}

    def advance(seconds):
        clock["sleeps"].append(seconds)
        clock["seconds"] += seconds

    # Only the receipt reader uses virtual time; real lock behavior stays intact.
    monkeypatch.setattr(
        probe, "time", SimpleNamespace(monotonic=lambda: clock["seconds"], sleep=advance)
    )
    return clock


@pytest.fixture
def storage_run(isolated_output, monkeypatch, receipt_clock):
    monkeypatch.setattr(
        "src.cluster.evidence.current_identity",
        lambda: {"source_sha256": "fixture", "runtime_pins_sha256": "fixture"},
    )
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-test")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "node[1-2]")
    monkeypatch.setenv("SLURMD_NODENAME", "node1")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "wice")
    monkeypatch.setattr(probe, "storage_backend", lambda _: "lustre")
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
    assert report["cluster"] == "wice"
    assert report["storage_backend"] == "lustre"
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
def test_storage_probe_keeps_failed_launch_evidence(isolated_output, storage_run, failure, capsys):
    storage_run.update(failure=failure, stdout="diagnostic, not a JSON result\n")
    expected = {
        "exit": subprocess.CalledProcessError,
        "timeout": subprocess.TimeoutExpired,
        "missing_receipt": TimeoutError,
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
    if failure != "missing_receipt":
        main_log = capsys.readouterr().err
        assert "Synthetic launcher diagnostic" in main_log
        assert report["worker_logs"] in main_log


def test_storage_probe_waits_for_released_receipt_without_relaunching_worker(
    isolated_output, storage_run, monkeypatch, receipt_clock
):
    original_read = Path.read_text
    hidden_reads = []

    def delayed_read(path, *args, **kwargs):
        if path.name == "released.json" and len(hidden_reads) < 3:
            assert path.is_file()  # Published on the worker, not yet visible to the parent.
            hidden_reads.append(path)
            raise FileNotFoundError(path)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", delayed_read)
    report = probe.run(execute=True)
    assert report["status"].startswith("storage_contracts_passed")
    assert report["snapshot_restore"]
    assert len(storage_run["calls"]) == 2  # Each database operation executes once.
    assert report["cross_node_exclusion"]["receipt_visibility_wait_seconds"] == 0
    assert report["release_and_reacquire"]["receipt_visibility_wait_seconds"] == 1.5
    assert receipt_clock["sleeps"] == [0.5, 0.5, 0.5]


def test_receipt_reader_stops_at_deadline(tmp_path, receipt_clock):
    with pytest.raises(TimeoutError, match="not visible within 1.25s"):
        probe.read_worker_receipt(tmp_path / "missing.json", timeout=1.25)
    assert receipt_clock["seconds"] == 1.25
    assert receipt_clock["sleeps"] == [0.5, 0.5, 0.25]


@pytest.mark.parametrize("failure", ["invalid_json", "permission", "io"])
def test_receipt_reader_does_not_retry_corruption_or_access_errors(
    tmp_path, monkeypatch, receipt_clock, failure
):
    receipt = tmp_path / "receipt.json"
    receipt.write_text("invalid JSON", encoding="utf-8")
    expected = json.JSONDecodeError
    if failure != "invalid_json":
        expected = PermissionError if failure == "permission" else OSError

        def fail(*args, **kwargs):
            raise expected("Synthetic read failure")

        monkeypatch.setattr(Path, "read_text", fail)
    with pytest.raises(expected):
        probe.read_worker_receipt(receipt)
    assert not receipt_clock["sleeps"]


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


@pytest.mark.parametrize(
    "folder,expected",
    [
        ("/gpfs1/scratch/user/JEVPFN/results", "gpfs"),
        ("/lustre1/project/stg_00211/JEVPFN/results", "lustre"),
    ],
)
def test_storage_backend_uses_actual_native_results_root(folder, expected):
    assert probe.storage_backend(folder) == expected


@pytest.mark.parametrize("folder", ["/data/user/gpfs1/results", "/tmp/lustre1", "/gpfsfake/data"])
def test_storage_backend_rejects_unknown_or_nested_root(folder):
    with pytest.raises(ValueError, match="recognized native"):
        probe.storage_backend(folder)


def test_storage_probe_reports_native_gpfs(storage_run, monkeypatch):
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "mindwell")
    monkeypatch.setattr(probe, "storage_backend", lambda _: "gpfs")
    report = probe.run(execute=True)
    assert report["cluster"] == "mindwell"
    assert report["storage_backend"] == "gpfs"
