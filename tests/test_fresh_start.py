"""A fresh output tree must never mean lost research data or active-job output."""

import json
from pathlib import Path

import pytest

from src.cluster import fresh_start, maintenance
from src.utils import paths


def test_archive_both_tiers_preserves_every_file_and_protected_data(isolated_output, monkeypatch):
    monkeypatch.setattr(fresh_start, "require_idle_vsc", lambda: None)
    roots = [paths.outputs_dir(), paths.large_outputs_dir()]
    protected = []
    for root in roots:
        root.mkdir(parents=True)
        for name in (
            "3. experiment_0/logs/run.log",
            "3. experiment_0/results/counts.sqlite3",
            "transfers/cache_backups/paid.sqlite3",
        ):
            file = root / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(b"historical evidence")
        for name in (
            "data/raw/source.csv",
            "data/decision_inputs/questions.sqlite3",
            "data/jev_cache/features.sqlite3",
            "checkpoints/weights.bin",
            "environments/python",
        ):
            file = root.parent / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(b"protected")
            protected.append(file)
    preview = fresh_start.run()
    assert preview["status"] == "preview"
    assert all((r / "3. experiment_0/logs/run.log").exists() for r in roots)
    result = fresh_start.run(apply=True, writers_stopped=True)
    assert result["status"] == "fresh_outputs_ready"
    for move in result["moves"]:
        archive = Path(move["archive"])
        assert (
            archive / "3. experiment_0/results/counts.sqlite3"
        ).read_bytes() == b"historical evidence"
        assert (archive / "transfers/cache_backups/paid.sqlite3").is_file()
        assert not list(Path(move["source"]).rglob("*.sqlite3"))
    assert all(p.read_bytes() == b"protected" for p in protected)
    assert paths.logs_dir("experiment_0").is_dir()


def test_archive_requires_stopped_writers_and_refuses_busy_scheduler(isolated_output, monkeypatch):
    for root in (paths.outputs_dir(), paths.large_outputs_dir()):
        root.mkdir(parents=True)
    with pytest.raises(ValueError, match="writers"):
        fresh_start.run(apply=True)
    monkeypatch.setattr(maintenance, "storage_policy_failures", lambda _: [])
    monkeypatch.setattr(
        maintenance.subprocess,
        "check_output",
        lambda *a, **kw: "CLUSTER: wice\n123|jevpfn-exp0-tokens|/elsewhere\n",
    )
    with pytest.raises(RuntimeError, match="123"):
        fresh_start.run(apply=True, writers_stopped=True)
    assert not list(isolated_output.rglob("archived_output_JEVPFN"))


def test_scheduler_check_allows_other_projects_but_fails_on_unreadable_listing(
    isolated_output, monkeypatch
):
    monkeypatch.setattr(maintenance, "storage_policy_failures", lambda _: [])
    monkeypatch.setattr(
        maintenance.subprocess,
        "check_output",
        lambda *a, **kw: "CLUSTER: wice\n123|creditpfn|/some/other/project\n",
    )
    maintenance.require_idle_vsc()
    monkeypatch.setattr(
        maintenance.subprocess, "check_output", lambda *a, **kw: "unexpected format"
    )
    with pytest.raises(ValueError, match="interpret"):
        maintenance.require_idle_vsc()


def test_partial_archive_failure_keeps_recoverable_journal(isolated_output, monkeypatch):
    monkeypatch.setattr(fresh_start, "require_idle_vsc", lambda: None)
    for root in (paths.outputs_dir(), paths.large_outputs_dir()):
        root.mkdir(parents=True)
        (root / "evidence").write_text("keep")
    original = Path.rename

    def interrupt(path, destination):
        if path == paths.large_outputs_dir():
            raise OSError("simulated interruption")
        return original(path, destination)

    monkeypatch.setattr(Path, "rename", interrupt)
    with pytest.raises(OSError):
        fresh_start.run(apply=True, writers_stopped=True)
    receipt = next((paths.outputs_dir().parent / "archived_output_JEVPFN").glob("*.json"))
    body = json.loads(receipt.read_text())
    assert body["status"] == "interrupted_archives_preserved"
    assert (Path(body["moves"][0]["archive"]) / "evidence").read_text() == "keep"
    assert (paths.large_outputs_dir() / "evidence").read_text() == "keep"


def test_archive_retries_brief_sharing_lock_without_overwriting(tmp_path, monkeypatch):
    source, destination = tmp_path / "source", tmp_path / "archive"
    source.mkdir()
    (source / "evidence").write_text("keep")
    original = Path.rename
    attempts = []

    def transient_lock(path, target):
        attempts.append(path)
        if len(attempts) == 1:
            raise PermissionError("brief sharing lock")
        return original(path, target)

    monkeypatch.setattr(Path, "rename", transient_lock)
    monkeypatch.setattr(fresh_start.time, "sleep", lambda _: None)
    fresh_start.archive_directory(source, destination)
    assert len(attempts) == 2
    assert (destination / "evidence").read_text() == "keep"
    source.mkdir()
    with pytest.raises(FileExistsError):
        fresh_start.archive_directory(source, destination)
    assert source.is_dir()
    assert (destination / "evidence").read_text() == "keep"


def test_permanent_permission_failure_preserves_source(tmp_path, monkeypatch):
    source, destination = tmp_path / "source", tmp_path / "archive"
    source.mkdir()
    (source / "evidence").write_text("keep")
    attempts = []

    def denied(*args):
        attempts.append(1)
        raise PermissionError("access denied")

    monkeypatch.setattr(Path, "rename", denied)
    monkeypatch.setattr(fresh_start.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError):
        fresh_start.archive_directory(source, destination)
    assert len(attempts) == 7
    assert (source / "evidence").read_text() == "keep"
    assert not destination.exists()
