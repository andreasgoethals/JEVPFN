"""A fresh output tree must never mean lost research data or active-job output."""

import errno
import json
import os
import stat
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
            "data/text_embeddings/embeddings.sqlite3",
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


@pytest.mark.parametrize("delete", [False, True])
def test_reset_requires_stopped_writers_and_refuses_busy_scheduler(
    isolated_output, monkeypatch, delete
):
    for root in (paths.outputs_dir(), paths.large_outputs_dir()):
        root.mkdir(parents=True)
    with pytest.raises(ValueError, match="writers"):
        fresh_start.run(apply=True, delete=delete)
    monkeypatch.setattr(maintenance, "storage_policy_failures", lambda _: [])
    monkeypatch.setattr(
        maintenance.subprocess,
        "check_output",
        lambda *a, **kw: "CLUSTER: wice\n123|jevpfn-exp0-tokens|/elsewhere\n",
    )
    with pytest.raises(RuntimeError, match="123"):
        fresh_start.run(apply=True, writers_stopped=True, delete=delete)
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


def test_local_reset_archives_outputs_without_touching_data(isolated_output, monkeypatch):
    monkeypatch.delenv("VSC_DATA")
    root = paths.notebook_outputs_dir()
    root.mkdir(parents=True)
    (root / "All Results.md").write_text("previous report")
    data = root.parent / "data" / "raw" / "source.csv"
    data.parent.mkdir(parents=True)
    data.write_text("protected")
    monkeypatch.setattr(fresh_start, "require_idle_vsc", lambda: pytest.fail("No local scheduler"))
    assert fresh_start.run(local=True)["status"] == "preview"
    assert (root / "All Results.md").is_file()
    with pytest.raises(ValueError, match="writers"):
        fresh_start.run(local=True, apply=True)
    result = fresh_start.run(local=True, apply=True, writers_stopped=True)
    assert result["status"] == "fresh_outputs_ready"
    assert not list(root.iterdir())
    assert (Path(result["moves"][0]["archive"]) / "All Results.md").read_text() == "previous report"
    assert data.read_text() == "protected"


def test_local_reset_cannot_bypass_vsc_checks(isolated_output):
    with pytest.raises(RuntimeError, match="not a VSC shell"):
        fresh_start.run(local=True, apply=True, writers_stopped=True)


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("include_archives", [False, True])
def test_delete_cleans_only_output_and_explicit_archives(
    isolated_output, monkeypatch, local, include_archives
):
    if local:
        monkeypatch.delenv("VSC_DATA")
        roots = [paths.notebook_outputs_dir()]
        monkeypatch.setattr(fresh_start, "require_idle_vsc", lambda: pytest.fail("Local reset"))
    else:
        roots = [paths.outputs_dir(), paths.large_outputs_dir()]
        monkeypatch.setattr(fresh_start, "require_idle_vsc", lambda: None)
    protected = []
    archives = []
    for root in roots:
        old_output = root / "3. experiment_0" / "results" / "old_counts.sqlite3"
        old_output.parent.mkdir(parents=True)
        old_output.write_bytes(b"old audit")
        archive = root.parent / "archived_output_JEVPFN" / "old-run" / "All Results.md"
        archive.parent.mkdir(parents=True)
        archive.write_text("old results")
        archives.append(archive)
        for name in (
            "data/raw/source.csv",
            "data/decision_inputs/questions.sqlite3",
            "data/jev_cache/paid_features.sqlite3",
            "checkpoints/model.safetensors",
            "environments/python",
            "README.md",
        ):
            target = root.parent / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"must survive byte for byte")
            protected.append(target)
    options = dict(local=local, delete=True, include_archives=include_archives)
    preview = fresh_start.run(**options)
    assert preview["status"] == "preview"
    assert all(
        root.joinpath("3. experiment_0/results/old_counts.sqlite3").is_file() for root in roots
    )
    assert all(p.exists() for p in archives)
    with pytest.raises(ValueError, match="writers"):
        fresh_start.run(apply=True, **options)
    result = fresh_start.run(apply=True, writers_stopped=True, **options)
    assert result["action"] == "permanently_delete_outputs"
    assert result["status"] == "fresh_outputs_ready"
    assert "moves" not in result
    assert all(entry["deleted"] for entry in result["deletions"])
    assert all(root.is_dir() and not any(p.is_file() for p in root.rglob("*")) for root in roots)
    assert all(p.read_bytes() == b"must survive byte for byte" for p in protected)
    for archive in archives:
        assert archive.exists() is (not include_archives)
        if include_archives:
            assert not archive.parent.parent.exists()
    if not local:
        assert paths.logs_dir("experiment_0").is_dir()
    # Repeating a completed reset is safe and does not turn inputs into cleanup targets.
    fresh_start.run(apply=True, writers_stopped=True, **options)
    assert all(p.read_bytes() == b"must survive byte for byte" for p in protected)


def test_archive_deletion_requires_explicit_delete_mode():
    with pytest.raises(ValueError, match="requires --delete"):
        fresh_start.run(include_archives=True)


@pytest.mark.parametrize("entry_kind", ["symlink", "junction", "mount"])
def test_delete_validates_every_tier_before_removing_any_file(
    isolated_output, monkeypatch, entry_kind
):
    monkeypatch.setattr(fresh_start, "require_idle_vsc", lambda: None)
    roots = [paths.outputs_dir(), paths.large_outputs_dir()]
    for root in roots:
        root.mkdir(parents=True)
        (root / "evidence").write_bytes(b"untouched")
    linked = roots[1] / "linked-subtree"
    linked.mkdir()
    attribute = {"symlink": "is_symlink", "junction": "is_junction", "mount": "is_mount"}[
        entry_kind
    ]
    original = getattr(Path, attribute, lambda _: False)
    monkeypatch.setattr(Path, attribute, lambda p: p == linked or original(p), raising=False)
    with pytest.raises(ValueError, match="Linked|mounted"):
        fresh_start.run(delete=True, apply=True, writers_stopped=True)
    assert all((root / "evidence").read_bytes() == b"untouched" for root in roots)


def test_delete_refuses_linked_archive_before_clearing_active_output(isolated_output, monkeypatch):
    monkeypatch.delenv("VSC_DATA")
    root = paths.notebook_outputs_dir()
    root.mkdir(parents=True)
    (root / "evidence").write_text("untouched")
    archive = root.parent / "archived_output_JEVPFN"
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda p: p == archive or original(p))
    with pytest.raises(ValueError, match="Linked"):
        fresh_start.run(
            local=True, delete=True, include_archives=True, apply=True, writers_stopped=True
        )
    assert (root / "evidence").read_text() == "untouched"


def test_interrupted_deletion_can_be_repeated_without_touching_inputs(isolated_output, monkeypatch):
    monkeypatch.setattr(fresh_start, "require_idle_vsc", lambda: None)
    roots = [paths.outputs_dir(), paths.large_outputs_dir()]
    for root in roots:
        root.mkdir(parents=True)
        (root / "evidence").write_text("old")
    protected = roots[1].parent / "data" / "jev_cache" / "paid.sqlite3"
    protected.parent.mkdir(parents=True)
    protected.write_bytes(b"paid responses")
    original = fresh_start.shutil.rmtree

    def deny_project(path, *args, **kwargs):
        if path == roots[1]:
            raise PermissionError("stop writers")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(fresh_start.shutil, "rmtree", deny_project)
    with pytest.raises(PermissionError):
        fresh_start.run(delete=True, apply=True, writers_stopped=True)
    assert (roots[1] / "evidence").read_text() == "old"
    monkeypatch.setattr(fresh_start.shutil, "rmtree", original)
    fresh_start.run(delete=True, apply=True, writers_stopped=True)
    assert all(not (root / "evidence").exists() for root in roots)
    assert protected.read_bytes() == b"paid responses"


@pytest.mark.skipif(os.name != "nt", reason="Real Windows read-only attribute behavior")
def test_delete_readonly_windows_files_and_directories(isolated_output, monkeypatch):
    monkeypatch.delenv("VSC_DATA")
    root = paths.notebook_outputs_dir()
    archive = root.parent / "archived_output_JEVPFN"
    for tree in (root, archive):
        nested = tree / "old-run" / "1. question_preparation" / "questions"
        nested.mkdir(parents=True)
        (nested / "old.json").write_text("old report")
        for path in [*tree.rglob("*"), tree]:
            path.chmod(stat.S_IREAD)
    protected = root.parent / "data" / "jev_cache" / "paid.sqlite3"
    protected.parent.mkdir(parents=True)
    protected.write_bytes(b"keep paid responses")
    protected.chmod(stat.S_IREAD)
    attributes = protected.stat().st_file_attributes
    try:
        result = fresh_start.run(
            local=True, delete=True, include_archives=True, apply=True, writers_stopped=True
        )
        assert result["status"] == "fresh_outputs_ready"
        assert root.is_dir() and not list(root.iterdir())
        assert not archive.exists()
        assert protected.read_bytes() == b"keep paid responses"
        assert protected.stat().st_file_attributes == attributes
    finally:
        protected.chmod(stat.S_IWRITE)


def test_delete_retries_transient_windows_sharing_lock(tmp_path, monkeypatch):
    root = tmp_path / "output_JEVPFN"
    root.mkdir()
    target = root / "locked.log"
    target.write_text("old log")
    unlink = os.unlink
    attempts, sleeps = [], []

    def transient(path, *args, **kwargs):
        if Path(path).name == target.name:
            attempts.append(path)
            if len(attempts) <= 2:
                error = PermissionError(13, "sharing violation", str(path))
                error.winerror = 32
                raise error
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", transient)
    monkeypatch.setattr(fresh_start.time, "sleep", sleeps.append)
    fresh_start.delete_directory(root)
    assert not root.exists()
    assert len(attempts) == 3 and len(sleeps) == 1


def test_delete_persistent_windows_denial_stops_with_actionable_error(tmp_path, monkeypatch):
    root = tmp_path / "output_JEVPFN"
    root.mkdir()
    target = root / "locked.log"
    target.write_text("old log")
    attempts, sleeps = [], []

    def denied(path, *args, **kwargs):
        attempts.append(path)
        error = PermissionError(13, "access denied", str(path))
        error.winerror = 5
        raise error

    monkeypatch.setattr(os, "unlink", denied)
    monkeypatch.setattr(fresh_start.time, "sleep", sleeps.append)
    monkeypatch.setattr(Path, "chmod", lambda *a, **kw: pytest.fail("No ACL/permission changes"))
    with pytest.raises(PermissionError, match="access denied") as caught:
        fresh_start.delete_directory(root)
    assert "retry the same cleanup" in " ".join(caught.value.__notes__)
    assert len(attempts) == 8 and len(sleeps) == 6
    assert target.read_text() == "old log"


@pytest.mark.parametrize("error_type", [PermissionError, OSError])
def test_delete_does_not_repair_posix_permissions_or_suppress_io_errors(
    tmp_path, monkeypatch, error_type
):
    error_number = errno.EACCES if error_type is PermissionError else errno.EIO
    root = tmp_path / "output_JEVPFN"
    root.mkdir()
    target = root / "evidence"
    target.write_text("keep")
    attempts = []

    def denied(path, *args, **kwargs):
        attempts.append(path)
        # POSIX rmtree attaches a filename to the exception. Real OS exceptions
        # retain errno/strerror; a one-argument OSError loses its message then.
        error = error_type(error_number, "not a Windows attribute error", str(path))
        error.filename = str(root)
        raise error

    monkeypatch.setattr(os, "unlink", denied)
    monkeypatch.setattr(Path, "chmod", lambda *a, **kw: pytest.fail("No chmod"))
    monkeypatch.setattr(fresh_start.time, "sleep", lambda _: pytest.fail("No retry"))
    with pytest.raises(error_type, match="not a Windows") as caught:
        fresh_start.delete_directory(root)
    assert type(caught.value) is error_type and caught.value.errno == error_number
    assert len(attempts) == 1 and target.read_text() == "keep"


@pytest.mark.parametrize("unsafe", ["outside", "junction", "mount"])
def test_delete_error_recovery_rechecks_path_boundaries(tmp_path, monkeypatch, unsafe):
    root = tmp_path / "output_JEVPFN"
    root.mkdir()
    target = (tmp_path if unsafe == "outside" else root) / "protected"
    target.write_text("keep")
    if unsafe in {"junction", "mount"}:
        attribute = "is_junction" if unsafe == "junction" else "is_mount"
        original = getattr(Path, attribute, lambda _: False)
        monkeypatch.setattr(Path, attribute, lambda p: p == target or original(p), raising=False)

    def failure(path, *, onexc=None, onerror=None):
        error = PermissionError(13, "access denied", str(target))
        error.winerror = 5
        if onexc:
            onexc(os.unlink, str(target), error)
        else:
            onerror(os.unlink, str(target), (type(error), error, None))

    monkeypatch.setattr(fresh_start.shutil, "rmtree", failure)
    monkeypatch.setattr(Path, "chmod", lambda *a, **kw: pytest.fail("No chmod outside safe output"))
    with pytest.raises(ValueError, match="outside|Linked|mounted"):
        fresh_start.delete_directory(root)
    assert target.read_text() == "keep"
