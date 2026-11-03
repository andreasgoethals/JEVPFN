"""Migration copies preserve data, link semantics and WAL safety; no VSC calls."""

import os
import sqlite3

import pytest

from src.cluster import project_migration as migration
from src.cluster import storage_config


def test_inventory_and_relocated_internal_links_preserve_identity(tmp_path):
    source = tmp_path / "old/JEVPFN"
    target = tmp_path / "new/JEVPFN"
    for p in (source, target):
        p.mkdir(parents=True)
        (p / "weights").write_bytes(b"exact weights")
    try:
        (source / "link").symlink_to(source / "weights")
        (target / "link").symlink_to(source / "weights")
    except OSError:
        pytest.skip("Symlink privilege unavailable")
    expected = migration.inventory(source)
    migration.compare(expected, migration.inventory(target), source, target)
    assert migration.rewrite_links(source, target) == ["link"]
    migration.compare(expected, migration.inventory(target), source, target, relocated=True)
    (target / "weights").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="bytes"):
        migration.compare(expected, migration.inventory(target), source, target, relocated=True)


def test_sqlite_wal_requires_snapshot_or_quiescence(tmp_path):
    p = tmp_path / "fixture.sqlite3"
    with sqlite3.connect(p) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE records (value)")
        db.execute("INSERT INTO records VALUES ('preserve')")
        db.commit()
        with pytest.raises(ValueError, match="uncheckpointed"):
            migration.verify_sqlite(tmp_path)
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        assert migration.verify_sqlite(tmp_path) == ["fixture.sqlite3"]


def test_checkpoint_rewrite_leaves_container_image_prefixes_to_sealed_relocator(tmp_path):
    source, target = tmp_path / "old/JEVPFN", tmp_path / "new/JEVPFN"
    target.mkdir(parents=True)
    link = target / "containers/legacy/_bin/python"
    link.parent.mkdir(parents=True)
    try:
        link.symlink_to(source / "environments/legacy/bin/python")
        (target / "checkpoint").symlink_to(source / "weights")
    except OSError:
        pytest.skip("Symlink privilege unavailable")
    assert migration.rewrite_links(source, target) == ["checkpoint"]
    assert os.readlink(link).removeprefix("\\\\?\\") == str(source / "environments/legacy/bin/python")


def test_shared_root_configuration_does_not_infer_site_allocation(tmp_path, monkeypatch):
    import json

    file = tmp_path / "private.json"
    monkeypatch.setenv("JEVPFN_STORAGE_CONFIG", str(file))
    monkeypatch.setenv("VSC_PROJECT_GPFS1", "/gpfs1/project")
    assert storage_config.configured_root() is None
    file.write_text(json.dumps({"durable_root": str(tmp_path / "allocated")}))
    assert storage_config.configured_root() == tmp_path / "allocated"
    monkeypatch.setenv("JEVPFN_DURABLE_ROOT", str(tmp_path / "explicit"))
    assert storage_config.configured_root() == tmp_path / "explicit"


def test_migration_rejects_parent_allocation_and_nested_destination(tmp_path):
    source = tmp_path / "allocation/JEVPFN"
    source.mkdir(parents=True)
    with pytest.raises(ValueError, match="exact"):
        migration.roots(source.parent, tmp_path / "other")
    with pytest.raises(ValueError, match="disjoint"):
        migration.roots(source, source / "child/JEVPFN")


def test_hardlinks_are_kept_and_checked(tmp_path):
    source, target = tmp_path / "a", tmp_path / "b"
    for root in (source, target):
        root.mkdir()
        (root / "a").write_bytes(b"same")
    os.link(source / "a", source / "b")
    (target / "b").write_bytes(b"same")
    with pytest.raises(ValueError, match="hardlink"):
        migration.compare(migration.inventory(source), migration.inventory(target), source, target)


def test_native_verification_refuses_a_stale_same_size_destination(tmp_path, monkeypatch):
    import json
    import shutil

    source, target = tmp_path / "a/JEVPFN", tmp_path / "b/JEVPFN"
    source.mkdir(parents=True)
    target.parent.mkdir()
    (source / "weights").write_bytes(b"correct")
    receipt = tmp_path / "receipt.json"
    monkeypatch.setenv("SLURM_JOB_ID", "copy")
    monkeypatch.setattr(migration.subprocess, "run", lambda *a, **k: shutil.copytree(source, target, dirs_exist_ok=True))
    migration.copy(source, target, receipt, writers_stopped=True, defer_verification=True)
    assert json.loads(receipt.read_text())["status"] == "copied_requires_native_verification"
    (target / "weights").write_bytes(b"CORRUPT")
    with pytest.raises(ValueError, match="bytes"):
        migration.verify_copy(receipt)
    assert json.loads(receipt.read_text())["status"] == "copied_requires_native_verification"
    (target / "weights").write_bytes(b"correct")
    monkeypatch.setenv("SLURM_JOB_ID", "native-check")
    assert migration.verify_copy(receipt)["verification_job_id"] == "native-check"


def test_node_local_integrity_copies_are_verified_and_removed(tmp_path):
    root, local = tmp_path / "project", tmp_path / "node"
    root.mkdir()
    local.mkdir()
    expected = {}
    for name in ("one.sqlite3", "two.sqlite3"):
        p = root / name
        db = sqlite3.connect(p)
        db.execute("CREATE TABLE values_to_preserve (value)")
        db.execute("INSERT INTO values_to_preserve VALUES ('original')")
        db.commit()
        db.close()
        expected[name] = migration.sha(p)
    assert migration.verify_sqlite(root, temporary_root=local, workers=2, expected_hashes=expected) == sorted(expected)
    assert {p.name: migration.sha(p) for p in root.iterdir()} == expected
    assert not list(local.iterdir())


def test_allocated_group_permissions_do_not_make_raw_inputs_writable(tmp_path):
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    p = raw / "input.csv"
    p.write_bytes(b"immutable input")
    assert migration.protect_raw_permissions(tmp_path) == 2
    try:
        assert not raw.stat().st_mode & 0o222
        assert not p.stat().st_mode & 0o222
        assert p.read_bytes() == b"immutable input"
    finally:
        raw.chmod(0o700)
        p.chmod(0o600)
