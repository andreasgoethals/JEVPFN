"""Only this allocation's reproducible scratch leaf may be retired."""

import json

import pytest

from src.cluster import runtime_cache as cache


def setup(tmp_path, monkeypatch, cluster="mindwell"):
    for name in ("node", "gpfs", "lustre"):
        (tmp_path / name).mkdir()
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", cluster)
    monkeypatch.setenv("VSC_SCRATCH_GPFS1", str(tmp_path / "gpfs"))
    monkeypatch.setenv("VSC_SCRATCH_LUSTRE1", str(tmp_path / "lustre"))
    monkeypatch.setenv("VSC_SCRATCH", str(tmp_path / "lustre"))


def test_mindwell_uses_gpfs_and_only_removes_its_owned_leaf(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    own = cache.create()
    other = tmp_path / "gpfs/other-project"
    other.mkdir()
    (own / "compiled-kernel").write_text("rebuildable")
    assert own.parent == tmp_path / "gpfs"
    cache.cleanup(own)
    assert not own.exists() and other.is_dir()
    cache.cleanup(own)  # A repeated exit hook is harmless.


def test_node_local_preferred_and_foreign_job_or_changed_receipt_refused(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch, "wice")
    assert cache.native_root() == tmp_path / "lustre"
    monkeypatch.setenv("VSC_SCRATCH_NODE", str(tmp_path / "node"))
    own = cache.create()
    assert own.parent == tmp_path / "node"
    monkeypatch.setenv("SLURM_JOB_ID", "456")
    with pytest.raises(ValueError, match="escaped"):
        cache.cleanup(own)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    marker = own / cache.MARKER
    saved = json.loads(marker.read_text())
    marker.write_text(json.dumps({**saved, "uid": "another-owner"}))
    with pytest.raises(ValueError, match="another allocation"):
        cache.cleanup(own)
    assert own.exists()


def test_missing_native_scratch_never_falls_back_to_data_or_project(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    monkeypatch.delenv("VSC_SCRATCH_GPFS1")
    with pytest.raises(ValueError, match="no explicit native scratch"):
        cache.create()
    monkeypatch.delenv("SLURM_JOB_ID")
    with pytest.raises(RuntimeError, match="Slurm allocation"):
        cache.create()
