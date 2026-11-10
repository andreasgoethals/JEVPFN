"""Deployment moves may change locations, never the scientific representation policy."""

import copy
import json
import shutil
from pathlib import Path

import pytest

from src.cluster import scheduling
from src.cluster.storage_migration import same_science
from src.utils import paths
from src.utils.serialization import write_json


def test_shared_immutable_paths_keep_mutable_stores_isolated(tmp_path, monkeypatch):
    durable, native = tmp_path / "allocated", tmp_path / "scratch/runtime"
    monkeypatch.setattr(paths, "on_vsc", lambda: True)
    monkeypatch.setenv("JEVPFN_DURABLE_ROOT", str(durable))
    monkeypatch.setenv("JEVPFN_STAGING_ROOT", str(native))
    monkeypatch.setenv("JEVPFN_SHARED_IMMUTABLES", "1")
    assert paths.checkpoints_dir() == durable / "JEVPFN/checkpoints"
    assert paths.raw_dir() == durable / "JEVPFN/data/raw"
    assert paths.dataset_question_path("DATASET") == durable / "JEVPFN/data/decision_inputs/DATASET.sqlite3"
    assert paths.token_audit_dir().is_relative_to(durable)
    assert paths.dataset_cache_path("DATASET").is_relative_to(native)
    assert paths.results_dir(phase="experiment_1").is_relative_to(native)


def test_cpu_execution_follows_explicit_authority(monkeypatch):
    monkeypatch.setenv("JEVPFN_DURABLE_ROOT", "/gpfs1/project/example")
    assert scheduling.cpu_cluster() == "mindwell"
    monkeypatch.setenv("JEVPFN_DURABLE_ROOT", "/lustre1/project/example")
    assert scheduling.cpu_cluster() == "wice"


def test_mindwell_publication_uses_declared_node_disk_for_temporary_unions(tmp_path, monkeypatch):
    from src.cluster.scratch_stage import _publication_temporary_root

    local = tmp_path / "node"
    local.mkdir()
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "mindwell")
    monkeypatch.setenv("SLURM_JOB_ID", "99")
    monkeypatch.setenv("VSC_SCRATCH_NODE", str(local))
    assert _publication_temporary_root(tmp_path / "shared_work") == local


def _node_with_free(tmp_path, monkeypatch, cluster, free_by_root):
    from collections import namedtuple

    from src.cluster import scratch_stage

    local = tmp_path / "node"
    local.mkdir()
    scratch = tmp_path / "scratch"
    work = scratch / "jevpfn-work" / "kev" / "JEVPFN"
    work.mkdir(parents=True)
    monkeypatch.setenv("SLURM_CLUSTER_NAME", cluster)
    monkeypatch.setenv("SLURM_JOB_ID", "99")
    monkeypatch.setenv("VSC_SCRATCH_NODE", str(local))
    usage = namedtuple("usage", "total used free")
    real = shutil.disk_usage

    def disk_usage(path):
        for root, free in free_by_root(local, scratch).items():
            if Path(path).is_relative_to(root):
                return usage(free, 0, free)
        return real(path)

    monkeypatch.setattr(scratch_stage.shutil, "disk_usage", disk_usage)
    return scratch_stage, local, scratch, work


def test_publication_union_falls_back_to_native_scratch_when_node_disk_is_too_small(tmp_path, monkeypatch):
    stage, local, scratch, work = _node_with_free(
        tmp_path, monkeypatch, "mindwell", lambda local, scratch: {local: 2**30, scratch: 2**42})
    assert stage._publication_temporary_root(work, 2**29) == local
    fallback = stage._publication_temporary_root(work, 140 * 2**30)
    assert fallback == scratch / "jevpfn-transfers" / "publication-temporary"
    assert not fallback.is_relative_to(work.parent.parent)  # Outside every workspace.
    assert stage._publication_temporary_root(work) == local  # Unsized callers are unchanged.


def test_publication_union_fallback_respects_the_personal_scratch_budget(tmp_path, monkeypatch):
    stage, _, _, work = _node_with_free(
        tmp_path, monkeypatch, "mindwell", lambda local, scratch: {local: 2**30, scratch: 2**42})
    with pytest.raises(OSError, match="Insufficient native space"):
        stage._publication_temporary_root(work, 600 * 2**30)  # Above the 500 GiB allocation.


def test_wice_publication_never_falls_back_to_cross_mounted_scratch(tmp_path, monkeypatch):
    stage, _, _, work = _node_with_free(
        tmp_path, monkeypatch, "wice", lambda local, scratch: {local: 2**30, scratch: 2**42})
    with pytest.raises(OSError, match="Insufficient native space"):
        stage._publication_temporary_root(work, 140 * 2**30)


def test_explicit_durable_authority_wins_over_old_project_alias(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "on_vsc", lambda: True)
    monkeypatch.setenv("JEVPFN_PROJECT_ROOT", str(tmp_path / "old"))
    monkeypatch.setenv("JEVPFN_DURABLE_ROOT", str(tmp_path / "new"))
    assert paths.staging_root() == tmp_path / "new"
    monkeypatch.setenv("JEVPFN_STAGING_ROOT", str(tmp_path / "native"))
    assert paths.staging_root() == tmp_path / "native"


def test_only_host_execution_identity_can_change():
    original = {"model": "pinned", "request_configuration": {
        "checkpoint_sha256": "weights", "counter_id": "tokens", "batch_size": 2,
        "packages": {"torch": "2.9"}, "execution_container": {"container_id": "old"}}}
    relocated = copy.deepcopy(original)
    relocated["request_configuration"]["execution_container"] = {"container_id": "new"}
    assert same_science(original, relocated)
    for key in ("checkpoint_sha256", "counter_id", "batch_size", "packages"):
        changed = copy.deepcopy(relocated)
        changed["request_configuration"][key] = "different"
        assert not same_science(original, changed)


def test_historical_cuda_field_uses_only_its_exact_pinned_wheel_and_profile():
    from src.cluster.storage_migration import same_installed_runtime

    sealed = {"runtime_profile": "legacy-cu126", "runtime": {"packages": {"torch": "2.9.0+cu126"}}}
    actual = {"packages": {"torch": "2.9.0+cu126"}, "torch_cuda": "12.6"}
    assert same_installed_runtime(sealed, actual)
    assert same_installed_runtime({"runtime": sealed["runtime"]}, actual)
    assert not same_installed_runtime(sealed, {**actual, "torch_cuda": "13.0"})
    assert not same_installed_runtime({**sealed, "runtime_profile": "blackwell-cu130"}, actual)
    assert not same_installed_runtime(sealed, {**actual, "packages": {"torch": "2.10.0+cu126"}})


def test_missing_private_authority_never_infers_a_project(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "on_vsc", lambda: True)
    for name in ("JEVPFN_STAGING_ROOT", "JEVPFN_DURABLE_ROOT", "JEVPFN_PROJECT_ROOT", "VSC_PROJECT_ROOT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("JEVPFN_STORAGE_CONFIG", str(tmp_path / "absent.json"))
    with pytest.raises(ValueError, match="explicitly"):
        paths.staging_root()


def test_migration_rebind_preserves_36_completed_jobs_and_all_encoder_partitions(tmp_path, monkeypatch):
    from src.cluster import (
        experiment_zero,
        feature_admission,
        freeze,
        maintenance,
        storage_migration,
    )
    from src.jev.catalog import SELF_HOSTED

    monkeypatch.setenv("JEVPFN_DURABLE_ROOT", "/gpfs1/project/example")
    monkeypatch.setenv("VSC_SCRATCH_GPFS1", "/gpfs1/scratch/example")
    monkeypatch.setattr(paths, "manifests_dir", lambda *a: tmp_path)
    monkeypatch.setattr(maintenance, "require_idle_vsc", lambda: None)
    monkeypatch.setattr(freeze, "check_if_present", lambda: {"frozen": True, "source_sha256": "new"})
    monkeypatch.setattr(freeze, "runtime_pins", lambda: {})
    monkeypatch.setattr(freeze, "source_snapshot", lambda: {"files": {
        "src/jev/self_hosted.py": "same", "src/cluster/preflight.py": "new native storage policy"}})
    previous = tmp_path / "previous.json"
    write_json(previous, {"frozen_receipt": {"files": {
        "src/jev/self_hosted.py": "same", "src/cluster/preflight.py": "old Lustre storage policy"}}})
    original = feature_admission.plan()
    encoders = [s["name"] for s in original if s["name"].startswith("embeddings_")]
    expanded = feature_admission.partition_encoders(original, encoders)
    completed = set()
    for number, step in enumerate(expanded):
        if step.get("pool") in SELF_HOSTED[:2]:
            step.update(status="completed", job_id=str(1000 + number), scheduler_state="COMPLETED", exit_code="0:0")
            completed.add(step["job_id"])
        elif (step.get("workspace_stage") or step.get("workspace_release")) in SELF_HOSTED[:2]:
            step.update(status="completed", job_id=str(1000 + number))
        else:
            step["status"] = "pending"
    monkeypatch.setattr(experiment_zero, "scheduler_state", lambda *a, **k: ("COMPLETED", "0:0"))
    write_json(tmp_path / "open_weight_campaign.json", {
        "steps": expanded, "parallel_admission": {"policy": {"split_encoders": encoders}}})
    write_json(storage_migration.records() / "retired.json", {"status": "old_project_and_published_native_copies_retired"})
    result = storage_migration.rebind(previous, "authorized fixture")
    saved = json.loads((tmp_path / "open_weight_campaign.json").read_text())
    assert result["reused_providers"] == sorted(SELF_HOSTED[:2])
    assert len(saved["steps"]) == 274
    assert {s["job_id"] for s in saved["steps"] if s.get("job_id")} == completed
    assert len(completed) == 36
    assert len([s for s in saved["steps"] if s["name"].startswith("embeddings_")]) == 54
    assert all(s.get("status", "pending") == "pending" for s in saved["steps"] if s["name"].startswith("embeddings_"))


@pytest.mark.parametrize("changed", [False, True])
def test_retirement_removes_only_verified_old_project_and_preserves_allocation(tmp_path, monkeypatch, changed):
    import os

    from src.cluster import maintenance, project_migration, storage_migration
    from src.utils.serialization import file_sha256

    source, destination = tmp_path / "old/JEVPFN", tmp_path / "new/JEVPFN"
    unrelated = source.parent / "another-project"
    unrelated.mkdir(parents=True)
    (unrelated / "keep").write_bytes(b"other research")
    for p in (source, destination):
        p.mkdir(parents=True)
        (p / "weights").write_bytes(b"verified bytes")
        (p / "data/raw").mkdir(parents=True)
        (p / "data/raw/input.csv").write_bytes(b"preserve original data")
    (source / "data/raw").chmod(0o500)
    root = tmp_path / "records"
    root.mkdir()
    monkeypatch.setattr(storage_migration, "records", lambda: root)
    monkeypatch.setattr(maintenance, "require_idle_vsc", lambda **k: None)
    monkeypatch.setattr(paths, "checkpoints_dir", lambda: destination / "checkpoints")
    monkeypatch.setenv("SLURM_JOB_ID", "fixture")
    monkeypatch.setenv("VSC_SCRATCH_GPFS1", str(tmp_path / "scratch"))
    monkeypatch.setattr(os, "getuid", lambda: source.stat().st_uid, raising=False)
    proof = root / "proof.json"
    write_json(proof, {"passed": True})
    write_json(root / "copy_verified.json", {
        "status": "copied_and_verified_requires_relocation_and_runtime_acceptance",
        "source": str(source), "destination": str(destination),
        "source_inventory": project_migration.inventory(source)})
    write_json(root / "accepted.json", {"status": "accepted_requires_consumer_check_and_retirement",
                                       "destination": str(destination),
                                       "evidence": [{"path": str(proof), "sha256": file_sha256(proof)}]})
    from src.cluster.scratch_stage import MARKER

    work = tmp_path / "scratch/jevpfn-work/fixture/JEVPFN"
    write_json(work / MARKER, {"runtime": "fixture", "durable_root": str(source), "files": {}})
    write_json(root / "workspaces_published.json", {"workspaces": [{
        "runtime": "fixture", "working_root": str(work), "old_marker_sha256": file_sha256(work / MARKER)}]})
    from src.cluster import containers, control_runtime
    from src.utils.serialization import digest

    executing = destination / "containers/control-v1"
    host = tmp_path / "unmasked_control"
    for relative in ("container.sif", "deps.sqfs", "bin/python"):
        p = host / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"host wrapper or immutable image")
    (executing / "bin").mkdir(parents=True)
    (executing / "bin/python").write_bytes(b"masked installed executable")
    seal = {"files": containers.files(host)}
    seal["container_id"] = digest(seal)
    write_json(executing / containers.MARKER, seal)
    monkeypatch.setenv("JEVPFN_CONTROL_IMAGE", str(executing))
    monkeypatch.setattr(control_runtime, "HOST_VIEW", host)
    if changed:
        (source / "weights").write_bytes(b"unreviewed change")
        with pytest.raises(ValueError, match="bytes"):
            storage_migration.retire_old()
        assert source.is_dir()
    else:
        assert storage_migration.retire_old()["status"] == "old_project_and_published_native_copies_retired"
        assert not source.exists()
        assert not work.exists()
    assert source.parent.is_dir()
    assert (unrelated / "keep").read_bytes() == b"other research"
    assert (destination / "weights").read_bytes() == b"verified bytes"
