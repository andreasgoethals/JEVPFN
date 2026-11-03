"""Deployment moves may change locations, never the scientific representation policy."""

import copy
import json

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
    monkeypatch.setattr(freeze, "source_snapshot", lambda: {"files": {"src/jev/self_hosted.py": "same"}})
    previous = tmp_path / "previous.json"
    write_json(previous, {"frozen_receipt": {"files": {"src/jev/self_hosted.py": "same"}}})
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
