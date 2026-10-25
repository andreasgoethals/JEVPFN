"""A native-GPFS GPU receipt can retire only its unchanged durable source image."""

import copy
import json
import shutil

import pytest

from src.cluster import containers, runtime_profiles, scratch_stage
from src.utils import paths
from src.utils.serialization import digest, write_json


@pytest.fixture(params=["intern", "tabpfn_3_5"])
def relocated(request, isolated_output, monkeypatch):
    provider = request.param
    durable = isolated_output / "lustre_project"
    scratch = isolated_output / "gpfs_scratch"
    durable.mkdir()
    scratch.mkdir()
    monkeypatch.setenv("JEVPFN_STAGING_ROOT", str(durable))
    monkeypatch.setenv("JEVPFN_DURABLE_ROOT", str(durable))
    monkeypatch.setenv("VSC_SCRATCH_GPFS1", str(scratch))
    monkeypatch.setenv("JEVPFN_RUNTIME_PROFILE", runtime_profiles.BLACKWELL)
    monkeypatch.setenv("SLURM_JOB_ID", "fixture-build")
    source = containers.container_dir(provider)
    for name, content in {
        "bin/python": "#!/bin/bash\ntrue\n",
        "base.sif": "fixture container, never executed",
        "install.sqfs": "fixture environment, never executed",
        "common.sh": f'IMAGE="{source}/install.sqfs"\n',
    }.items():
        file = source / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content, encoding="utf-8")
    native = containers.native_dir(provider)
    (native / "bin").mkdir(parents=True)
    (native / "bin/python").write_text("obsolete native fixture", encoding="utf-8")
    monkeypatch.setattr(
        containers,
        "probe",
        lambda *args: {
            "python": "3.12.0",
            "packages": {"torch": "fixture"},
            "torch_cuda": "13.0",
            "sentinel_visible": False,
        },
    )
    original = containers.seal(provider, wrapped_existing=True)
    _, work = scratch_stage.roots(provider)
    destination = work / "containers" / runtime_profiles.BLACKWELL / provider
    shutil.copytree(source, destination)
    relocated_manifest = containers.relocate(source, destination)
    monkeypatch.setenv("JEVPFN_CONTAINER_PROVIDER", provider)
    monkeypatch.setenv("JEVPFN_CONTAINER_ID", relocated_manifest["container_id"])
    with scratch_stage.staging_environment(work):
        execution = containers.active_identity()
    monkeypatch.delenv("SLURM_JOB_ID")
    monkeypatch.setattr(containers, "require_idle_vsc", lambda: None)
    return provider, original, execution, source, destination, native


def _save_gpu_evidence(provider, execution):
    base = paths.manifests_dir("experiment_0")
    if provider in containers.PREDICTORS:
        settings = {"model": provider}
        target = base / "after_extraction" / f"predictor_{provider}_fixture.json"
        report = {
            "model": provider,
            "status": "predictor_contracts_passed_requires_full_fold_profile",
            "execution_container": execution,
        }
    else:
        settings = {"request_configuration": {"execution_container": execution}}
        target = base / "before_extraction/gpu_contracts/fixture.json"
        report = {
            "provider": provider,
            "status": "gpu_contracts_passed_requires_review",
            "runtime_token_count_parity": True,
            "semantic_diagnostics": {"passed": True},
        }
    write_json(target, {**report, "settings": settings, "settings_sha256": digest(settings)})


def test_native_gpu_receipt_authorizes_only_its_durable_environment(relocated):
    provider, original, execution, source, destination, native = relocated
    assert execution["container_id"] != original["container_id"]
    assert execution["relocation"]["source_container_id"] == original["container_id"]
    _save_gpu_evidence(provider, execution)
    assert len(containers.retire(provider)["gpu_evidence"]) == 1
    assert (native / "bin/python").is_file()
    assert containers.retire(provider, apply=True, writers_stopped=True)["applied"]
    assert list(native.iterdir()) == []
    assert (destination / "install.sqfs").read_bytes() == (source / "install.sqfs").read_bytes()
    assert containers.verify(provider)["container_id"] == original["container_id"]


@pytest.mark.parametrize(
    "case", ["missing", "source", "destination", "identity", "bytes", "resealed_image"]
)
def test_unproven_or_changed_native_image_cannot_authorize_retirement(relocated, case):
    provider, original, execution, source, destination, native = relocated
    execution = copy.deepcopy(execution)
    if case == "missing":
        execution.pop("relocation")
    elif case == "source":
        execution["relocation"]["source_container_id"] = "wrong-source"
    elif case == "destination":
        execution["relocation"]["destination"] = str(destination.parent / "other")
    elif case == "identity":
        execution["container_id"] = "wrong-execution"
    elif case == "bytes":
        (destination / "install.sqfs").write_text("corrupted image", encoding="utf-8")
    else:
        (destination / "install.sqfs").write_text("different sealed image", encoding="utf-8")
        marker = destination / containers.MARKER
        changed = json.loads(marker.read_text(encoding="utf-8"))
        changed["files"] = containers.files(destination)
        changed["container_id"] = digest({k: v for k, v in changed.items() if k != "container_id"})
        write_json(marker, changed)
        execution.update(container_id=changed["container_id"], files=changed["files"])
    _save_gpu_evidence(provider, execution)
    assert containers.retire(provider)["gpu_evidence"] == []
    with pytest.raises(ValueError, match="GPU evidence"):
        containers.retire(provider, apply=True, writers_stopped=True)
    assert (native / "bin/python").read_text(encoding="utf-8") == "obsolete native fixture"
    assert containers.verify(provider)["container_id"] == original["container_id"]
