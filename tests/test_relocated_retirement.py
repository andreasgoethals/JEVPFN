"""Retirement follows verified B200 relocation without trusting a claimed source ID."""

import copy
import json
import shutil

import pytest

from src.cluster import containers, scratch_stage
from src.utils import paths
from src.utils.serialization import digest, write_json


@pytest.fixture(params=["cloudflare", "tabstar"])
def relocated_runtime(request, isolated_output, monkeypatch):
    provider = request.param
    durable = isolated_output / "lustre"
    scratch = isolated_output / "gpfs"
    durable.mkdir()
    scratch.mkdir()
    monkeypatch.setenv("JEVPFN_STAGING_ROOT", str(durable))
    monkeypatch.setenv("JEVPFN_DURABLE_ROOT", str(durable))
    monkeypatch.setenv("VSC_SCRATCH_GPFS1", str(scratch))
    monkeypatch.setenv("JEVPFN_RUNTIME_PROFILE", "blackwell-cu130")
    monkeypatch.setenv("SLURM_JOB_ID", "fixture-build")
    monkeypatch.setattr(containers, "require_idle_vsc", lambda: None)
    monkeypatch.setattr(
        containers,
        "probe",
        lambda *args: {
            "python": "3.12.0", "torch_cuda": "13.0",
            "packages": {"torch": "fixture"}, "sentinel_visible": False,
        },
    )
    source = containers.container_dir(provider)
    for name in ("bin/python", "base.sif", "img.sqfs"):
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"synthetic fixture; never executed")
    (source / "common.sh").write_text(f"IMAGE={source}/img.sqfs\n", encoding="utf-8")
    native = containers.native_dir(provider)
    (native / "bin").mkdir(parents=True)
    (native / "bin/python").write_bytes(b"unpacked environment; never executed")
    source_seal = containers.seal(provider, wrapped_existing=True)
    _, work = scratch_stage.roots(provider)
    target = work / "containers/blackwell-cu130" / provider
    shutil.copytree(source, target)
    relocated = containers.relocate(source, target)
    assert relocated["container_id"] != source_seal["container_id"]
    monkeypatch.setenv("JEVPFN_CONTAINER_PROVIDER", provider)
    monkeypatch.setenv("JEVPFN_CONTAINER_ID", relocated["container_id"])
    with scratch_stage.staging_environment(work):
        identity = containers.active_identity()
    assert identity["relocation"]["source_container_id"] == source_seal["container_id"]
    return provider, source, native, target, identity


def write_evidence(provider, identity):
    settings = {"request_configuration": {"execution_container": identity}}
    body = {"settings": settings, "settings_sha256": digest(settings)}
    root = paths.manifests_dir("experiment_0")
    if provider in containers.PREDICTORS:
        target = root / "after_extraction" / f"predictor_{provider}_fixture.json"
        body.update(
            model=provider,
            status="predictor_contracts_passed_requires_full_fold_profile",
            execution_container=identity,
        )
    else:
        target = root / "before_extraction/gpu_contracts/fixture.json"
        body.update(
            provider=provider,
            status="gpu_contracts_passed_requires_review",
            runtime_token_count_parity=True,
            semantic_diagnostics={"passed": True},
        )
    write_json(target, body)


def test_native_gpu_evidence_retires_only_original_environment(relocated_runtime):
    provider, source, native, target, identity = relocated_runtime
    source_before = containers.files(source)
    target_before = containers.files(target)
    write_evidence(provider, identity)
    assert len(containers.retire(provider)["gpu_evidence"]) == 1
    assert containers.retire(provider, apply=True, writers_stopped=True)["applied"]
    assert native.is_dir() and list(native.iterdir()) == []
    assert containers.files(source) == source_before
    assert containers.files(target) == target_before


@pytest.mark.parametrize("mutation", ["source_id", "destination", "missing_proof", "wrapper", "image"])
def test_relocation_cannot_authorize_a_different_image(relocated_runtime, mutation):
    provider, _, native, target, original = relocated_runtime
    identity = copy.deepcopy(original)
    if mutation == "source_id":
        identity["relocation"]["source_container_id"] = "other-image"
    elif mutation == "destination":
        identity["relocation"]["destination"] += "-other"
    elif mutation == "missing_proof":
        identity.pop("relocation")
    elif mutation == "wrapper":
        (target / "common.sh").write_text("modified after GPU evidence", encoding="utf-8")
    else:
        # Even a self-consistent native seal and matching report must not change image bytes.
        (target / "img.sqfs").write_bytes(b"different model runtime")
        seal = json.loads((target / containers.MARKER).read_text(encoding="utf-8"))
        seal["files"] = containers.files(target)
        seal["container_id"] = digest({k: v for k, v in seal.items() if k != "container_id"})
        write_json(target / containers.MARKER, seal)
        identity.update(container_id=seal["container_id"], files=seal["files"])
    write_evidence(provider, identity)
    assert containers.retire(provider)["gpu_evidence"] == []
    with pytest.raises(ValueError, match="GPU evidence"):
        containers.retire(provider, apply=True, writers_stopped=True)
    assert (native / "bin/python").is_file()
