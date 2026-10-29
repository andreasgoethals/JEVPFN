"""CPU-image integrity and environment isolation before cluster cutover."""

import json

import pytest

from src.cluster import control_runtime as control


def test_host_bootstrap_detects_modified_image_and_manifest(tmp_path):
    image = tmp_path / "control-v1"
    image.mkdir()
    dependency = image / "environment.sqfs"
    dependency.write_bytes(b"immutable dependencies")
    body = dict(
        provider="control", runtime_profile="cpu-control-v1", files=control.image_files(image)
    )
    body["container_id"] = control.digest(body)
    marker = image / control.MARKER
    marker.write_text(json.dumps(body))
    assert control.verify(image)["container_id"] == body["container_id"]
    dependency.write_bytes(b"changed")
    with pytest.raises(ValueError, match="sealed identity"):
        control.verify(image)
    dependency.write_bytes(b"immutable dependencies")
    body["provider"] = "another"
    marker.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="sealed identity"):
        control.verify(image)


def test_nested_child_preserves_allocation_and_clears_parent_container_state(monkeypatch):
    for key in ("CONDA_PREFIX", "CONDA_SHLVL", "_CONDA_ROOT", "_CE_CONDA", "APPTAINER_BIND"):
        monkeypatch.setenv(key, "parent-only")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    env = control.child_environment()
    assert env["CUDA_VISIBLE_DEVICES"] == "0" and env["SLURM_JOB_ID"] == "123"
    assert all(
        key not in env
        for key in ("CONDA_PREFIX", "CONDA_SHLVL", "_CONDA_ROOT", "_CE_CONDA", "APPTAINER_BIND")
    )
    assert control.os.environ["CONDA_PREFIX"] == "parent-only"


def test_controller_reuses_its_interpreter_but_preserves_model_wrapper(monkeypatch):
    monkeypatch.setenv("JEVPFN_CONTROL_IMAGE", "/project/JEVPFN/containers/control-v1")
    command = ["/project/JEVPFN/containers/control-v1/bin/python", "-m", "src.cluster.monitor"]
    assert control.child_command(command) == [control.sys.executable, "-m", "src.cluster.monitor"]
    model = ["/gpfs/work/containers/liquid/bin/python", "-m", "src.jev.self_hosted"]
    assert control.child_command(model) == model


def test_integrity_checks_use_the_readonly_unmasked_host_view(tmp_path, monkeypatch):
    source, unmasked = tmp_path / "source", tmp_path / "unmasked"
    source.mkdir()
    unmasked.mkdir()
    for root in (source, unmasked):
        (root / "env.sqfs").write_bytes(b"sealed image")
        (root / "_bin").mkdir()
        (root / "_bin/python").write_bytes(b"host wrapper")
    body = dict(
        provider="control", runtime_profile="cpu-control-v1", files=control.image_files(source)
    )
    body["container_id"] = control.digest(body)
    (source / control.MARKER).write_text(json.dumps(body))
    (source / "_bin/python").write_bytes(b"image-masked executable")
    monkeypatch.setenv("JEVPFN_CONTROL_IMAGE", str(source))
    monkeypatch.setattr(control, "HOST_VIEW", unmasked)
    assert control.verify(source)["container_id"] == body["container_id"]
    (unmasked / "env.sqfs").write_bytes(b"changed host image")
    with pytest.raises(ValueError, match="sealed identity"):
        control.verify(source)


def test_install_allocation_can_ignore_only_its_own_cluster_job(isolated_output, monkeypatch):
    from src.cluster import maintenance

    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "wice")
    monkeypatch.setattr(maintenance, "storage_policy_failures", lambda _: [])
    monkeypatch.setattr(
        maintenance.subprocess,
        "check_output",
        lambda command, **k: (
            "123|jevpfn-input-install|/JEVPFN\n"
            if "--clusters=wice" in command
            else "456|creditpfn|/other\n"
        ),
    )
    maintenance.require_idle_vsc(allow_current=True)
    monkeypatch.setattr(
        maintenance.subprocess,
        "check_output",
        lambda command, **k: (
            "123|jevpfn-input-install|/JEVPFN\n"
            if "--clusters=wice" in command
            else "123|jevpfn-features|/JEVPFN\n"
        ),
    )
    with pytest.raises(RuntimeError, match="123"):
        maintenance.require_idle_vsc(allow_current=True)


def test_retirement_requires_matching_b200_evidence_and_preserves_other_environments(
    isolated_output, monkeypatch
):
    from src.cluster import maintenance
    from src.utils import paths
    from src.utils.serialization import write_json

    monkeypatch.setattr(maintenance, "require_idle_vsc", lambda: None)
    data = isolated_output / "personal-data"
    legacy = data / "conda_envs/JEVPFN"
    (legacy / "conda-meta").mkdir(parents=True)
    (legacy / "conda-meta/history").write_text("recorded bootstrap")
    other = data / "conda_envs/other-project"
    other.mkdir()
    (other / "preserve").write_text("untouched")
    monkeypatch.setenv("VSC_DATA", str(data))
    image = paths.checkpoints_dir().parent / "containers" / control.LEAF
    image.mkdir(parents=True)
    (image / "dependencies.sqfs").write_bytes(b"image")
    body = dict(
        provider="control", runtime_profile="cpu-control-v1", files=control.image_files(image)
    )
    body["container_id"] = control.digest(body)
    write_json(image / control.MARKER, body)
    write_json(image.parent / "control.active.json", dict(container_id=body["container_id"]))
    receipt = paths.manifests_dir("experiment_0") / "before_extraction/control_gpu_777.json"
    report = dict(
        status="control_b200_passed",
        job_id="777",
        source_container_id="wrong",
        gpu=dict(basic_kernel_check="passed"),
    )
    write_json(receipt, report)
    with pytest.raises(ValueError, match="matching real B200"):
        control.retire_legacy("777")
    assert legacy.exists()
    write_json(receipt, {**report, "source_container_id": body["container_id"]})
    assert control.retire_legacy("777")["status"] == "legacy_control_retired"
    assert not legacy.exists() and (other / "preserve").read_text() == "untouched"
