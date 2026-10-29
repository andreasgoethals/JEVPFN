"""CPU-image integrity and environment isolation before cluster cutover."""

import json

import pytest

from src.cluster import control_runtime as control


def test_host_bootstrap_detects_modified_image_and_manifest(tmp_path):
    image = tmp_path / "control-v1"
    image.mkdir()
    dependency = image / "environment.sqfs"
    dependency.write_bytes(b"immutable dependencies")
    body = dict(provider="control", runtime_profile="cpu-control-v1", files=control.image_files(image))
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
    assert all(key not in env for key in ("CONDA_PREFIX", "CONDA_SHLVL", "_CONDA_ROOT", "_CE_CONDA", "APPTAINER_BIND"))
    assert control.os.environ["CONDA_PREFIX"] == "parent-only"


def test_controller_reuses_its_interpreter_but_preserves_model_wrapper(monkeypatch):
    monkeypatch.setenv("JEVPFN_CONTROL_IMAGE", "/project/JEVPFN/containers/control-v1")
    command = ["/project/JEVPFN/containers/control-v1/bin/python", "-m", "src.cluster.monitor"]
    assert control.child_command(command) == [control.sys.executable, "-m", "src.cluster.monitor"]
    model = ["/gpfs/work/containers/liquid/bin/python", "-m", "src.jev.self_hosted"]
    assert control.child_command(model) == model


def test_install_allocation_can_ignore_only_its_own_cluster_job(isolated_output, monkeypatch):
    from src.cluster import maintenance

    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "wice")
    monkeypatch.setattr(maintenance, "storage_policy_failures", lambda _: [])
    monkeypatch.setattr(maintenance.subprocess, "check_output", lambda *a, **k:
                        "CLUSTER: wice\n123|jevpfn-input-install|/JEVPFN\nCLUSTER: mindwell\n456|creditpfn|/other\n")
    maintenance.require_idle_vsc(allow_current=True)
    monkeypatch.setattr(maintenance.subprocess, "check_output", lambda *a, **k:
                        "CLUSTER: wice\n123|jevpfn-input-install|/JEVPFN\nCLUSTER: mindwell\n123|jevpfn-features|/JEVPFN\n")
    with pytest.raises(RuntimeError, match="123"):
        maintenance.require_idle_vsc(allow_current=True)
