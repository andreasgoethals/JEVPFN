"""No GPU or downloads: exercise profile isolation and safe wrapper relocation."""

import io
import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.cluster import containers
from src.cluster import runtime_profiles as rp
from src.cluster.container_templates import MINIFORGE_VERSION, pinned_miniforge
from src.utils.serialization import digest, write_json


def test_profiles_do_not_overwrite_h100_images(isolated_output, monkeypatch):
    legacy = containers.container_dir("intern")
    blackwell = containers.container_dir("intern", profile=rp.BLACKWELL)
    assert blackwell == legacy.parent / rp.BLACKWELL / "intern"
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "mindwell")
    assert containers.container_dir("intern") == blackwell
    with pytest.raises(ValueError, match="Mindwell"):
        containers.container_dir("intern", profile=rp.LEGACY)
    with pytest.raises(ValueError, match="CUDA 13.0"):
        rp.validate_build({"torch_cuda": "12.6"})
    rp.validate_build({"torch_cuda": "13.0"})


@pytest.mark.parametrize(
    "listing", ["580.105.08\n", "580.105.08\n580.105.08\n", "", "unknown\n", "580.1\n590.1\n"]
)
def test_driver_query_requires_unambiguous_numeric_result(monkeypatch, listing):
    monkeypatch.setattr(rp.subprocess, "check_output", lambda *args, **kwargs: listing)
    if listing and set(listing.splitlines()) == {"580.105.08"}:
        assert rp.driver_version() == "580.105.08"
    else:
        with pytest.raises(RuntimeError, match="driver version"):
            rp.driver_version()


@pytest.mark.parametrize(
    "case", ["pass", "cpu", "h100", "missing_kernel", "wrong_result", "old_driver"]
)
def test_blackwell_kernel_contract_requires_allocated_architecture_and_math(monkeypatch, case):
    monkeypatch.setenv("JEVPFN_RUNTIME_PROFILE", rp.BLACKWELL)
    monkeypatch.setattr(
        rp, "driver_version", lambda: "575.1" if case == "old_driver" else "580.105.08"
    )
    calls = []

    class Tensor:
        def __matmul__(self, other):
            calls.append("matmul")
            return self

        def float(self):
            return self

        def mean(self):
            return self

        def item(self):
            return 31.0 if case == "wrong_result" else 32.0

    fake = SimpleNamespace(
        __version__="fixture",
        version=SimpleNamespace(cuda="13.0"),
        bfloat16="bf16",
        ones=lambda *args, **kw: Tensor(),
        cuda=SimpleNamespace(
            is_available=lambda: case != "cpu",
            get_device_capability=lambda: (9, 0) if case == "h100" else (10, 0),
            get_device_name=lambda: "H100" if case == "h100" else "NVIDIA B200",
            get_arch_list=lambda: ["sm_90"] if case == "missing_kernel" else ["sm_100"],
            synchronize=lambda: calls.append("sync"),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    if case == "pass":
        assert rp.gpu_check()["basic_kernel_check"] == "passed"
        assert calls == ["matmul", "sync"]
    else:
        with pytest.raises(RuntimeError):
            rp.gpu_check()
        if case != "wrong_result":
            assert calls == []


def _sealed(root, profile=rp.BLACKWELL):
    for name, value in {
        "bin/python": "#!/bin/bash\ntrue\n",
        "base.sif": "image",
        "install.sqfs": "packed environment",
    }.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    write_json(root / "share/conf.yaml", {"installation_path": "/opt/image_conda"})
    (root / "common.sh").write_text(f'IMAGE="{root}/install.sqfs"\nPREFIX=/opt/image_conda\n')
    manifest = {
        "schema": "jevpfn.container.v1",
        "provider": "intern",
        "runtime_profile": profile,
        "runtime": {"torch_cuda": "13.0"},
        "files": containers.files(root),
    }
    manifest["container_id"] = digest(manifest)
    write_json(root / containers.MARKER, manifest)
    return manifest


def test_relocation_validates_copy_keeps_images_and_preserves_image_prefix(tmp_path):
    source, target = tmp_path / "durable", tmp_path / "scratch"
    manifest = _sealed(source)
    shutil.copytree(source, target, symlinks=True)
    result = containers.relocate(source, target)
    assert result["relocation"]["source_container_id"] == manifest["container_id"]
    assert result["container_id"] != manifest["container_id"]
    assert result["files"] == containers.files(target)
    assert str(source) not in (target / "common.sh").read_text()
    assert "/opt/image_conda" in (target / "common.sh").read_text()
    assert (source / "install.sqfs").read_bytes() == (target / "install.sqfs").read_bytes()
    assert json.loads((source / containers.MARKER).read_text()) == manifest
    assert containers.relocate(source, target) == result
    (target / "common.sh").write_text("modified relocated wrapper")
    with pytest.raises(ValueError, match="Previously relocated"):
        containers.relocate(source, target)


def test_relocation_refuses_corrupted_copy_before_mutation(tmp_path):
    source, target = tmp_path / "durable", tmp_path / "scratch"
    _sealed(source)
    shutil.copytree(source, target, symlinks=True)
    original = (target / "common.sh").read_bytes()
    (target / "install.sqfs").write_text("corrupt")
    with pytest.raises(ValueError, match="Destination bytes"):
        containers.relocate(source, target)
    assert (target / "common.sh").read_bytes() == original


def test_relocation_keeps_image_links_but_rewrites_absolute_host_links(tmp_path):
    source, target = tmp_path / "durable", tmp_path / "scratch"
    manifest = _sealed(source)
    (source / "_bin").mkdir()
    image_prefix = tmp_path / "image_only"
    write_json(source / "share/conf.yaml", {"installation_path": str(image_prefix)})
    try:
        (source / "_bin/python").symlink_to(image_prefix / "bin/python")
        (source / "host_alias").symlink_to(source / "common.sh")
    except OSError:
        pytest.skip("Symlink creation unavailable")
    manifest["files"] = containers.files(source)
    manifest["container_id"] = digest({k: v for k, v in manifest.items() if k != "container_id"})
    write_json(source / containers.MARKER, manifest)
    shutil.copytree(source, target, symlinks=True)
    result = containers.relocate(source, target)
    assert result["files"] == containers.files(target)
    assert (target / "host_alias").resolve() == target / "common.sh"
    assert (target / "_bin/python").is_symlink()


def test_relocation_uses_sealed_normalization_for_tykky_double_separators(tmp_path):
    source, target = tmp_path / "durable", tmp_path / "scratch"
    manifest = _sealed(source)
    (source / "_bin").mkdir()
    prefix = tmp_path / "image_only"
    write_json(source / "share/conf.yaml", {"installation_path": str(prefix)})
    link = str(prefix) + "/bin//python"
    try:
        os.symlink(link, source / "_bin/python")
    except OSError:
        pytest.skip("Symlink creation unavailable")
    manifest["files"] = containers.files(source)
    manifest["container_id"] = digest({k: v for k, v in manifest.items() if k != "container_id"})
    write_json(source / containers.MARKER, manifest)
    shutil.copytree(source, target, symlinks=True)
    result = containers.relocate(source, target)
    assert result["files"] == containers.files(target)
    assert (source / "install.sqfs").read_bytes() == (target / "install.sqfs").read_bytes()
    assert containers.relocate(source, target) == result


def test_relocation_still_rejects_a_changed_link(tmp_path):
    source, target = tmp_path / "durable", tmp_path / "scratch"
    manifest = _sealed(source)
    try:
        os.symlink("./common.sh", source / "alias")
    except OSError:
        pytest.skip("Symlink creation unavailable")
    manifest["files"] = containers.files(source)
    manifest["container_id"] = digest({k: v for k, v in manifest.items() if k != "container_id"})
    write_json(source / containers.MARKER, manifest)
    shutil.copytree(source, target, symlinks=True)
    (target / "alias").unlink()
    os.symlink("./install.sqfs", target / "alias")
    with pytest.raises(ValueError, match="Destination bytes.*alias"):
        containers.relocate(source, target)


@pytest.mark.parametrize("missing", [False, True])
def test_wheel_preflight_requires_exact_platform_and_cuda(monkeypatch, missing):
    @contextmanager
    def fake_open(url, timeout):
        package = url.rstrip("/").split("/")[-1]
        version = "2.12.1" if package == "torch" else "0.27.1"
        cuda = "cu126" if missing else "cu130"
        html = f'<a href="/{package}-{version}%2B{cuda}-cp312-cp312-manylinux_2_28_x86_64.whl">wheel</a>'
        yield io.BytesIO(html.encode())

    monkeypatch.setattr(rp, "urlopen", fake_open)
    if missing:
        with pytest.raises(RuntimeError, match="No official Python 3.12"):
            rp.check_wheels("intern", rp.BLACKWELL)
    else:
        assert (
            rp.check_wheels("intern", rp.BLACKWELL)["exact_wheels_available"]["torch"] == "2.12.1"
        )


def test_miniforge_pin_and_http_checksum_guards_replace_latest_lookup():
    source = """[ "$CW_CONDA_VERSION" = "latest" ] && CW_CONDA_VERSION=$(curl -s https://github.com/conda-forge/miniforge/releases/latest)
curl -sL "$url" --output Miniforge_inst.sh &>/dev/null
bash Miniforge_inst.sh -b
"""
    patched = pinned_miniforge(source)
    assert f"CW_CONDA_VERSION={MINIFORGE_VERSION}" in patched
    assert "releases/latest" not in patched
    assert (
        patched.index("curl --fail")
        < patched.index("sha256sum --check")
        < patched.index("bash Miniforge_inst.sh")
    )
    with pytest.raises(ValueError, match="template changed"):
        pinned_miniforge(source.replace("--output Miniforge_inst.sh", "-o changed.sh"))


@pytest.mark.parametrize("failure", ["http", "wrong_checksum"])
def test_bad_miniforge_download_never_executes_installer(tmp_path, failure):
    bash = shutil.which("bash")
    if not bash and (git := shutil.which("git")):
        bash = str(Path(git).parent.parent / "bin/bash.exe")
    if not bash:
        pytest.skip("Bash unavailable")
    template = """[ "$CW_CONDA_VERSION" = "latest" ] && CW_CONDA_VERSION=$(curl -s https://github.com/conda-forge/miniforge/releases/latest)
curl -sL "$url" --output Miniforge_inst.sh &>/dev/null
bash Miniforge_inst.sh
"""
    fake_download = (
        "return 22" if failure == "http" else "printf 'not an installer' > Miniforge_inst.sh"
    )
    script = tmp_path / "check.sh"
    script.write_text(
        "set -euo pipefail\nCW_CONDA_ARCH=Linux-x86_64\n"
        f"curl() {{ {fake_download}; }}\n" + pinned_miniforge(template) + "touch must_not_exist\n",
        encoding="utf-8",
        newline="\n",
    )
    result = subprocess.run(
        [bash, str(script)], cwd=tmp_path, capture_output=True, text=True, timeout=15
    )
    assert result.returncode != 0
    assert not (tmp_path / "must_not_exist").exists()
