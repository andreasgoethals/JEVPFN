import pytest

from src.cluster import containers
from src.utils import paths
from src.utils.serialization import digest, write_json


@pytest.fixture
def packed(isolated_output, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-build")
    root = containers.container_dir("cloudflare")
    for name in ("bin/python", "base.sif", "img.sqfs"):
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"synthetic fixture, never executed")
    native = containers.native_dir("cloudflare")
    (native / "bin").mkdir(parents=True)
    (native / "bin/python").write_bytes(b"original fixture")
    monkeypatch.setattr(
        containers,
        "probe",
        lambda *args: {
            "python": "3.12.0",
            "packages": {"torch": "fixture"},
            "sentinel_visible": False,
        },
    )
    containers.seal("cloudflare", wrapped_existing=True)
    return root, native


def test_image_and_wrapper_hashes_are_checked_and_no_overwrite(packed):
    root, _ = packed
    assert containers.verify("cloudflare")["wrapped_existing"]
    with pytest.raises(ValueError, match="already exists"):
        containers.seal("cloudflare")
    (root / "bin/python").write_bytes(b"changed wrapper")
    with pytest.raises(ValueError, match="fingerprint"):
        containers.verify("cloudflare")


def test_source_masking_must_hide_the_unpacked_environment(isolated_output, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-build")
    root = containers.container_dir("intern")
    root.mkdir(parents=True)
    original = containers.native_dir("intern")
    original.mkdir(parents=True)
    monkeypatch.setattr(
        containers,
        "probe",
        lambda *a: {"python": "3.12.0", "packages": {}, "sentinel_visible": True},
    )
    with pytest.raises(ValueError, match="unpacked source"):
        containers.seal("intern", wrapped_existing=True)
    assert list(original.iterdir()) == []
    assert not (root / containers.MARKER).exists()


def test_retirement_needs_matching_gpu_evidence_and_never_touches_other_data(packed, monkeypatch):
    root, native = packed
    monkeypatch.setattr(containers, "require_idle_vsc", lambda: None)
    with pytest.raises(ValueError, match="GPU evidence"):
        containers.retire("cloudflare", apply=True, writers_stopped=True)
    assert (native / "bin/python").is_file()
    marker = containers.verify("cloudflare")
    settings = {
        "request_configuration": {"execution_container": {"container_id": marker["container_id"]}}
    }
    evidence = paths.manifests_dir("experiment_0") / "before_extraction/gpu_contracts/fixture.json"
    write_json(
        evidence,
        {
            "provider": "cloudflare",
            "status": "gpu_contracts_passed_requires_review",
            "runtime_token_count_parity": True,
            "settings": settings,
            "settings_sha256": digest(settings),
            "semantic_diagnostics": {"passed": True},
        },
    )
    other = containers.native_dir("intern")
    other.mkdir()
    (other / "preserved").write_text("keep")
    assert containers.retire("cloudflare")["applied"] is False
    assert containers.retire("cloudflare", apply=True, writers_stopped=True)["applied"]
    assert native.is_dir() and list(native.iterdir()) == []
    assert (other / "preserved").read_text() == "keep"
    assert containers.verify("cloudflare")["container_id"] == marker["container_id"]


def test_active_identity_rejects_different_launch_image(packed, monkeypatch):
    marker = containers.verify("cloudflare")
    monkeypatch.setenv("JEVPFN_CONTAINER_PROVIDER", "cloudflare")
    monkeypatch.setenv("JEVPFN_CONTAINER_ID", marker["container_id"])
    assert containers.active_identity()["container_id"] == marker["container_id"]
    monkeypatch.setenv("JEVPFN_CONTAINER_ID", "changed")
    with pytest.raises(ValueError, match="launch identity"):
        containers.active_identity()


def test_external_wrapper_symlink_is_rejected(packed, tmp_path):
    root, _ = packed
    outside = tmp_path / "outside"
    outside.write_text("external")
    try:
        (root / "bad_link").symlink_to(outside)
    except OSError:
        pytest.skip("Symlink creation is unavailable")
    with pytest.raises(ValueError, match="external symlink"):
        containers.verify("cloudflare")


def test_image_executable_links_are_hashed_without_reading_unpacked_files(packed):
    import yaml

    root, native = packed
    config = root / "share/conf.yaml"
    config.parent.mkdir()
    config.write_text(yaml.safe_dump({"installation_path": str(native)}))
    link = root / "_bin/python"
    link.parent.mkdir()
    try:
        link.symlink_to(native / "bin/python")
    except OSError:
        pytest.skip("Symlink creation is unavailable")
    before = containers.files(root)
    (native / "bin/python").unlink()
    assert containers.files(root) == before
