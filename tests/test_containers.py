import pytest

from src.cluster import containers
from src.utils import paths
from src.utils.serialization import digest, write_json


def test_nested_guard_upgrade_preserves_sealed_images_and_detects_repeat(tmp_path):
    import json
    import shutil

    source, target = tmp_path / "sealed", tmp_path / "native"
    (source / "bin").mkdir(parents=True)
    guard = (
        "if  grep -q 'singularity/mnt/session\\|apptainer/mnt/session' /proc/self/mountinfo ;then"
    )
    for name, text in {
        "base.sif": "image",
        "env.sqfs": "dependencies",
        "bin/python": "wrapper",
        "common.sh": "INSTALLATION_PATH=/IMAGE_PREFIX\n"
        + guard
        + "\nexport _CW_IN_CONTAINER=Yes\nfi\n",
    }.items():
        (source / name).write_text(text)
    body = dict(schema="jevpfn.container.v1", provider="fixture", files=containers.files(source))
    body["container_id"] = digest(body)
    write_json(source / containers.MARKER, body)
    shutil.copytree(source, target)
    current = containers.relocate(source, target)
    assert '&& [[ -e "$INSTALLATION_PATH" ]]' in (target / "common.sh").read_text()
    assert guard in (source / "common.sh").read_text()
    assert containers.relocate(source, target) == current
    # A verified older native copy upgrades without changing its image bytes.
    text = (
        (target / "common.sh")
        .read_text()
        .replace(' && [[ -e "$INSTALLATION_PATH" ]]; then', " ;then")
    )
    (target / "common.sh").write_text(text)
    old = json.loads((target / containers.MARKER).read_text())
    old["relocation"]["wrapper_policy_version"] = 1
    old["files"] = containers.files(target)
    old["container_id"] = digest({k: v for k, v in old.items() if k != "container_id"})
    write_json(target / containers.MARKER, old)
    upgraded = containers.relocate(source, target)
    assert upgraded["relocation"]["wrapper_policy_version"] == 2
    assert (target / "env.sqfs").read_bytes() == (source / "env.sqfs").read_bytes()
    assert '&& [[ -e "$INSTALLATION_PATH" ]]' in (target / "common.sh").read_text()


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
    with pytest.raises(ValueError, match="sealed container"):
        containers.discard_unsealed("cloudflare", apply=True, writers_stopped=True)
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


def test_discard_failed_build_is_guarded_and_scoped(isolated_output, monkeypatch):
    failed = containers.container_dir("liquid")
    failed.mkdir(parents=True)
    (failed / "bad.sqfs").write_text("incomplete")
    weights = paths.checkpoints_dir() / "decision_models/liquid/keep"
    weights.parent.mkdir(parents=True)
    weights.write_text("weights")
    native = containers.native_dir("liquid")
    native.mkdir(parents=True)
    (native / "keep").write_text("preserve")
    assert containers.discard_unsealed("liquid")["applied"] is False
    assert failed.exists()
    with pytest.raises(ValueError, match="writers stopped"):
        containers.discard_unsealed("liquid", apply=True)

    def busy():
        raise RuntimeError("jobs remain")

    monkeypatch.setattr(containers, "require_idle_vsc", busy)
    with pytest.raises(RuntimeError, match="jobs remain"):
        containers.discard_unsealed("liquid", apply=True, writers_stopped=True)
    assert failed.exists()
    monkeypatch.setattr(containers, "require_idle_vsc", lambda: None)
    assert containers.discard_unsealed("liquid", apply=True, writers_stopped=True)["applied"]
    assert not failed.exists() and weights.read_text() == "weights" and (native / "keep").exists()


def test_active_identity_rejects_different_launch_image(packed, monkeypatch):
    marker = containers.verify("cloudflare")
    monkeypatch.setenv("JEVPFN_CONTAINER_PROVIDER", "cloudflare")
    monkeypatch.setenv("JEVPFN_CONTAINER_ID", marker["container_id"])
    assert containers.active_identity()["container_id"] == marker["container_id"]
    monkeypatch.setenv("JEVPFN_CONTAINER_ID", "changed")
    with pytest.raises(ValueError, match="launch identity"):
        containers.active_identity()


def test_new_image_needs_explicit_replacement_review_before_retiring_partial_source(
    packed, monkeypatch
):
    root, native = packed
    marker = containers.verify("cloudflare")
    marker["wrapped_existing"] = False
    marker["container_id"] = digest({k: v for k, v in marker.items() if k != "container_id"})
    write_json(root / containers.MARKER, marker)
    with pytest.raises(ValueError, match="allow-replacement"):
        containers.retire("cloudflare")
    assert containers.retire("cloudflare", allow_replacement=True)["native_entries_to_remove"] == 2
    with pytest.raises(ValueError, match="GPU evidence"):
        containers.retire("cloudflare", allow_replacement=True, apply=True, writers_stopped=True)
    assert (native / "bin/python").is_file()


def test_inventory_counts_entries_without_verifying_or_deleting(packed):
    root, native = packed
    result = containers.inventory()
    cf = next(x for x in result["providers"] if x["provider"] == "cloudflare")
    assert cf["native_entries"] == 2 and cf["sealed_container_present"]
    assert (native / "bin/python").exists() and root.exists()


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
