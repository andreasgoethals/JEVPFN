import json

import pytest

from src.cluster import accept_features as acceptance
from src.utils import paths
from src.utils.acceptance import reviewed_acceptance
from src.utils.serialization import digest, write_json


@pytest.fixture
def evidence(isolated_output, monkeypatch):
    from src.cluster.evidence import current_identity

    validation = current_identity()
    root = paths.manifests_dir("experiment_0")
    before = root / "before_extraction"
    inputs = [{"dataset_id": "fixture", "input_set_id": "v3", "input_file_sha256": "sha"}]
    monkeypatch.setattr(acceptance, "selected_inputs", lambda: inputs)
    write_json(
        before / "tokens_liquid_fixed.json",
        {
            "provider": "liquid",
            "status": "counted_all_inputs",
            "overflow_unique_inputs": 0,
            "audit_id": "token",
            "datasets": inputs,
        },
    )
    identity = {"provider": "liquid", "version": "pin"}
    common = {
        "provider": "liquid",
        "token_audit_id": "token",
        "settings_sha256": digest(identity),
        "settings": identity,
        "validation_identity": validation,
    }
    write_json(
        before / "gpu_contracts/liquid_fixed.json",
        common
        | {
            "status": "gpu_contracts_passed_requires_review",
            "runtime_token_count_parity": True,
            "semantic_diagnostics": {"passed": True},
            "prompt_robustness": {"structurally_valid": True},
            "real_prompt_robustness": {"structurally_valid": True},
            "batch_parity": {"passed": True},
            "real_datasets": [{"dataset_id": "fixture"}],
        },
    )
    write_json(
        before / "liquid_fixed.json",
        common
        | {
            "status": "completed_requires_review",
            "mode": "pilot",
            "datasets": [{"dataset_id": "fixture", "failed_requests": 0}],
        },
    )
    write_json(
        root / "preflight_cpu.json",
        {
            "status": "preflight_passed",
            "validation_identity": validation,
            "execution_site": "vsc_compute_node",
            "stage": "before_extraction",
            "contract_tests": {"status": "passed"},
            "datasets": [{"dataset_id": "fixture"}],
        },
    )
    write_json(
        before / "storage_contracts/fixed.json",
        {
            "status": "storage_contracts_passed_requires_quota_and_backup_review",
            "snapshot_restore": True,
            "validation_identity": validation,
            "storage_backend": "lustre",
        },
    )
    write_json(
        before / "storage_contracts/gpfs.json",
        {
            "status": "storage_contracts_passed_requires_quota_and_backup_review",
            "snapshot_restore": True,
            "validation_identity": validation,
            "storage_backend": "gpfs",
            "runtime": "liquid",
        },
    )
    write_json(
        root / "question_audit.json",
        {
            "status": "mechanical_checks_passed_requires_semantic_review",
            "datasets": [{"dataset_id": "fixture", "input_set_id": "v3", "file_sha256": "sha"}],
        },
    )
    return root, before, inputs


def test_evidence_preview_is_not_approval_and_signature_preserves_hashes(evidence):
    _, before, _ = evidence
    preview = acceptance.approve("liquid", reviewer=None)
    assert not preview["accepted"] and not (before / "accepted_liquid.json").exists()
    with pytest.raises(ValueError, match="quota"):
        acceptance.approve("liquid", reviewer="owner", apply=True)
    accepted = acceptance.approve(
        "liquid",
        reviewer="owner",
        quota_verified=True,
        backup_verified=True,
        throughput_reviewed=True,
        robustness_reviewed=True,
        apply=True,
    )
    assert (
        reviewed_acceptance(before / "accepted_liquid.json", stage="before_extraction") == accepted
    )
    with pytest.raises(ValueError, match="already exists"):
        acceptance.approve(
            "liquid",
            reviewer="owner",
            quota_verified=True,
            backup_verified=True,
            throughput_reviewed=True,
            robustness_reviewed=True,
            apply=True,
        )


@pytest.mark.parametrize(
    "fault", ["stale_inputs", "wrong_runtime", "incomplete", "failed_gpu", "stale_question_audit"]
)
def test_stale_or_incomplete_tests_cannot_be_signed(evidence, fault):
    root, before, inputs = evidence
    if fault == "stale_inputs":
        inputs[0]["input_set_id"] = "future-version"
    else:
        file = before / "liquid_fixed.json"
        if fault == "failed_gpu":
            file = before / "gpu_contracts/liquid_fixed.json"
        if fault == "stale_question_audit":
            file = root / "question_audit.json"
        body = json.loads(file.read_text())
        if fault == "wrong_runtime":
            body["settings_sha256"] = "foreign"
        elif fault == "incomplete":
            body["datasets"] = []
        elif fault == "stale_question_audit":
            body["datasets"][0]["input_set_id"] = "v2"
        else:
            body["status"] = "failed"
        write_json(file, body)
    with pytest.raises(ValueError):
        acceptance.approve(
            "liquid",
            reviewer="owner",
            quota_verified=True,
            backup_verified=True,
            throughput_reviewed=True,
            apply=True,
        )
    assert not (before / "accepted_liquid.json").exists()


def test_stale_cpu_code_identity_cannot_approve(evidence):
    root, _, _ = evidence
    file = root / "preflight_cpu.json"
    body = json.loads(file.read_text())
    body["validation_identity"]["source_sha256"] = "old-code"
    write_json(file, body)
    with pytest.raises(ValueError, match="matching"):
        acceptance.evidence_for("liquid")


def test_explicit_completed_campaign_disambiguates_successful_reruns(evidence, monkeypatch):
    from src.cluster.experiment_zero import steps
    from src.cluster.freeze import runtime_pins

    root, before, _ = evidence
    frozen = {"frozen": True, "source_sha256": "frozen-fixture"}
    monkeypatch.setattr("src.cluster.freeze.check_if_present", lambda: frozen)
    plan = steps("vsc")
    identity = digest({"source": frozen, "runtime_pins": runtime_pins(), "steps": plan})
    jobs = {s["name"]: str(i + 100) for i, s in enumerate(plan)}
    for name, file in (
        ("cpu", root / "preflight_cpu.json"),
        ("storage", before / "storage_contracts/fixed.json"),
        ("liquid_gpfs_storage", before / "storage_contracts/gpfs.json"),
        ("liquid_contracts", before / "gpu_contracts/liquid_fixed.json"),
        ("liquid_pilot", before / "liquid_fixed.json"),
    ):
        body = json.loads(file.read_text())
        write_json(file, body | {"job_id": jobs[name]})
    gpu = json.loads((before / "gpu_contracts/liquid_fixed.json").read_text())
    write_json(before / "gpu_contracts/liquid_older.json", gpu | {"job_id": "old"})
    with pytest.raises(ValueError, match="found 2"):
        acceptance.evidence_for("liquid")
    receipt = root / "0A_vsc/chosen.json"
    write_json(
        receipt,
        {
            "status": "vsc_checks_completed_requires_review",
            "stage_identity": identity,
            "steps": [s | {"status": "completed", "job_id": jobs[s["name"]]} for s in plan],
        },
    )
    result = acceptance.evidence_for("liquid", campaign=receipt)
    assert not any("older" in item["path"] for item in result["evidence"])
    assert result["evidence"][-1]["path"] == str(receipt.resolve())


def test_b200_requires_both_active_and_durable_storage_checks(evidence):
    _, before, _ = evidence
    (before / "storage_contracts/gpfs.json").unlink()
    with pytest.raises(ValueError, match="found 0"):
        acceptance.evidence_for("liquid")


@pytest.mark.parametrize("foreign_provider", ["liquid_lfm3", "intern"])
def test_provider_name_prefix_or_mislabeled_report_cannot_certify_another_runtime(
    evidence, foreign_provider
):
    _, before, _ = evidence
    sources = [
        before / "tokens_liquid_fixed.json",
        before / "gpu_contracts/liquid_fixed.json",
        before / "liquid_fixed.json",
    ]
    for source in sources:
        body = json.loads(source.read_text())
        # Keep every other field identical, including successful counts and settings.
        # Provider identity must disambiguate even a misleading filename.
        write_json(source.with_name(source.stem + "_foreign.json"), body | {"provider": foreign_provider})
    result = acceptance.evidence_for("liquid")
    assert result["provider"] == "liquid"
    assert not any("foreign" in item["path"] for item in result["evidence"])
    for source in sources:
        source.unlink()
    with pytest.raises(ValueError, match="found 0"):
        acceptance.evidence_for("liquid")
