import json

import pytest

from src.cluster import accept_features as acceptance
from src.utils import paths
from src.utils.acceptance import reviewed_acceptance
from src.utils.serialization import digest, write_json


@pytest.fixture
def evidence(isolated_output, monkeypatch):
    root = paths.manifests_dir("experiment_0")
    before = root / "before_extraction"
    inputs = [{"dataset_id": "fixture", "input_set_id": "v3", "input_file_sha256": "sha"}]
    monkeypatch.setattr(acceptance, "selected_inputs", lambda: inputs)
    write_json(
        before / "tokens_liquid_fixed.json",
        {
            "status": "counted_all_inputs",
            "overflow_unique_inputs": 0,
            "audit_id": "token",
            "datasets": inputs,
        },
    )
    identity = {"provider": "liquid", "version": "pin"}
    common = {"token_audit_id": "token", "settings_sha256": digest(identity), "settings": identity}
    write_json(
        before / "gpu_contracts/liquid_fixed.json",
        common
        | {
            "status": "gpu_contracts_passed_requires_review",
            "runtime_token_count_parity": True,
            "semantic_diagnostics": {"passed": True},
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
