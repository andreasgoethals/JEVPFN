"""Offline Exp0 evidence must never be confused with live or VSC acceptance."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from src.cluster import preflight
from src.evaluation import acceptance
from src.utils import paths


@pytest.fixture
def local_preflight(tmp_path, monkeypatch):
    cfg = deepcopy(preflight.load_yaml("experiment_0/debug"))
    cfg["dataset_selection"]["exclude"] = []
    monkeypatch.setenv("JEVPFN_PHASE", "experiment_0")
    for variable in (
        "VSC_DATA",
        "SLURM_JOB_ID",
        "SLURM_CLUSTER_NAME",
        "JEVPFN_STAGING_ROOT",
        "VSC_STAGING_ROOT",
        "VSC_PROJECT_LUSTRE1",
        "VSC_PROJECT_GPFS1",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(preflight, "load_yaml", lambda _name: cfg)
    monkeypatch.setattr(preflight, "load_catalog", lambda: {"datasets": []})
    # This fixture intentionally has no benchmark configuration/source tree.
    # The real identity's drift/activation rules are tested separately.
    monkeypatch.setattr(acceptance, "evaluation_identity", lambda: "fixture-evaluation")
    return cfg


def test_local_preflight_does_not_create_inputs_or_claim_live_acceptance(local_preflight, tmp_path):
    report = preflight.run_preflight(local=True)
    assert report["status"] == "preflight_passed"
    assert report["execution_site"] == "local" and report["job_id"] is None
    assert not report["full_experiment_0_passed"]
    assert not report["performed_checks"]["live_decision_inference"]
    assert not report["performed_checks"]["predictive_models_fitted"]
    assert not report["performed_checks"]["cuda_tensor_and_tabpfn_import"]
    assert {check["id"] for check in report["pending_acceptance"]} >= {
        "provider_task_contracts",
        "allocation_and_native_storage",
        "full_fold_resource_profile",
    }
    assert not (tmp_path / "data").exists()
    assert (tmp_path / "output_JEVPFN").is_dir()


def test_local_mode_cannot_bypass_compute_guard(local_preflight, monkeypatch, tmp_path):
    monkeypatch.setenv("VSC_DATA", str(tmp_path / "data"))
    with pytest.raises(RuntimeError, match="--local must run off VSC"):
        preflight.run_preflight(local=True)


def test_exp0_before_and_after_features_have_distinct_requirements(local_preflight):
    before = preflight.run_preflight(local=True, stage="before_extraction")
    assert before["status"] == "preflight_passed"
    assert before["feature_validation"] == "not_required_before_extraction"
    assert "provider_task_contracts" in {c["id"] for c in before["stage_pending_acceptance"]}
    after = preflight.run_preflight(local=True, stage="after_extraction")
    assert after["status"] == "failed"
    assert after["evaluation_identity"] == "fixture-evaluation"
    assert any("0B requires" in failure for failure in after["failures"])


@pytest.mark.parametrize(
    ("cluster", "allocation", "message"),
    [("mindwell", "lustre1", "native GPFS"), ("wice", "gpfs1", "native Lustre")],
)
def test_cross_mounted_project_storage_rejected(
    local_preflight, monkeypatch, tmp_path, cluster, allocation, message
):
    project = tmp_path / allocation / "allocation"
    project.mkdir(parents=True)
    monkeypatch.setenv("VSC_DATA", str(tmp_path / "personal"))
    monkeypatch.setenv("JEVPFN_STAGING_ROOT", str(project))
    assert any(message in failure for failure in preflight.storage_policy_failures(cluster))


def test_missing_project_mount_is_not_created(local_preflight, monkeypatch, tmp_path):
    project = tmp_path / "absent-project"
    monkeypatch.setenv("VSC_DATA", str(tmp_path / "personal"))
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-job")
    monkeypatch.setenv("JEVPFN_STAGING_ROOT", str(project))
    report = preflight.run_preflight()
    assert report["status"] == "failed"
    assert any("must already exist" in failure for failure in report["failures"])
    assert not project.exists()


def test_project_cannot_be_under_data(local_preflight, monkeypatch, tmp_path):
    personal = tmp_path / "personal"
    project = personal / "misplaced-project"
    project.mkdir(parents=True)
    monkeypatch.setenv("VSC_DATA", str(personal))
    monkeypatch.setenv("JEVPFN_STAGING_ROOT", str(project))
    assert any("outside personal DATA" in fail for fail in preflight.storage_policy_failures())


def _dataset(task="binary_classification"):
    target = [0, 1] * 5 if task != "regression" else list(range(10))
    return SimpleNamespace(
        frame=pd.DataFrame({"text": ["sample"] * 10, "target": target}),
        metadata=SimpleNamespace(
            dataset_id="example",
            task_type=task,
            target="target",
            classes=(0, 1) if task != "regression" else (),
            text_columns=("text",),
            fingerprint="schema",
        ),
        source={"data_identity": "data"},
        issues=[],
    )


@pytest.mark.parametrize(
    "task", ["binary_classification", "multiclass_classification", "regression"]
)
def test_offline_data_checks_construct_real_splits_without_fitting(monkeypatch, task):
    dataset = _dataset(task)
    monkeypatch.setattr(preflight, "load_catalog", lambda: {"datasets": [{"id": "example"}]})
    loader = Mock(return_value=dataset)
    monkeypatch.setattr(preflight, "load_dataset", loader)
    result = preflight.dataset_checks({})
    assert result[0]["status"] == "passed" and result[0]["outer_folds"] == 5
    assert len(result[0]["split_id"]) == 64
    loader.assert_called_once_with({"id": "example"}, allow_download=False)
    assert "target" not in result[0] and "class_frequencies" not in result[0]


def test_broken_dataset_does_not_hide_later_dataset_checks(monkeypatch):
    monkeypatch.setattr(
        preflight, "load_catalog", lambda: {"datasets": [{"id": "bad"}, {"id": "example"}]}
    )
    loader = Mock(side_effect=[ValueError("corrupt hash"), _dataset()])
    monkeypatch.setattr(preflight, "load_dataset", loader)
    results = preflight.dataset_checks({})
    assert [result["status"] for result in results] == ["failed", "passed"]
    assert results[0]["error_type"] == "ValueError"


def test_corrupt_artifact_is_recorded_instead_of_aborting_report(local_preflight, monkeypatch):
    local_preflight["feature_artifacts"] = [{"dataset_id": "example", "artifact_id": "123"}]
    monkeypatch.setattr(preflight, "load_catalog", lambda: {"datasets": [{"id": "example"}]})
    monkeypatch.setattr(preflight, "load_block", Mock(side_effect=ValueError("checksum mismatch")))
    report = preflight.run_preflight(local=True)
    assert report["status"] == "failed"
    assert report["feature_artifacts"][0]["status"] == "failed"
    assert report["feature_artifacts"][0]["error"] == "checksum mismatch"


def test_foreign_feature_rows_are_rejected(local_preflight, monkeypatch):
    local_preflight["feature_artifacts"] = [{"dataset_id": "example", "artifact_id": "123"}]
    monkeypatch.setattr(preflight, "load_catalog", lambda: {"datasets": [{"id": "example"}]})
    block = SimpleNamespace(
        values=pd.DataFrame({"row_id": ["csv:999"]}),
        manifest={
            "identity": {
                "dataset_id": "example",
                "data_identity": "data",
                "task_schema_sha256": "schema",
            }
        },
    )
    monkeypatch.setattr(preflight, "load_block", Mock(return_value=block))
    monkeypatch.setattr(preflight, "load_dataset", Mock(return_value=_dataset()))
    report = preflight.run_preflight(local=True)
    assert report["status"] == "failed"
    assert "Feature rows/source identity" in report["feature_artifacts"][0]["error"]


def test_contract_suite_uses_current_interpreter_and_preserves_failed_log(
    local_preflight, monkeypatch
):
    run = Mock(return_value=SimpleNamespace(returncode=1, stdout="1 failed\n", stderr=""))
    monkeypatch.setattr(preflight.subprocess, "run", run)
    report = preflight.run_preflight(local=True, tests=True)
    assert run.call_args.args[0] == [preflight.sys.executable, "-m", "pytest", "-q"]
    assert report["status"] == "failed"
    assert Path(report["contract_tests"]["log"]).read_text() == "1 failed\n"
    assert not report["full_experiment_0_passed"]
