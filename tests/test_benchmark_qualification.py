"""Exercise the real 0B gate without allocating models or using synthetic evidence as approval."""

from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from src.cluster.predictor_contracts import CONTRACT_VERSION, required_fixture_cases
from src.evaluation import qualification as q
from src.evaluation.features import assemble_feature_set, missing_column
from src.evaluation.models import MAIN_MODELS, TEXT_MODELS
from src.utils.serialization import file_sha256, write_json


@pytest.fixture
def reports(monkeypatch):
    model = MAIN_MODELS[0]
    arm = {
        "model": model,
        "phase": "experiment_1",
        "dataset_id": "data",
        "provider": "provider",
        "feature_set": "non_text_plus_both",
        "predictor_columns": 4,
    }
    monkeypatch.setattr(q, "evaluation_identity", lambda: "current")
    monkeypatch.setattr(q, "profile_plan", lambda: [arm])
    monkeypatch.setattr(q, "load_catalog", lambda: {"datasets": [{"id": "data"}]})
    monkeypatch.setattr(
        q,
        "load_yaml",
        lambda name: {"candidate_extractors": ["provider"], "encoders": ["encoder"]},
    )
    preflight = {
        "phase": "experiment_0_preflight",
        "stage": "after_extraction",
        "execution_site": "vsc_compute_node",
        "status": "preflight_passed",
        "contract_tests": {"status": "passed"},
        "runtime_parity": {"matches_control": True},
        "datasets": [
            {
                "dataset_id": "data",
                "status": "passed",
                "task_type": "regression",
                "data_identity": "raw",
                "split_id": "split",
            }
        ],
        "decision_artifacts": [
            {
                "dataset_id": "data",
                "provider": "provider",
                "status": "passed",
                "artifact_ids": {"joint": "joint", "per_column": "columns"},
                "artifact_checksums": {"joint": "j", "columns": "c"},
            }
        ],
        "embedding_artifacts": [
            {
                "dataset_id": "data",
                "encoder_id": "encoder",
                "status": "passed",
                "artifact_id": "vectors",
                "prepared_pca_keys": 5,
            }
        ],
    }
    runtime = {"container_id": "seal"}
    predictors = [
        {
            "model": m,
            "status": "predictor_contracts_passed_requires_full_fold_profile",
            "contract_version": CONTRACT_VERSION,
            "runtime_identity": runtime,
            "fixtures": [
                {
                    "task_type": task,
                    "classes": n,
                    "feature_case": case,
                    "resume_without_refitting": True,
                }
                for task, n, case in sorted(required_fixture_cases(m))
            ],
        }
        for m in MAIN_MODELS + TEXT_MODELS
    ]
    profile = {
        "arm": arm,
        "status": "full_fold_profile_passed",
        "runtime_identity": runtime,
        "data_identity": "raw",
        "split_id": "split",
        "feature_artifact_ids": ["joint", "columns"],
        "decision_artifact_checksums": {"joint": "j", "columns": "c"},
        "predictor_columns": 4,
        "roles": [{"role": "outer_test", "rows": 2, "record_id": "test", "prediction_sha256": "sha"}],
    }
    return arm, [preflight, *predictors, profile]


def write_approval(tmp_path, bodies):
    evidence = []
    for i, body in enumerate(bodies):
        path = tmp_path / f"report_{i}.json"
        write_json(path, {**body, "evaluation_identity": "current", "job_id": str(i + 1)})
        evidence.append({"path": path.name, "sha256": file_sha256(path)})
    accepted = tmp_path / "review.json"
    write_json(
        accepted,
        {
            "accepted": True,
            "stage": "after_extraction",
            "reviewed_by": "reviewer",
            "evaluation_identity": "current",
            "evidence": evidence,
        },
    )
    return accepted


def test_full_fold_profile_reads_saved_binary_outer_and_oof_predictions(
    isolated_output, monkeypatch
):
    import json

    from sklearn.dummy import DummyClassifier

    from src.cluster.predictor_contracts import feature_cases, fixtures
    from src.evaluation.protocol import evaluate_cv, shared_splits
    from src.utils import paths

    dataset = next(d for d in fixtures() if d.metadata.task_type == "binary_classification")
    _, features, _ = next(feature_cases(dataset, "linear"))
    features.attrs["decision_artifact_checksums"] = {}
    arm = {
        "phase": "experiment_1", "model": "linear", "feature_set": "non_text",
        "dataset_id": dataset.metadata.dataset_id,
        "predictor_columns": features.shape[1],
    }
    split = shared_splits(dataset, seed=q.load_yaml("experiment_1/main")["random_seed"])
    records = evaluate_cv(
        dataset, features, DummyClassifier, specification=split,
        model_id="linear", checkpoint_id="test", feature_set="non_text",
        phase="experiment_0", allow_model_runs=True, fold_numbers=[0],
        estimator_settings={"fixture": "dummy"},
    )
    with pytest.raises(ValueError, match="lacks required prediction roles"):
        q.save_profile(arm, dataset, features, {"container_id": "test"}, records[:1])
    with pytest.raises(ValueError, match="predictor width"):
        q.save_profile(arm | {"predictor_columns": 1}, dataset, features, {}, records)
    q.save_profile(arm, dataset, features, {"container_id": "test"}, records)
    path, = (paths.manifests_dir("experiment_0") / "after_extraction").glob("profile_*.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert {r["role"]: r["rows"] for r in report["roles"]} == {
        "outer_test": len(split["folds"][0]["test"]),
        "inner_oof_binary_threshold": len(split["folds"][0]["train"]),
    }
    with pytest.raises(ValueError, match="outside scratch"):
        q.verify_published_profile(report)
    durable = paths.staging_override()
    monkeypatch.setattr(paths, "durable_project_override", lambda: durable)
    q.verify_published_profile(report)
    report["roles"][0]["record_id"] = "missing"
    with pytest.raises(ValueError, match="Missing evaluation record"):
        q.verify_published_profile(report)


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("missing_mode", "exact artifact IDs"),
        ("missing_pca", "exact artifact IDs"),
        ("partial_contract", "fixtures or sealed runtime"),
        ("wrong_profile_split", "full-fold profile"),
        ("wrong_profile_input", "full-fold profile"),
        ("wrong_profile_runtime", "full-fold profile"),
    ],
)
def test_checksummed_but_incomplete_or_mismatched_0b_rejected(tmp_path, reports, fault, message):
    arm, bodies = deepcopy(reports)
    if fault == "missing_mode":
        bodies[0]["decision_artifacts"][0]["artifact_ids"].pop("joint")
    elif fault == "missing_pca":
        bodies[0]["embedding_artifacts"][0]["prepared_pca_keys"] = 0
    elif fault == "partial_contract":
        bodies[1]["fixtures"].pop()
    elif fault == "wrong_profile_split":
        bodies[-1]["split_id"] = "other"
    elif fault == "wrong_profile_input":
        bodies[-1]["decision_artifact_checksums"]["joint"] = "different"
    else:
        bodies[-1]["runtime_identity"] = {"container_id": "different"}
    with pytest.raises(ValueError, match=message):
        q.benchmark_evidence(write_approval(tmp_path, bodies), arm)


def test_qualified_content_and_runtime_checked_before_benchmark_fit(tmp_path, reports):
    arm, bodies = reports
    evidence = q.benchmark_evidence(write_approval(tmp_path, bodies), arm)
    dataset = SimpleNamespace(source={"data_identity": "raw"})
    features = pd.DataFrame()
    features.attrs.update(
        feature_artifact_ids=["columns", "joint"],
        decision_artifact_checksums={"columns": "c", "joint": "j"},
    )
    runtime = {"container_id": "seal"}
    split = {"split_id": "split"}
    q.verify_inputs(evidence, dataset, features, arm, runtime, specification=split)
    with pytest.raises(ValueError, match="qualified 0B partition"):
        q.verify_inputs(
            evidence, dataset, features, arm, runtime, specification={"split_id": "other"}
        )
    features.attrs["decision_artifact_checksums"]["joint"] = "changed"
    with pytest.raises(ValueError, match="Feature artifacts"):
        q.verify_inputs(evidence, dataset, features, arm, runtime, specification=split)


def test_missing_text_signal_is_available_to_the_structured_baseline():
    frame = pd.DataFrame({"text": ["present", None], "number": [1, 2], "target": [2, 4]})
    dataset = SimpleNamespace(
        features=frame.drop(columns="target"),
        frame=frame,
        metadata=SimpleNamespace(
            text_columns=("text",), non_text_columns=("number",), target="target"
        ),
    )
    matrix = assemble_feature_set(dataset, [], "non_text")
    assert list(matrix.columns) == ["number", missing_column(0)]
    assert matrix[missing_column(0)].tolist() == [0, 1]


def test_raw_copies_recodings_and_duplicate_rows_are_reported_separately():
    frame = pd.DataFrame({
        "x": [1, 1, 2, 3], "copy": [0, 0, 1, 1], "inverse": [1, 1, 0, 0],
        "alias": pd.Categorical(["a", "a", "b", "b"]), "noisy": [0, 0, 1, 0],
        "target": pd.Categorical([0, 0, 1, 1]),
    })
    dataset = SimpleNamespace(
        frame=frame,
        features=frame.drop(columns="target"),
        metadata=SimpleNamespace(target="target", task_type="binary_classification"),
    )
    audit = q.source_leakage_diagnostics(dataset, {"folds": [{"train": [0, 2], "test": [1, 3]}]})
    assert audit["exact_target_copies"] == ["copy"]
    assert audit["exact_class_target_recodings"] == ["inverse", "alias"]
    assert audit["exact_feature_row_overlap_by_fold"] == [1]
    assert audit["test_rows_by_fold"] == [2]


def test_benchmark_publication_copies_only_selected_evaluation_store(tmp_path):
    from src.cluster.scratch_stage import _publish
    from src.evaluation.store import connect

    work, durable = tmp_path / "scratch" / "JEVPFN", tmp_path / "project" / "JEVPFN"
    results = work / "output_JEVPFN" / "4. experiment_1" / "results"
    evaluation = results / "evaluation"
    evaluation.mkdir(parents=True)
    for dataset in ("DATA1", "DATA2"):
        with connect(evaluation / f"{dataset}.sqlite3"):
            pass
    (results / "splits").mkdir()
    (results / "splits" / "split.json").write_text("{}", encoding="utf-8")

    _publish(durable, work, dataset_id="DATA1")
    target = durable / results.relative_to(work)
    assert (target / "evaluation" / "DATA1.sqlite3").is_file()
    assert not (target / "evaluation" / "DATA2.sqlite3").exists()
    assert (target / "splits" / "split.json").is_file()
