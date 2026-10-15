"""Evaluation provenance uses exact inputs, without fitting a prediction model."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data.loaders import LoadedDataset
from src.data.metadata import extract_metadata
from src.evaluation import protocol


def test_feature_identity_includes_values_row_order_and_dtype():
    features = pd.DataFrame({"number": [1.0, 2.0]}, index=["csv:0", "csv:1"])
    original = protocol.feature_input_identity(features)
    assert original == protocol.feature_input_identity(features.copy())
    assert original != protocol.feature_input_identity(features * 2)
    assert original != protocol.feature_input_identity(features.iloc[::-1])
    assert original != protocol.feature_input_identity(features.astype("float32"))
    categories = pd.DataFrame(
        {"category": pd.Categorical(["a", "b"], categories=["a", "b"], ordered=True)},
        index=features.index,
    )
    changed = categories.copy()
    changed["category"] = changed.category.cat.reorder_categories(["b", "a"])
    assert protocol.feature_input_identity(categories) != protocol.feature_input_identity(changed)


@pytest.fixture
def cv_fixture(monkeypatch, tmp_path):
    frame = pd.DataFrame(
        {"text": ["example"] * 10, "number": np.arange(10), "target": np.arange(10.0)}
    )
    official = {"target": "target", "task_type": "reg"}
    metadata = extract_metadata(
        frame, official, dataset_id="fixture", dataset_name="Fixture", text_columns=["text"]
    )
    dataset = LoadedDataset(frame, metadata, official, {"data_identity": "fixed"}, [])
    features = frame[["number"]].copy()
    features.index = [f"csv:{i}" for i in range(len(frame))]
    saved, calls = [], []

    def fake_estimator():
        return SimpleNamespace(
            fit=lambda x, y: calls.append(len(x)),
            predict=lambda x: x.number.to_numpy(),
        )

    def fake_save(**kwargs):
        saved.append(kwargs)
        return tmp_path / kwargs["run_id"]

    monkeypatch.setattr(protocol, "save_evaluation", fake_save)
    # This fixture tests identities without serializing records; resume is tested separately.
    monkeypatch.setattr(protocol, "completed_fold", lambda *a: None)
    monkeypatch.setattr(protocol, "complete_fold", lambda *a: None)
    monkeypatch.setattr(
        protocol.paths, "results_dir", lambda *parts, **kwargs: tmp_path.joinpath(*parts)
    )
    arguments = dict(
        specification=protocol.shared_splits(dataset),
        model_id="fake",
        checkpoint_id="fixture-v1",
        feature_set="fixture-features",
        allow_model_runs=True,
        estimator_settings={"fixture": "fixed-defaults"},
        feature_artifact_ids=[],
    )
    return dataset, features, fake_estimator, arguments, saved, calls


def test_changed_features_settings_or_artifacts_cannot_reuse_evaluation_identity(cv_fixture):
    dataset, features, factory, arguments, saved, _ = cv_fixture
    cases = [
        (features, {}),
        (features * 2, {}),
        (features, {"estimator_settings": {"fixture": "changed"}}),
        (features, {"feature_artifact_ids": ["different-extractor-or-question-version"]}),
    ]
    run_ids = []
    for matrix, overrides in cases:
        records = protocol.evaluate_cv(dataset, matrix, factory, **(arguments | overrides))
        run_ids.append(tuple(path.name for path in records))
    assert len(set(run_ids)) == len(cases)
    assert not saved[0]["predictions"].equals(saved[5]["predictions"])
    assert saved[0]["metadata"]["feature_input_identity"] == protocol.feature_input_identity(
        features
    )
    assert saved[-1]["metadata"]["feature_artifact_ids"] == [
        "different-extractor-or-question-version"
    ]


@pytest.mark.parametrize("missing", ["estimator_settings", "feature_artifact_ids"])
def test_missing_provenance_fails_before_any_fit(cv_fixture, missing):
    dataset, features, factory, arguments, _, calls = cv_fixture
    arguments.pop(missing)
    with pytest.raises(ValueError, match="settings and explicit feature artifact IDs"):
        protocol.evaluate_cv(dataset, features, factory, **arguments)
    assert not calls


def test_artifact_ids_can_follow_the_assembled_feature_matrix(cv_fixture):
    dataset, features, factory, arguments, saved, _ = cv_fixture
    features.attrs["feature_artifact_ids"] = ["saved-artifact"]
    arguments.pop("feature_artifact_ids")
    protocol.evaluate_cv(dataset, features, factory, **arguments)
    assert saved[0]["metadata"]["feature_artifact_ids"] == ["saved-artifact"]


@pytest.mark.parametrize("binary", [False, True])
def test_next_fold_does_not_load_before_previous_estimator_is_released(cv_fixture, binary):
    dataset, features, _, arguments, _, _ = cv_fixture
    if binary:
        dataset.frame["target"] = np.arange(len(dataset.frame)) % 2
        dataset.official.update(task_type="cls", num_classes=2)
        dataset.metadata = extract_metadata(
            dataset.frame,
            dataset.official,
            dataset_id="fixture",
            dataset_name="Fixture",
            text_columns=["text"],
        )
        arguments["specification"] = protocol.shared_splits(dataset)

    class Checkpoint:
        active = 0
        maximum = 0
        classes_ = [0, 1]

        def __init__(self):
            type(self).active += 1
            type(self).maximum = max(type(self).active, type(self).maximum)

        def __del__(self):
            type(self).active -= 1

        def fit(self, x, y):
            pass

        def predict(self, x):
            return x.number.to_numpy()

        def predict_proba(self, x):
            return np.tile([0.4, 0.6], (len(x), 1))

    protocol.evaluate_cv(dataset, features, Checkpoint, **arguments)
    assert Checkpoint.maximum == 1
    assert Checkpoint.active == 0
