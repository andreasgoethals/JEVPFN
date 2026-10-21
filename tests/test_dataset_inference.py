import numpy as np
import pandas as pd
import pytest

from src.evaluation.inference import GROUP, across_datasets, confirmatory_tests, dataset_differences


def example():
    base = dict(
        task_type="regression",
        model_id="catboost",
        checkpoint_id="pin",
        feature_set="non_text_plus_joint",
        comparison_id="jev",
        reference="non_text",
        metric="r2",
    )
    rows = [
        {
            **base,
            "dataset_id": f"data{i}",
            "split_id": f"split{i}",
            "outer_fold": f,
            "difference": i / 10 + f / 100,
        }
        for i in range(4)
        for f in range(5)
    ]
    return base, pd.DataFrame(rows)


def test_five_folds_become_one_dataset_unit_and_undefined_stays_visible():
    _, rows = example()
    datasets = dataset_differences(rows)
    assert len(datasets) == 4 and datasets.complete.all()
    assert across_datasets(datasets).iloc[0].complete_datasets == 4
    rows.loc[0, "difference"] = np.nan
    result = dataset_differences(rows)
    assert len(result) == 4 and result.complete.sum() == 3
    assert across_datasets(result).iloc[0].complete_datasets == 3


def test_inference_requires_prespecified_complete_dataset_family():
    base, rows = example()
    datasets = dataset_differences(rows)
    contrast = {"match": base, "datasets": [f"data{i}" for i in range(4)], "direction": 1}
    plan = {"approved_before_results": True, "reviewed_by": "fixture", "contrasts": [contrast]}
    out = confirmatory_tests(datasets, plan)
    assert out.iloc[0].datasets == 4
    assert out.iloc[0].p_holm == out.iloc[0].p_raw
    with pytest.raises(ValueError, match="before results"):
        confirmatory_tests(datasets, plan | {"approved_before_results": False})
    with pytest.raises(ValueError, match="Every planned dataset"):
        confirmatory_tests(datasets.iloc[1:], plan)
    with pytest.raises(ValueError, match="Duplicate planned"):
        confirmatory_tests(datasets, plan | {"contrasts": [contrast, contrast]})
    assert set(base) == set(GROUP)


def test_native_references_are_not_misleading_pairs_and_e5_is_a_comparison(isolated_output):
    from types import SimpleNamespace

    from src.data.loaders import LoadedDataset
    from src.data.metadata import extract_metadata
    from src.evaluation.protocol import evaluate_cv, shared_splits
    from src.evaluation.results import collect

    frame = pd.DataFrame({"text": ["sample"] * 20, "x": np.arange(20.0), "y": np.arange(20.0)})
    official = {"target": "y", "task_type": "reg"}
    meta = extract_metadata(
        frame, official, dataset_id="fixture", dataset_name="Fixture", text_columns=["text"]
    )
    data = LoadedDataset(frame, meta, official, {"data_identity": "fixed"}, [])
    matrix = frame[["x"]].set_axis([f"csv:{i}" for i in range(20)])
    for model, phase, features, error, provider in (
        ("catboost", "experiment_1", "non_text", 3, None),
        ("catboost", "experiment_1", "non_text_plus_joint", 1, "jev"),
        ("catboost", "experiment_1", "non_text_plus_joint", 1.5, "liquid"),
        ("catboost", "experiment_1", "non_text_plus_frozen_e5", 2, None),
        ("catboost", "experiment_1", "non_text_plus_frozen_e5_and_joint", 0.5, "jev"),
        ("catboost", "experiment_1", "non_text_plus_frozen_e5_and_joint", 0.25, "liquid"),
        ("tabstar", "experiment_2", "original_features", 1, None),
        (
            "catboost",
            "experiment_3",
            "non_text_plus_frozen_e5__embedding_dimension__pca_16",
            2.5,
            None,
        ),
    ):
        evaluate_cv(
            data,
            matrix,
            lambda error=error: SimpleNamespace(
                fit=lambda x, y: None, predict=lambda x: x.x.to_numpy() + error
            ),
            specification=shared_splits(data),
            model_id=model,
            phase=phase,
            checkpoint_id="pin",
            feature_set=features,
            allow_model_runs=True,
            estimator_settings={},
            feature_artifact_ids=[],
            representation_recipe={
                "provider": provider,
                "view": "main",
                "embedding": {"model": "synthetic"} if "e5" in features else None,
            },
        )
    report, _, summary, paired = collect()
    assert "tabstar" in set(summary.model_id) and "tabstar" not in set(paired.model_id)
    contrast = paired.loc[
        (paired.feature_set == "non_text_plus_joint")
        & (paired.provider == "jev")
        & (paired.reference == "frozen_e5")
        & (paired.metric == "rmse")
    ]
    assert len(contrast) == 5
    np.testing.assert_allclose(contrast.difference, -1)
    for provider, expected in (("jev", -0.5), ("liquid", -1.25)):
        contrast = paired.loc[
            (paired.feature_set == "non_text_plus_frozen_e5_and_joint")
            & (paired.provider == provider)
            & (paired.reference == "same_decision_features_without_e5")
            & (paired.metric == "rmse")
        ]
        assert len(contrast) == 5
        np.testing.assert_allclose(contrast.difference, expected)
    assert (
        len(
            paired.loc[
                paired.feature_set.eq("non_text_plus_frozen_e5_and_joint")
                & paired.reference.eq("frozen_e5")
                & paired.metric.eq("rmse")
            ]
        )
        == 10
    )
    assert "descriptive" in report["fold_sd"]
    pca_contrast = paired.loc[
        paired.feature_set.eq("non_text_plus_frozen_e5__embedding_dimension__pca_16")
        & paired.reference.eq("frozen_e5")
        & paired.metric.eq("rmse")
    ]
    assert len(pca_contrast) == 5
    np.testing.assert_allclose(pca_contrast.difference, 0.5)


def test_target_unit_error_magnitudes_are_not_pooled_across_datasets():
    _, rows = example()
    rows["metric"] = "rmse"
    result = across_datasets(dataset_differences(rows))
    assert result.mean_difference.isna().all() and result.median_difference.isna().all()
    assert result.positive_deltas.iloc[0] == 4
