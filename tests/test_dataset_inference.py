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
        ("catboost", "experiment_2", "non_text_plus_frozen_e5", 2, None),
        ("catboost", "experiment_5", "non_text_plus_frozen_e5_and_joint", 0.5, "jev"),
        ("catboost", "experiment_5", "non_text_plus_frozen_e5_and_joint", 0.25, "liquid"),
        ("catboost", "experiment_2", "non_text_plus_frozen_e5", 4, None),
        ("catboost", "experiment_5", "non_text_plus_frozen_e5_and_joint", 3.5, "jev"),
        ("tabstar", "experiment_3", "original_features", 1, None),
        ("tabstar", "experiment_5", "original_features_plus_joint", 0.5, "jev"),
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
                **(
                    {"inference_status": "exploratory_post_selection", "selection": {"id": "test"}}
                    if phase == "experiment_5"
                    else {}
                ),
                "view": "main",
                "embedding": {"model": "encoder2" if error >= 3.5 else "encoder1"}
                if "e5" in features
                else None,
            },
        )
    report, _, summary, paired = collect()
    assert "tabstar" in set(summary.model_id)
    native = paired.loc[paired.model_id.eq("tabstar") & paired.metric.eq("rmse")]
    assert len(native) == 5
    assert set(native.reference) == {"original_text_without_decision_features"}
    np.testing.assert_allclose(native.difference, -0.5)
    for provider, expected in (("jev", -0.5), ("liquid", -1.25)):
        contrast = paired.loc[
            (paired.feature_set == "non_text_plus_frozen_e5_and_joint")
            & (paired.provider == provider)
            & (paired.reference == "same_decision_features_without_e5")
            & (paired.metric == "rmse")
            & (paired.value < 1)
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
        == 15
    )
    second_encoder = paired.loc[
        paired.feature_set.eq("non_text_plus_frozen_e5_and_joint")
        & paired.reference.eq("frozen_e5")
        & paired.metric.eq("rmse")
        & paired.value.gt(3)
    ]
    assert len(second_encoder) == 5
    np.testing.assert_allclose(second_encoder.reference_value, 4)
    np.testing.assert_allclose(second_encoder.difference, -0.5)
    assert "descriptive" in report["fold_sd"]


def test_target_unit_error_magnitudes_are_not_pooled_across_datasets():
    _, rows = example()
    rows["metric"] = "rmse"
    result = across_datasets(dataset_differences(rows))
    assert result.mean_difference.isna().all() and result.median_difference.isna().all()
    assert result.positive_deltas.iloc[0] == 4


def test_test_selected_combinations_never_enter_confirmatory_analysis():
    base, rows = example()
    rows["inference_status"] = "exploratory_post_selection"
    datasets = dataset_differences(rows)
    assert across_datasets(datasets).iloc[0].inference_status == "exploratory_post_selection"
    plan = {
        "approved_before_results": True,
        "reviewed_by": "fixture",
        "contrasts": [{"match": base, "datasets": [f"data{i}" for i in range(4)], "direction": 1}],
    }
    with pytest.raises(ValueError, match="exploratory"):
        confirmatory_tests(datasets, plan)
