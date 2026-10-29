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


def test_primary_results_pair_exact_predictors_and_keep_native_references(isolated_output):
    from types import SimpleNamespace

    from src.data.loaders import LoadedDataset
    from src.data.metadata import extract_metadata
    from src.evaluation.protocol import evaluate_cv, shared_splits
    from src.evaluation.results import collect
    frame = pd.DataFrame({"text": ["sample"] * 20, "x": np.arange(20.0), "y": np.arange(20.0)})
    official = {"target": "y", "task_type": "reg"}
    meta = extract_metadata(frame, official, dataset_id="fixture", dataset_name="Fixture", text_columns=["text"])
    data = LoadedDataset(frame, meta, official, {"data_identity": "fixed"}, [])
    matrix = frame[["x"]].set_axis([f"csv:{i}" for i in range(20)])
    for model, phase, features, error, provider in (
        ("catboost", "experiment_1", "non_text", 3, None),
        ("catboost", "experiment_1", "non_text_plus_joint", 1, "jev"),
        ("catboost", "experiment_1", "non_text_plus_joint", 1.5, "liquid"),
        ("catboost", "experiment_2", "non_text_plus_frozen_e5", 2, None),
        ("catboost", "experiment_2", "non_text_plus_frozen_e5", 4, None),
        ("tabstar", "experiment_3", "original_features", 1, None),
    ):
        evaluate_cv(data, matrix,
            lambda error=error: SimpleNamespace(fit=lambda x,y: None, predict=lambda x: x.x.to_numpy()+error),
            specification=shared_splits(data), model_id=model, phase=phase, checkpoint_id="pin",
            feature_set=features, allow_model_runs=True, estimator_settings={}, feature_artifact_ids=[],
            representation_recipe={"provider":provider, "view":"main",
                "embedding": {"model":f"encoder{error}"} if "e5" in features else None})
    report, _, summary, paired = collect()
    assert "tabstar" in set(summary.model_id)
    assert not paired.model_id.eq("tabstar").any()
    assert set(paired.reference) == {"non_text"}
    rmse = paired[paired.metric.eq("rmse")]
    assert len(rmse) == 20
    np.testing.assert_allclose(rmse.reference_value, 3)
    np.testing.assert_allclose(rmse.difference, rmse.value - 3)
    assert len(rmse[rmse.phase.eq("experiment_2")].embedding_recipe_id.unique()) == 2
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


@pytest.mark.parametrize("n, floor", [(3, 0.25), (6, 0.03125), (9, 0.00390625)])
def test_exact_resolution_reports_all_task_sizes_and_drops_zeros(n, floor):
    from src.evaluation.inference import exact_resolution

    assert exact_resolution([1] * n + [0, 0]) == {
        "nonzero_datasets": n,
        "ideal_exact_two_sided_p_floor_nonzero": floor,
    }
    assert exact_resolution([0, 0])["ideal_exact_two_sided_p_floor_nonzero"] == 1


def test_holm_diagnostics_use_step_down_ranks_not_a_blanket_family_threshold():
    base, _ = example()
    small = {**base, "task_type": "multiclass_classification"}
    larger = {**base, "task_type": "regression"}
    datasets = pd.DataFrame(
        [
            {**keys, "dataset_id": f"d{i}", "mean_difference": i + 1.0, "complete": True}
            for keys, n in ((small, 6), (larger, 9))
            for i in range(n)
        ]
    )
    plan = {
        "approved_before_results": True,
        "reviewed_by": "synthetic",
        "contrasts": [
            {"match": keys, "datasets": [f"d{i}" for i in range(n)], "direction": 1}
            for keys, n in ((small, 6), (larger, 9))
        ],
    }
    result = confirmatory_tests(datasets, plan).set_index("task_type")
    assert result.wilcoxon_method.tolist() == ["auto", "auto"]
    assert (result.holm_family_size == 2).all()
    assert result.loc["multiclass_classification", "holm_rank"] == 2
    assert result.loc["multiclass_classification", "holm_step_threshold_at_0_05"] == 0.05
    assert result.loc["regression", "holm_step_threshold_at_0_05"] == 0.025
    assert (result.p_holm < 0.05).all()  # Six datasets can pass in a mixed two-contrast family.
