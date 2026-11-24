"""MulTaBench-style normalised score (`group_b.normalized_scores`, `summarize_normalized`):
per dataset and fold min-max over the compared configurations, negative R² clipped at 0,
tied or incomplete cells excluded, equal-dataset summaries with a dataset bootstrap."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import group_b


def scores(values, *, task="binary_classification", dataset="D", folds=(0,), model="m"):
    """{arm: value or [value per fold]} -> long fold-score frame for one dataset and predictor."""
    rows = []
    for arm, value in values.items():
        per_fold = value if isinstance(value, list) else [value] * len(folds)
        rows += [dict(dataset_id=dataset, task_type=task, model_id=model, representation=arm,
                      outer_fold=f, value=v) for f, v in zip(folds, per_fold)]
    return pd.DataFrame(rows)


def test_min_max_per_fold_then_mean_over_folds():
    frame = scores({"a": [0.6, 0.7], "b": [0.8, 0.7 + 0.2], "c": [0.7, 0.8]}, folds=(0, 1))
    out = group_b.normalized_scores(frame).set_index("representation")
    # Fold 0: a=0, b=1, c=0.5; fold 1: a=0, b=1, c=0.5.
    assert out.normalized_score.to_dict() == pytest.approx({"a": 0.0, "b": 1.0, "c": 0.5})
    assert out.folds.eq(2).all() and out.tied_folds.eq(0).all()
    assert out.normalized_score.between(0, 1).all()


def test_negative_r2_is_clipped_before_normalising_but_auc_is_not():
    regression = scores({"a": -3.0, "b": 0.5, "c": 0.25}, task="regression")
    out = group_b.normalized_scores(regression).set_index("representation").normalized_score
    assert out.to_dict() == pytest.approx({"a": 0.0, "b": 1.0, "c": 0.5})  # -3 -> 0, not stretching
    unclipped = group_b.normalized_scores(regression, clip_negative_r2=False).set_index("representation")
    assert unclipped.normalized_score["c"] == pytest.approx(3.25 / 3.5)
    auc = scores({"a": 0.3, "b": 0.5, "c": 0.4})  # AUC below 0.5 is a real (bad) score
    assert group_b.normalized_scores(auc).set_index("representation").normalized_score["c"] == pytest.approx(0.5)


def test_tied_cells_are_excluded_and_counted():
    # Fold 0: every R² is negative -> all clipped to 0 -> tied, excluded; fold 1 is used.
    frame = scores({"a": [-0.2, 0.1], "b": [-0.5, 0.3]}, task="regression", folds=(0, 1))
    full = group_b.normalized_scores(frame)
    out = full.set_index("representation")
    assert out.normalized_score.to_dict() == {"a": 0.0, "b": 1.0}
    assert out.folds.eq(1).all() and out.tied_folds.eq(1).all()
    assert full.attrs["excluded_cells"].reason.tolist() == ["all_tied"]
    # A dataset tied on every fold is absent (no arbitrary value) and listed.
    tied = scores({"a": 0.7, "b": 0.7}, dataset="T")
    both = group_b.normalized_scores(pd.concat([frame, tied]))
    assert "T" not in set(both.dataset_id)
    assert set(both.attrs["excluded_cells"].dataset_id) == {"D", "T"}


def test_incomplete_cells_are_excluded_for_every_configuration():
    frame = scores({"a": [0.6, 0.6], "b": [0.8, np.nan], "c": [0.7, 0.9]}, folds=(0, 1))
    out = group_b.normalized_scores(frame)
    assert out.folds.eq(1).all()  # fold 1 lacks b, so nobody is normalised on it
    assert out.attrs["excluded_cells"].reason.tolist() == ["incomplete"]
    missing = frame[~(frame.representation.eq("b") & frame.outer_fold.eq(1))]
    assert group_b.normalized_scores(missing).folds.eq(1).all()


def test_scores_depend_on_the_comparison_set_and_within_separates_sets():
    two = pd.concat([scores({"a": 0.6, "b": 0.8}, model="x"), scores({"a": 0.7, "b": 0.75}, model="y")])
    pooled = group_b.normalized_scores(two).set_index(["model_id", "representation"]).normalized_score
    assert pooled[("y", "a")] == pytest.approx(0.5) and pooled[("x", "b")] == 1.0
    separate = group_b.normalized_scores(two, within=("model_id",)).set_index(
        ["model_id", "representation"]).normalized_score
    assert separate[("y", "a")] == 0.0 and separate[("y", "b")] == 1.0
    with pytest.raises(ValueError, match="Several scores"):
        group_b.normalized_scores(pd.concat([two, two]))


def test_summary_pools_all_datasets_and_keeps_task_families():
    tasks = ["binary_classification"] * 3 + ["multiclass_classification"] * 3 + ["regression"] * 6
    parts = []
    for i, task in enumerate(tasks):
        low = -0.5 if task == "regression" else 0.5
        parts.append(scores({"baseline": low, "arm": 0.8 + 0.01 * i, "other": 0.6 + 0.02 * (i % 4)},
                            task=task, dataset=f"D{i:02d}", folds=(0, 1, 2, 3, 4)))
    per_dataset = group_b.normalized_scores(pd.concat(parts, ignore_index=True))
    summary = group_b.summarize_normalized(per_dataset, n_boot=300)
    assert set(summary.task_family) == {"all", "classification", "regression"}
    pooled = summary[summary.task_family.eq("all")].set_index("representation")
    assert pooled.datasets.eq(12).all() and pooled.configurations.eq(3).all()
    assert pooled.loc["arm", "mean_normalized_score"] == pytest.approx(1.0)
    assert pooled.loc["baseline", "mean_normalized_score"] == pytest.approx(0.0)
    other = pooled.loc["other"]
    assert other.normalized_ci_low <= other.mean_normalized_score <= other.normalized_ci_high
    assert other.median_normalized_ci_low <= other.median_normalized_score <= other.median_normalized_ci_high
    families = summary[summary.task_family.ne("all")].groupby("task_family").datasets.max().to_dict()
    assert families == {"classification": 6, "regression": 6}
    assert group_b.summarize_normalized(pd.DataFrame()).empty
    assert group_b.normalized_scores(pd.DataFrame()).empty
