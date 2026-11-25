"""Headroom-normalised gains, dataset bootstrap and predictor ranking agreement."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import aggregate

TEXT = {"BIN_A": 2, "BIN_B": 3, "BIN_ONE": 1, "REG_A": 2}


def arm(dataset, *, model="catboost", phase="experiment_1", feature_set="non_text_plus_joint",
        provider="jev", encoder=None, scope=None, components=None, baseline=0.8, augmented=0.9,
        folds=5, task="binary_classification"):
    metric = {"binary_classification": "roc_auc", "regression": "r2"}[task]
    identity = f"{dataset}/{model}/{phase}/{feature_set}/{provider}/{encoder}/{scope}{components}"
    return [dict(arm_id=identity, phase=phase, dataset_id=dataset, task_type=task, model_id=model,
                 checkpoint_id=model, feature_set=feature_set, provider=provider, encoder=encoder,
                 pca_scope=scope, pca_components=components, split_id=dataset, outer_fold=fold,
                 metric=metric, value=augmented, non_text_value=baseline,
                 difference=augmented - baseline, reference="non_text",
                 inference_status="prespecified", view="main")
            for fold in range(folds)]


def frame(*arms):
    return pd.DataFrame([row for rows in arms for row in rows])


def summary_of(paired, **kwargs):
    return aggregate.normalized_gain_summary(paired, text_columns=TEXT, roster=["jev"], **kwargs)


def pick(table, representation, model="catboost", scope="all"):
    rows = table[table.representation.eq(representation) & table.model_id.eq(model)
                 & table.dataset_scope.eq(scope)]
    assert len(rows) == 1
    return rows.iloc[0]


def test_normalisation_closes_share_of_headroom_for_auc_and_r2():
    paired = frame(arm("BIN_A", baseline=0.8, augmented=0.9),
                   arm("REG_A", baseline=-0.5, augmented=0.25, task="regression"))
    gains = aggregate.dataset_gains(paired).set_index(["representation", "dataset_id"])
    assert gains.loc[("decision/joint/jev", "BIN_A"), "gain"] == pytest.approx(0.5)
    # R² below zero has more than one unit of headroom; the ceiling is still 1.
    assert gains.loc[("decision/joint/jev", "REG_A"), "gain"] == pytest.approx(0.5)
    perfect = aggregate.dataset_gains(frame(arm("BIN_A", baseline=1.0, augmented=0.99)), eps=0.1)
    assert perfect.gain.iloc[0] == pytest.approx(-0.1)


def test_datasets_have_equal_weight_regardless_of_fold_rows():
    paired = frame(arm("BIN_A", baseline=0.8, augmented=0.9), arm("BIN_B", baseline=0.5, augmented=0.5))
    # Duplicate a dataset's metric rows under a second, irrelevant metric: no extra weight.
    extra = frame(arm("BIN_A", baseline=0.0, augmented=1.0)).assign(metric="accuracy")
    row = pick(summary_of(pd.concat([paired, extra]), n_boot=200), "decision/joint/jev")
    assert row.n_datasets == 2
    assert row.mean_gain == pytest.approx((0.5 + 0.0) / 2)


def test_dataset_bootstrap_is_seeded_and_contains_the_mean():
    paired = frame(*(arm(d, baseline=0.6, augmented=0.6 + g)
                     for d, g in zip(["BIN_A", "BIN_B", "BIN_ONE", "REG_A"], [0.05, 0.1, 0.2, 0.0])))
    paired.loc[paired.dataset_id.eq("REG_A"), ["task_type", "metric"]] = ["regression", "r2"]
    first = pick(summary_of(paired, seed=3), "decision/joint/jev")
    again = pick(summary_of(paired, seed=3), "decision/joint/jev")
    assert (first.ci_low, first.ci_high) == (again.ci_low, again.ci_high)
    # The resampled unit is the dataset: four dataset gains, not twenty fold rows.
    gains = np.array([0.125, 0.25, 0.5, 0.0])
    draws = gains[np.random.default_rng(3).integers(0, 4, size=(2000, 4))].mean(axis=1)
    assert (first.ci_low, first.ci_high) == pytest.approx(tuple(np.percentile(draws, [2.5, 97.5])))
    assert first.ci_low <= first.mean_gain <= first.ci_high
    assert first.ci_low > -1e-12 and first.ci_high < 0.5 + 1e-12


def test_arms_without_five_paired_folds_are_excluded():
    paired = frame(arm("BIN_A"), arm("BIN_B", folds=4))
    missing = frame(arm("BIN_ONE"))
    missing.loc[missing.outer_fold.eq(0), "non_text_value"] = np.nan
    table = summary_of(pd.concat([paired, missing]))
    row = pick(table, "decision/joint/jev")
    assert row.n_datasets == 1 and row.dataset_ids == "BIN_A"


def test_single_text_datasets_leave_mode_contrasts_but_stay_against_baseline():
    paired = frame(*(arm(d, feature_set=f) for d in ["BIN_A", "BIN_ONE"]
                     for f in aggregate.DECISION_MODES))
    table = summary_of(paired)
    assert pick(table, "decision/per_column/median").n_datasets == 2
    contrast = pick(table, "decision/per_column/median", scope="multi_text")
    assert contrast.n_datasets == 1 and contrast.dataset_ids == "BIN_A"
    # Embedding recipes have no separate/joint/both identity, so they have no restricted scope.
    embedded = frame(arm("BIN_ONE", phase="experiment_2", feature_set="non_text_plus_frozen_e5",
                         provider=None, encoder="e5_small_v2", scope="joint", components=64))
    assert set(summary_of(embedded).dataset_scope) == {"all"}


def test_median_over_decision_models_requires_every_model_per_dataset():
    paired = frame(arm("BIN_A", provider="jev", augmented=0.82), arm("BIN_A", provider="kev", augmented=0.9),
                   arm("BIN_A", provider="sii", augmented=0.84), arm("BIN_B", provider="jev"))
    gains = aggregate.dataset_gains(paired, roster=("jev", "kev", "sii"))
    median = gains[gains.aggregation.eq(aggregate.MEDIAN)]
    assert list(median.dataset_id) == ["BIN_A"]
    assert median.gain.iloc[0] == pytest.approx(0.2)
    assert median.decision_models.iloc[0] == 3


def test_kendall_tau_b_on_constructed_rankings():
    recipes = [("e5_small_v2", "joint", 64), ("e5_large_v2", "joint", 64),
               ("qwen3_embedding_0_6b", "joint", 64)]
    rows = []
    for model, gains in {"catboost": [0.1, 0.2, 0.3], "linear": [0.1, 0.2, 0.3],
                         "tabpfn_3_5": [0.3, 0.2, 0.1]}.items():
        for (encoder, scope, components), gain in zip(recipes, gains):
            rows.append(arm("BIN_A", model=model, phase="experiment_2", provider=None,
                            feature_set="non_text_plus_frozen_e5", encoder=encoder, scope=scope,
                            components=components, baseline=0.5, augmented=0.5 + gain / 2))
    pairs, mean_tau = aggregate.ranking_stability(frame(*rows), text_columns=TEXT)
    tau = pairs.set_index(["model_a", "model_b"]).kendall_tau_b
    assert tau[("catboost", "linear")] == pytest.approx(1.0)
    assert tau[("catboost", "tabpfn_3_5")] == pytest.approx(-1.0)
    assert mean_tau == pytest.approx(-1 / 3)
    assert aggregate.kendall_tau_b([1, 2, 2], [1, 2, 3]) == pytest.approx(2 / np.sqrt(6))


def test_empty_input_gives_empty_frames_and_no_figure():
    empty = pd.DataFrame()
    table = aggregate.normalized_gain_summary(empty, text_columns=TEXT)
    assert table.empty and list(table.columns) == aggregate.SUMMARY_COLUMNS
    pairs, mean_tau = aggregate.ranking_stability(empty, text_columns=TEXT)
    assert pairs.empty and np.isnan(mean_tau)
