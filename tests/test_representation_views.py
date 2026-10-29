"""Synthetic fixtures test pairing/plots only; never written as research results."""

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.evaluation.representation_views import decision_vs_embedding
from src.visualize import style
from src.visualize.decision_comparison import LABELS
from src.visualize.paired import MODES
from src.visualize.representation_views import (
    RECIPES,
    direct_comparison_figures,
    overlap_figures,
)


def fixture_folds():
    rows = []
    for dataset in range(3):
        for fold in range(5):
            base = dict(dataset_id=f"dataset{dataset}", task_type="binary_classification",
                        model_id="catboost", checkpoint_id="same", split_id=f"split{dataset}",
                        outer_fold=fold, metric="roc_auc", inference_status="prespecified")
            for i, provider in enumerate(LABELS):
                for j, mode in enumerate(MODES):
                    rows.append(dict(base, phase="experiment_1", provider=provider, feature_set=mode,
                                     arm_id=f"{dataset}/{provider}/{mode}", comparison_id=f"{provider}/{mode}",
                                     value=0.65 + 0.013*i + 0.01*j + 0.005*dataset, encoder=None, pca_scope=None, pca_components=None))
            for scope, width in RECIPES:
                rows.append(dict(base, phase="experiment_2", provider=None, feature_set="non_text_plus_frozen_e5",
                                 arm_id=f"{dataset}/e5/{scope}/{width}", comparison_id=f"e5/{scope}/{width}",
                                 value=0.70 + width/6000, encoder="e5_small_v2", pca_scope=scope, pca_components=width))
    return pd.DataFrame(rows)


def fixture_paired():
    rows = fixture_folds().query("phase == 'experiment_1'").copy()
    rows["reference"] = "non_text"
    rows["reference_value"] = 0.66
    rows["difference"] = rows.value - rows.reference_value
    cohorts = []
    for i, cohort in enumerate(style.COHORT_COLORS):
        group = rows.copy()
        group["metric"] = f"text.{cohort}.roc_auc"
        group["difference"] -= i * 0.02
        cohorts.append(group)
    return pd.concat([rows, *cohorts], ignore_index=True)


def test_direct_pairing_does_not_select_winners_and_never_crosses_splits():
    folds = fixture_folds()
    paired = decision_vs_embedding(folds)
    assert len(paired) == 3 * 5 * 12 * 3 * 7
    assert (paired.difference == paired.value - paired.reference_value).all()
    bad = folds.copy()
    bad.loc[bad.phase.eq("experiment_2"), "checkpoint_id"] = "different"
    assert decision_vs_embedding(bad).empty
    bad = folds.copy()
    bad.loc[bad.phase.eq("experiment_2"), "split_id"] += "different"
    assert decision_vs_embedding(bad).empty
    incomplete = folds[~(folds.encoder.notna() & folds.outer_fold.eq(4))]
    assert decision_vs_embedding(incomplete).empty
    with pytest.raises(ValueError, match="Multiple"):
        decision_vs_embedding(pd.concat([folds, folds.iloc[[0]]]))


def test_new_panels_require_real_evidence_and_render_at_paper_width():
    empty = pd.DataFrame()
    for make in (direct_comparison_figures, overlap_figures):
        assert list(make(empty)) == []
    style.apply()
    figures = [*direct_comparison_figures(decision_vs_embedding(fixture_folds())),
               *overlap_figures(fixture_paired())]
    assert len(figures) == 2
    for fig, _, caption in figures:
        try:
            fig.canvas.draw()
            assert fig.get_size_inches()[0] == style.WIDTH_FULL
            assert caption and not any("index" in ax.get_xlabel().lower() for ax in fig.axes)
        finally:
            plt.close(fig)
