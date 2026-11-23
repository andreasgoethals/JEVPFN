"""Fold-level win rates and PAMA (Fernandez-Delgado et al. 2014) with dataset-cluster bootstrap CIs."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import fold_wins


def long(table):
    """{(dataset, fold): {learner: score}} -> long fold frame."""
    return pd.DataFrame([{"dataset_id": d, "outer_fold": f, "learner": learner, "score": value}
                         for (d, f), row in table.items() for learner, value in row.items()])


def paired(arm, reference):
    """Two learners over datasets x folds from equal-shaped arrays."""
    arm, reference = np.asarray(arm, float), np.asarray(reference, float)
    return long({(f"d{i}", j): {"arm": arm[i, j], "ref": reference[i, j]}
                 for i in range(arm.shape[0]) for j in range(arm.shape[1])})


def test_win_rate_counts_wins_ties_and_losses_and_respects_direction():
    reference = np.full((2, 5), 0.5)
    arm = np.array([[0.6, 0.6, 0.5, 0.4, 0.6], [0.6, 0.6, 0.6, 0.5, 0.4]])
    result = fold_wins.fold_win_rate(paired(arm, reference), learner="arm", versus="ref",
                                     higher_is_better=True, resamples=500)
    assert (result["wins"], result["ties"], result["losses"], result["fold_cells"]) == (6, 2, 2, 10)
    assert result["win_rate"] == pytest.approx(0.6) and result["tie_rate"] == pytest.approx(0.2)
    assert result["datasets"] == 2 and result["excluded_incomplete_cells"] == 0
    assert result["bootstrap_unit"] == "dataset"
    assert result["win_rate_ci_low_dataset_bootstrap"] <= 0.6 <= result["win_rate_ci_high_dataset_bootstrap"]
    loss = fold_wins.fold_win_rate(paired(arm, reference), learner="arm", versus="ref",
                                   higher_is_better=False, resamples=500)
    assert (loss["wins"], loss["ties"], loss["losses"]) == (2, 2, 6)


def test_incomplete_and_nonfinite_cells_are_excluded_and_counted_never_silently():
    reference = np.full((3, 5), 0.5)
    arm = np.full((3, 5), 0.6)
    arm[0, 1], reference[1, 2] = np.nan, np.inf
    frame = paired(arm, reference)
    frame = frame[~(frame.dataset_id.eq("d2") & frame.outer_fold.eq(4) & frame.learner.eq("arm"))]
    result = fold_wins.fold_win_rate(frame, learner="arm", versus="ref", higher_is_better=True, resamples=100)
    assert result["excluded_incomplete_cells"] == 3 and result["fold_cells"] == 12
    assert result["wins"] == 12 and result["win_rate"] == 1.0
    with pytest.raises(ValueError, match="No complete"):
        fold_wins.fold_win_rate(frame.assign(score=np.nan), learner="arm", versus="ref",
                                higher_is_better=True)
    with pytest.raises(ValueError, match="no fold rows"):
        fold_wins.fold_win_rate(frame, learner="arm", versus="other", higher_is_better=True)
    with pytest.raises(ValueError, match="Several scores"):
        fold_wins.fold_win_rate(pd.concat([frame, frame.iloc[:1]]), learner="arm", versus="ref",
                                higher_is_better=True)


def test_bootstrap_resamples_whole_datasets_not_folds():
    # Two datasets, one always won, one always lost: a dataset resample can only give 0, 1/2 or 1.
    low, high = fold_wins._cluster_bootstrap(np.array([[5.0], [0.0]]), np.array([5.0, 5.0]),
                                             resamples=2000, seed=1, level=0.95)
    assert low[0] == 0.0 and high[0] == 1.0
    # 18 datasets x 5 folds, nine datasets all wins and nine all losses: the fold-level share is
    # 0.5 with 90 cells, but the dataset-cluster interval reflects N = 18 clusters (SE ~0.12),
    # much wider than an (invalid) fold-level bootstrap would give (SE ~0.05).
    arm = np.repeat(np.r_[np.ones(9), np.zeros(9)][:, None], 5, axis=1)
    result = fold_wins.fold_win_rate(paired(arm, np.full((18, 5), 0.5)), learner="arm", versus="ref",
                                     higher_is_better=True, resamples=4000, seed=2)
    assert result["fold_cells"] == 90 and result["win_rate"] == pytest.approx(0.5)
    assert result["win_rate_ci_low_dataset_bootstrap"] < 0.33
    assert result["win_rate_ci_high_dataset_bootstrap"] > 0.67
    again = fold_wins.fold_win_rate(paired(arm, np.full((18, 5), 0.5)), learner="arm", versus="ref",
                                    higher_is_better=True, resamples=4000, seed=2)
    assert again == result  # seeded


def test_pama_credits_every_tied_top_learner_and_uses_complete_cells_only():
    table = {("d0", 0): {"a": 0.9, "b": 0.9, "c": 0.1},
             ("d0", 1): {"a": 0.8, "b": 0.7, "c": 0.1},
             ("d1", 0): {"a": 0.1, "b": 0.2, "c": 0.3},
             ("d1", 1): {"a": 0.5, "b": np.nan, "c": 0.9}}  # incomplete: excluded, counted
    result = fold_wins.pama(long(table), higher_is_better=True, resamples=200).set_index("learner")
    assert result.fold_cells.eq(3).all() and result.excluded_incomplete_cells.eq(1).all()
    assert result.top_cells.to_dict() == {"a": 2, "b": 1, "c": 1}
    assert result.shared_top_cells.to_dict() == {"a": 1, "b": 1, "c": 0}
    assert result.pama.sum() > 1  # ties credit both
    assert result.loc["a", "pama"] == pytest.approx(2 / 3)
    assert list(result.index) == ["a", "b", "c"]  # best first
    assert (result.pama_ci_low_dataset_bootstrap <= result.pama).all()
    assert (result.pama <= result.pama_ci_high_dataset_bootstrap).all()
    lower = fold_wins.pama(long(table), higher_is_better=False, resamples=200).set_index("learner")
    assert lower.top_cells.to_dict() == {"c": 2, "a": 1, "b": 0}
