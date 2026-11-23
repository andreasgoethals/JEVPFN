"""Fold-level win rates and PAMA with dataset-cluster bootstrap intervals (descriptive).

Both statistics count (dataset, outer fold) observations: 18 datasets x 5 outer folds = 90
cells when complete. The five outer folds of one dataset share its rows, so they are NOT
independent observations: every interval here is a percentile bootstrap that resamples
DATASETS with replacement and keeps all of a resampled dataset's folds together (a cluster
bootstrap). Folds are never resampled individually, and these shares carry no p-value; the
dataset-level Wilcoxon + Holm family in `inference` remains the confirmatory analysis.

* `fold_win_rate`: share of the paired (dataset, fold) cells on which an arm beats a
  reference (e.g. the same predictor without text); ties are counted separately.
* `pama`: Probability of Achieving Maximal Accuracy (Fernandez-Delgado et al. 2014), the share
  of (dataset, fold) cells on which a learner attains the top score; ties at the top credit
  every tied learner, so the shares can sum to more than one.

Input is a long frame with columns `FOLD_COLUMNS`. A (dataset, fold) cell enters only when
every compared learner has one finite score there; incomplete or non-finite cells are never
silently dropped but excluded explicitly and counted in ``excluded_incomplete_cells`` (as in
the reference implementation, which counts complete cells only). Callers should pass every
expected (dataset, fold, learner) row, with NaN for an absent score, so that a cell missing for
every learner is counted too. Scores are oriented by ``higher_is_better``; equality is exact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FOLD_COLUMNS = ("dataset_id", "outer_fold", "learner", "score")
BOOTSTRAP_UNIT = "dataset"  # Cluster bootstrap: a dataset's folds are resampled together.


def _oriented_cells(scores: pd.DataFrame, learners, higher_is_better: bool):
    """(complete (dataset, fold) x learner matrix oriented larger-is-better, excluded count)."""
    missing = [c for c in FOLD_COLUMNS if c not in scores]
    if missing:
        raise ValueError(f"Fold scores lack columns {missing}")
    learners = list(learners)
    if len(set(learners)) != len(learners) or len(learners) < 2:
        raise ValueError("Compare at least two distinct learners")
    absent = [learner for learner in learners if learner not in set(scores.learner)]
    if absent:
        raise ValueError(f"Learners have no fold rows: {absent}")
    part = scores[scores.learner.isin(learners)]
    if part.duplicated(["dataset_id", "outer_fold", "learner"]).any():
        raise ValueError("Several scores for one dataset, fold and learner; select one campaign")
    table = part.pivot(index=["dataset_id", "outer_fold"], columns="learner", values="score")
    table = table.reindex(columns=learners).astype(float)
    complete = np.isfinite(table.to_numpy()).all(axis=1)
    kept = table[complete]
    if kept.empty:
        raise ValueError("No complete (dataset, fold) cell for these learners")
    return (kept if higher_is_better else -kept), int((~complete).sum())


def _cluster_bootstrap(counts: np.ndarray, cells: np.ndarray, *, resamples: int, seed: int, level: float):
    """Percentile interval of sum(counts)/sum(cells) over datasets resampled with replacement.

    `counts` is datasets x statistics, `cells` the complete cells per dataset; each draw takes
    whole datasets (all their folds), never individual folds.
    """
    if resamples < 1 or not 0 < level < 1:
        raise ValueError("Use at least one resample and a level in (0, 1)")
    draws = np.random.default_rng(seed).integers(0, len(cells), size=(resamples, len(cells)))
    shares = counts[draws].sum(axis=1) / cells[draws].sum(axis=1)[:, None]
    tail = 100 * (1 - level) / 2
    return np.percentile(shares, tail, axis=0), np.percentile(shares, 100 - tail, axis=0)


def fold_win_rate(scores: pd.DataFrame, *, learner, versus, higher_is_better: bool,
                  resamples: int = 10_000, seed: int = 0, level: float = 0.95) -> dict:
    """Wins, ties and losses of `learner` against `versus` over paired (dataset, fold) cells.

    ``win_rate = wins / fold_cells`` over the complete cells only (both learners finite);
    ``excluded_incomplete_cells`` counts the rest. The interval is a percentile bootstrap over
    DATASETS (cluster bootstrap with all of a dataset's folds), not over folds, because the
    folds of a dataset share data. Descriptive; no p-value.
    """
    cells, excluded = _oriented_cells(scores, [learner, versus], higher_is_better)
    difference = cells[learner] - cells[versus]
    per = pd.DataFrame({"win": difference > 0, "tie": difference == 0, "loss": difference < 0})
    per = per.groupby(level="dataset_id").agg(["sum", "size"])
    counts = per.xs("sum", axis=1, level=1)[["win", "tie", "loss"]].to_numpy(float)
    n = per[("win", "size")].to_numpy(float)
    low, high = _cluster_bootstrap(counts, n, resamples=resamples, seed=seed, level=level)
    total, wins, ties, losses = int(n.sum()), *(int(v) for v in counts.sum(axis=0))
    return {
        "learner": learner, "versus": versus, "datasets": len(n), "fold_cells": total,
        "excluded_incomplete_cells": excluded, "wins": wins, "ties": ties, "losses": losses,
        "win_rate": wins / total, "tie_rate": ties / total, "loss_rate": losses / total,
        "win_rate_ci_low_dataset_bootstrap": float(low[0]),
        "win_rate_ci_high_dataset_bootstrap": float(high[0]),
        "ci_level": level, "bootstrap_unit": BOOTSTRAP_UNIT, "bootstrap_resamples": resamples,
    }


def fold_win_rates(scores: pd.DataFrame, pairs, *, higher_is_better: bool, **bootstrap) -> pd.DataFrame:
    """`fold_win_rate` for every (learner, versus) pair, one row each."""
    return pd.DataFrame([fold_win_rate(scores, learner=a, versus=b, higher_is_better=higher_is_better,
                                       **bootstrap) for a, b in pairs])


def pama(scores: pd.DataFrame, *, higher_is_better: bool, learners=None,
         resamples: int = 10_000, seed: int = 0, level: float = 0.95) -> pd.DataFrame:
    """Fold-level PAMA (Fernandez-Delgado et al. 2014) of every learner, best first.

    A (dataset, fold) cell counts only when EVERY learner has a finite score there, so no
    learner gains from a missing competitor (``excluded_incomplete_cells`` counts the rest).
    Ties at the top credit each tied learner (``shared_top_cells``). The interval is a
    percentile bootstrap over DATASETS (cluster bootstrap with all of a dataset's folds).
    """
    learners = list(dict.fromkeys(scores.learner)) if learners is None else list(learners)
    cells, excluded = _oriented_cells(scores, learners, higher_is_better)
    values = cells.to_numpy()
    top = values == values.max(axis=1, keepdims=True)
    shared = top & (top.sum(axis=1, keepdims=True) > 1)
    datasets = cells.index.get_level_values("dataset_id")
    per = pd.DataFrame(top, columns=learners).groupby(np.asarray(datasets), sort=True).sum()
    n = pd.Series(1, index=datasets).groupby(level=0, sort=True).size().to_numpy(float)
    low, high = _cluster_bootstrap(per.to_numpy(float), n, resamples=resamples, seed=seed, level=level)
    total = int(n.sum())
    out = pd.DataFrame({
        "learner": learners, "top_cells": top.sum(axis=0).astype(int),
        "shared_top_cells": shared.sum(axis=0).astype(int), "fold_cells": total,
        "datasets": len(n), "excluded_incomplete_cells": excluded,
        "pama": top.sum(axis=0) / total,
        "pama_ci_low_dataset_bootstrap": low, "pama_ci_high_dataset_bootstrap": high,
        "ci_level": level, "bootstrap_unit": BOOTSTRAP_UNIT, "bootstrap_resamples": resamples,
    })
    return out.sort_values("pama", ascending=False, kind="stable").reset_index(drop=True)
