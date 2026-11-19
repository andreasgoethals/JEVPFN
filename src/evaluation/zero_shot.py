"""Exp 4 appendix candidate: how decision models alone score on the outer-test rows.

REFERENCE ONLY — DECISION MODELS ARE NOT TABULAR PREDICTORS. Nothing is fitted. Each
decision model's stored probabilities are read from the feature stores (read-only; the
canonical per-dataset store first, then the portable notebook copy in
`data/decision_features/`) and scored directly on the outer-test rows of the shared
five-fold splits, as if the decision output were the prediction:

Artifacts are the ones pinned for RQ1A and RQ2 (else the single stored version).

* Classification uses the joint request (all text fields in one request) as the zero-shot
  prediction; a per-column request is used only when the dataset has a single text column
  (then both requests see the same text). Probabilities follow the dataset's canonical class
  order (binary: ``[1 - p_positive, p_positive]``, positive = second class). Scores are
  ROC-AUC (binary) or macro one-vs-rest ROC-AUC (multiclass) and log loss.
* Regression has no class probabilities: the expected evidence score (nine ordered levels
  −4..4 weighted by their probabilities) is rank-correlated with the target (Spearman).
  It is a rank-only description, never an R² or a calibrated estimate.

Rows without a stored probability (no usable text) receive the uninformative uniform
distribution (classification) or the neutral score 0 (regression); coverage is reported.
Rows are renormalised only within the probability-sum tolerance recorded per model.

Test labels are used only for this after-the-fact description. The scores never feed a
selection, a feature set, a predictor, a threshold or any other benchmark decision.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE_LABEL = "Reference only — decision models are not tabular predictors"
CLASSIFICATION_METRICS = {"binary_classification": "roc_auc", "multiclass_classification": "roc_auc_ovr_macro"}
REGRESSION_METRIC = "spearman_expected_score"
METRIC_LABELS = {"roc_auc": "ROC-AUC", "roc_auc_ovr_macro": "Macro OvR ROC-AUC",
                 "log_loss": "Log loss", REGRESSION_METRIC: "Spearman ρ (rank only)"}
LEVELS = np.arange(-4, 5, dtype=float)
#: Larger deviations from a unit row sum are reported as invalid rather than renormalised.
SUM_TOLERANCE = 1e-3
COLUMNS = ["dataset_id", "task_type", "provider", "source_mode", "metric", "value", "fold_sd",
           "folds", "coverage", "n_test_rows", "status", "label"]


def prediction_matrix(block, task, n_rows, text_columns):
    """Full-dataset probabilities of one decision artifact for zero-shot scoring, or None.

    Joint requests have one group (`all_text`); per-column requests are admitted only with
    exactly one text column. Returns ``(matrix, available_mask)`` in csv row order.
    """
    from src.evaluation.row_benefit import block_probabilities

    mode = block.manifest["identity"]["representation"]
    if mode == "per_column" and len(text_columns) != 1:
        return None
    groups = block_probabilities(block, task, n_rows)
    if len(groups) != 1:
        return None
    (matrix,) = groups.values()
    available = np.isfinite(matrix).all(axis=1)
    return matrix, available


def _complete(matrix, available, task):
    """Fill rows without text with the uninformative prediction; renormalise within tolerance."""
    p = np.array(matrix, dtype=float)
    k = p.shape[1]
    p[~available] = 1.0 / k
    sums = p.sum(axis=1)
    if np.any(np.abs(sums - 1) > SUM_TOLERANCE):
        raise ValueError("Decision probabilities deviate from a unit sum beyond tolerance")
    return p / sums[:, None]


def score_classification(y_index, p, task):
    """Threshold-free AUC and log loss of fixed probabilities on one test partition."""
    from sklearn.metrics import log_loss, roc_auc_score

    k = p.shape[1]
    present = len(np.unique(y_index)) == k
    if task == "binary_classification":
        auc = roc_auc_score(y_index, p[:, 1]) if present else np.nan
    else:
        auc = roc_auc_score(y_index, p, multi_class="ovr", average="macro", labels=list(range(k))) if present else np.nan
    loss = log_loss(y_index, np.clip(p, 1e-15, 1), labels=list(range(k)))
    return {CLASSIFICATION_METRICS[task]: float(auc), "log_loss": float(loss)}


def score_regression(y, p):
    from scipy.stats import spearmanr

    score = p @ LEVELS
    if np.ptp(score) == 0 or np.ptp(y) == 0:
        return {REGRESSION_METRIC: np.nan}
    return {REGRESSION_METRIC: float(spearmanr(score, y).statistic)}


def dataset_scores(dataset, splits, blocks: dict) -> pd.DataFrame:
    """Score every provider's block on the outer-test rows of the shared splits.

    ``blocks`` maps provider -> ``(block, source_mode)`` or ``(None, reason)``.
    """
    from src.utils.serialization import canonical_json, scalar

    metadata = dataset.metadata
    task = metadata.task_type
    y_raw = dataset.frame[metadata.target]
    n_rows = len(dataset.frame)
    if task == "regression":
        y = pd.to_numeric(y_raw).to_numpy(float)
    else:
        lookup = {canonical_json(scalar(c)): i for i, c in enumerate(metadata.classes)}
        y = np.array([lookup[canonical_json(scalar(v))] for v in y_raw], dtype=int)
    rows = []
    base = {"dataset_id": metadata.dataset_id, "task_type": task, "label": REFERENCE_LABEL}
    for provider, (block, mode) in blocks.items():
        if block is None:
            rows.append({**base, "provider": provider, "source_mode": None, "metric": None,
                         "status": f"pending: {mode}"})
            continue
        found = prediction_matrix(block, task, n_rows, list(metadata.text_columns))
        if found is None:
            rows.append({**base, "provider": provider, "source_mode": mode, "metric": None,
                         "status": "not_scored: per-column request with several text columns"})
            continue
        matrix, available = found
        try:
            p = _complete(matrix, available, task)
        except ValueError as error:
            rows.append({**base, "provider": provider, "source_mode": mode, "metric": None,
                         "status": f"invalid: {error}"})
            continue
        per_fold = []
        for fold in splits["folds"]:
            test = np.asarray(fold["test"], dtype=int)
            scores = (score_regression(y[test], p[test]) if task == "regression"
                      else score_classification(y[test], p[test], task))
            per_fold.append(scores | {"coverage": float(available[test].mean()), "n": len(test)})
        frame = pd.DataFrame(per_fold)
        for metric in [c for c in frame if c not in {"coverage", "n"}]:
            values = frame[metric].astype(float)
            rows.append({**base, "provider": provider, "source_mode": mode, "metric": metric,
                         "value": values.mean(), "fold_sd": values.std(ddof=1),
                         "folds": int(values.notna().sum()), "coverage": frame.coverage.mean(),
                         "n_test_rows": int(frame.n.sum()), "status": "descriptive"})
    return pd.DataFrame(rows, columns=COLUMNS)


def pinned_artifacts() -> dict:
    """The decision artifacts pinned for RQ1A and RQ2 (`comparison_report.report_config`)."""
    from src.evaluation.comparison_report import report_config

    return report_config()["comparison"]["artifacts"]


def inventory_loader(pins: dict | None = None):
    """Read-only loader over the notebook inventory: canonical store first, then portable copy.

    Uses the artifact pinned for RQ1A/RQ2 when one exists, otherwise the single available
    artifact; several unpinned versions are reported, never chosen between.
    """
    from src.evaluation.portable_features import notebook_inventory
    from src.jev.features import load_block

    pins = pinned_artifacts() if pins is None else pins
    inventories = {}

    def load(dataset_id, provider, mode, cfg=None):
        if dataset_id not in inventories:
            inventories[dataset_id] = notebook_inventory(dataset_id)
        inventory = inventories[dataset_id]
        matches = inventory[inventory.provider.eq(provider) & inventory.representation.eq(mode)]
        pinned = pins.get(dataset_id, {}).get(provider, {}).get(mode)
        if pinned and not matches.empty:
            matches = matches[matches.artifact_id.eq(pinned)]
        if matches.empty:
            return None, "artifact_not_found"
        if len(matches) > 1:
            return None, "several_unpinned_artifacts"
        selected = matches.iloc[0]
        block = load_block(Path(selected.feature_store), artifact_id=selected.artifact_id)
        if block.manifest["identity"]["dataset_id"] != dataset_id:
            raise ValueError("Feature file belongs to a different dataset")
        return block, selected.feature_store

    return load


def _blocks(dataset, roster, cfg, loader):
    text_columns = list(dataset.metadata.text_columns)
    blocks = {}
    for provider in roster:
        block, source = loader(dataset.metadata.dataset_id, provider, "joint", cfg)
        mode = "joint"
        if block is None and len(text_columns) == 1:
            block, source = loader(dataset.metadata.dataset_id, provider, "per_column", cfg)
            mode = "per_column"
        blocks[provider] = (block, mode) if block is not None else (None, source)
    return blocks


def direct_scores(entries=None, cfg=None, *, loader=None, datasets=None) -> pd.DataFrame:
    """Zero-shot reference scores for every selected dataset and configured decision model."""
    from src.data.loaders import load_catalog, load_dataset
    from src.data.selection import selected_entries
    from src.evaluation.protocol import shared_splits
    from src.utils.config import load_yaml

    cfg = cfg or load_yaml("experiment_1/main")
    loader = loader or inventory_loader()
    if datasets is None:
        entries = entries if entries is not None else selected_entries(load_catalog(), cfg)
        datasets = (load_dataset(e, allow_download=False) for e in entries)
    tables, started = [], time.monotonic()
    for dataset in datasets:
        splits = shared_splits(dataset)
        tables.append(dataset_scores(dataset, splits, _blocks(dataset, cfg["candidate_extractors"], cfg, loader)))
    result = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=COLUMNS)
    result.attrs["seconds"] = time.monotonic() - started
    return result


def structured_references(folds) -> tuple[pd.DataFrame, str]:
    """Per classification dataset: best no-text predictor and best Exp 4 arm (outer-test means).

    Both are maxima over saved, complete five-fold Exp 4 recipes, chosen after the fact for this
    description only. Returns ``(table, status)``; no saved results give an empty, pending table.
    """
    from src.evaluation.notebook_study import headline_summary

    columns = ["dataset_id", "metric", "no_text_best", "no_text_model", "best_structured",
               "best_structured_arm"]
    summary = headline_summary(folds) if folds is not None and not folds.empty else pd.DataFrame()
    if summary.empty:
        return pd.DataFrame(columns=columns), "pending: no completed Exp 4 outer-test results"
    summary = summary[summary.phase.eq("experiment_1") & summary.task_type.ne("regression")]
    if summary.empty:
        return pd.DataFrame(columns=columns), "pending: no completed Exp 4 classification results"
    rows = []
    for dataset_id, group in summary.groupby("dataset_id"):
        no_text = group[group.feature_set.eq("non_text")]
        best_no_text = no_text.loc[no_text["mean"].idxmax()] if not no_text.empty else None
        best = group.loc[group["mean"].idxmax()]
        rows.append({
            "dataset_id": dataset_id, "metric": best.metric,
            "no_text_best": best_no_text["mean"] if best_no_text is not None else np.nan,
            "no_text_model": best_no_text.model_id if best_no_text is not None else None,
            "best_structured": best["mean"],
            "best_structured_arm": f"{best.model_id}/{best.feature_set}"
                                   + (f"/{best.provider}" if isinstance(best.provider, str) else ""),
        })
    return pd.DataFrame(rows, columns=columns), "available"


def summary_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Median and range over decision models per dataset and metric (descriptive)."""
    scored = scores[scores.status.eq("descriptive")]
    if scored.empty:
        return pd.DataFrame(columns=["dataset_id", "task_type", "metric", "models", "median",
                                     "minimum", "maximum", "mean_coverage"])
    return (scored.groupby(["dataset_id", "task_type", "metric"], sort=False)
            .agg(models=("provider", "nunique"), median=("value", "median"),
                 minimum=("value", "min"), maximum=("value", "max"),
                 mean_coverage=("coverage", "mean"))
            .reset_index())
