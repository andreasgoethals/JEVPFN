"""A1 (exp01_outputs): stand-alone quality and calibration of each decision model.

REFERENCE ONLY — DECISION MODELS ARE NOT TABULAR PREDICTORS. Nothing is fitted or
recalibrated. Each decision model's stored JOINT-request probabilities (all available text
fields in one request; the artifacts pinned for A1 and RQ2) are scored directly against the
target, as if the decision output were the prediction. Per-column requests are never scored.

* Classification: ROC-AUC (binary; positive = second canonical class) or macro one-vs-rest
  ROC-AUC (multiclass), log loss, Brier score (binary: positive-class squared error;
  multiclass: summed over classes), expected calibration error of the top-label confidence
  over equal-width bins, and the reliability-diagram data of those bins.
* Regression: Spearman correlation between the expected evidence level (nine ordered levels
  −4…+4 weighted by their probabilities) and the target. Rank only; never an R².

Rows without a joint request (no usable text) are not scored and are reported as coverage;
nothing is imputed. Rows are renormalised only within the recorded probability-sum tolerance.
Every scored row counts once; the shared splits are irrelevant because nothing is trained.

Labels are used only for this after-the-fact description; the scores never select a model,
feature set, predictor, threshold or any other benchmark decision. The CPU export
(`comparison_report.build`) computes these tables on VSC; the notebook only renders them.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE_LABEL = "Reference only — decision models are not tabular predictors"
CLASSIFICATION_METRICS = {"binary_classification": "roc_auc", "multiclass_classification": "roc_auc_ovr_macro"}
REGRESSION_METRIC = "spearman_expected_level"
METRIC_LABELS = {"roc_auc": "ROC-AUC", "roc_auc_ovr_macro": "Macro OvR ROC-AUC",
                 "log_loss": "Log loss", "brier": "Brier score",
                 "ece": "Expected calibration error",
                 REGRESSION_METRIC: "Spearman ρ (rank only)"}
LEVELS = np.arange(-4, 5, dtype=float)
REPRESENTATION = "joint"
#: Larger deviations from a unit row sum are reported as invalid rather than renormalised.
SUM_TOLERANCE = 1e-3
COLUMNS = ["dataset_id", "task_type", "provider", "metric", "value", "scored_rows",
           "dataset_rows", "coverage", "status", "label"]
RELIABILITY_COLUMNS = ["dataset_id", "task_type", "provider", "bin", "lower", "upper", "count",
                       "mean_confidence", "accuracy"]


def joint_matrix(block, task, n_rows):
    """Full-dataset joint-request probabilities in csv row order and their availability."""
    from src.evaluation.row_benefit import block_probabilities

    if block.manifest["identity"]["representation"] != REPRESENTATION:
        raise ValueError("Stand-alone quality scores joint requests only")
    groups = block_probabilities(block, task, n_rows)
    if list(groups) != ["all_text"]:
        raise ValueError("A joint block has exactly one all-text group")
    matrix = groups["all_text"]
    # Scored per source row, not per distinct input: labels belong to rows, and identical
    # requests may carry different targets (owner decision, 20 November 2026).
    return matrix, np.isfinite(matrix).all(axis=1)


def _normalised(p):
    sums = p.sum(axis=1)
    if np.any(np.abs(sums - 1) > SUM_TOLERANCE):
        raise ValueError("Decision probabilities deviate from a unit sum beyond tolerance")
    return p / sums[:, None]


def top_label_correct(y_index, p):
    """Top-label correctness; an exact tie for the top gives each tied class equal credit."""
    top = p.max(axis=1, keepdims=True)
    tied = p == top
    return tied[np.arange(len(y_index)), y_index] / tied.sum(axis=1)


def reliability(y_index, p, bins):
    """Equal-width bins of the top-label confidence: count, mean confidence and accuracy."""
    confidence = p.max(axis=1)
    correct = top_label_correct(y_index, p)
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, bins - 1)
    rows = []
    for b in range(bins):
        inside = which == b
        rows.append({"bin": b, "lower": edges[b], "upper": edges[b + 1], "count": int(inside.sum()),
                     "mean_confidence": float(confidence[inside].mean()) if inside.any() else np.nan,
                     "accuracy": float(correct[inside].mean()) if inside.any() else np.nan})
    return pd.DataFrame(rows)


def expected_calibration_error(table: pd.DataFrame) -> float:
    used = table[table["count"] > 0]
    return float((used["count"] * (used.accuracy - used.mean_confidence).abs()).sum() / used["count"].sum())


def score_classification(y_index, p, task, bins):
    """Fixed probabilities only: AUC, log loss, Brier, ECE and reliability-diagram bins."""
    from sklearn.metrics import log_loss, roc_auc_score

    k = p.shape[1]
    present = len(np.unique(y_index)) == k
    if task == "binary_classification":
        auc = roc_auc_score(y_index, p[:, 1]) if present else np.nan
        brier = np.mean((p[:, 1] - (y_index == 1)) ** 2)
    else:
        auc = (roc_auc_score(y_index, p, multi_class="ovr", average="macro", labels=list(range(k)))
               if present else np.nan)
        brier = np.mean(((p - np.eye(k)[y_index]) ** 2).sum(axis=1))
    table = reliability(y_index, p, bins)
    scores = {CLASSIFICATION_METRICS[task]: float(auc),
              "log_loss": float(log_loss(y_index, np.clip(p, 1e-15, 1), labels=list(range(k)))),
              "brier": float(brier), "ece": expected_calibration_error(table)}
    return scores, table


def score_regression(y, p):
    from scipy.stats import spearmanr

    level = p @ LEVELS
    if len(y) < 3 or np.ptp(level) == 0 or np.ptp(y) == 0:
        return {REGRESSION_METRIC: np.nan}
    return {REGRESSION_METRIC: float(spearmanr(level, y).statistic)}


def target_values(dataset):
    """Regression targets as floats; class targets as canonical class indices."""
    from src.utils.serialization import canonical_json, scalar

    metadata = dataset.metadata
    raw = dataset.frame[metadata.target]
    if metadata.task_type == "regression":
        return pd.to_numeric(raw).to_numpy(float)
    lookup = {canonical_json(scalar(c)): i for i, c in enumerate(metadata.classes)}
    return np.array([lookup[canonical_json(scalar(v))] for v in raw], dtype=int)


def dataset_scores(dataset, blocks: dict, *, bins: int = 10):
    """Score every provider's joint block; ``blocks`` maps provider -> (block, source) or (None, reason).

    Returns ``(scores, reliability)`` tables.
    """
    metadata = dataset.metadata
    task = metadata.task_type
    y = target_values(dataset)
    n_rows = len(dataset.frame)
    base = {"dataset_id": metadata.dataset_id, "task_type": task, "label": REFERENCE_LABEL}
    rows, curves = [], []
    for provider, (block, source) in blocks.items():
        if block is None:
            rows.append({**base, "provider": provider, "metric": None, "dataset_rows": n_rows,
                         "status": f"pending: {source}"})
            continue
        matrix, available = joint_matrix(block, task, n_rows)
        try:
            p = _normalised(matrix[available])
        except ValueError as error:
            rows.append({**base, "provider": provider, "metric": None, "dataset_rows": n_rows,
                         "status": f"invalid: {error}"})
            continue
        if task == "regression":
            scores = score_regression(y[available], p)
        else:
            scores, table = score_classification(y[available], p, task, bins)
            curves.append(table.assign(dataset_id=metadata.dataset_id, task_type=task, provider=provider))
        for metric, value in scores.items():
            rows.append({**base, "provider": provider, "metric": metric, "value": value,
                         "scored_rows": int(available.sum()), "dataset_rows": n_rows,
                         "coverage": float(available.mean()), "status": "descriptive"})
    curves = (pd.concat(curves, ignore_index=True)[RELIABILITY_COLUMNS] if curves
              else pd.DataFrame(columns=RELIABILITY_COLUMNS))
    return pd.DataFrame(rows, columns=COLUMNS), curves


def pinned_artifacts() -> dict:
    """The decision artifacts pinned for A1 and RQ2 (`comparison_report.report_config`)."""
    from src.evaluation.comparison_report import report_config

    return report_config()["comparison"]["artifacts"]


def inventory_loader(pins: dict | None = None):
    """Read-only loader: canonical store first, then portable copy; joint requests only.

    Uses the artifact pinned for A1/RQ2 when one exists, otherwise the single available
    artifact; several unpinned versions are reported, never chosen between.
    """
    from src.evaluation.portable_features import notebook_inventory
    from src.jev.features import load_block

    pins = pinned_artifacts() if pins is None else pins
    inventories = {}

    def load(dataset_id, provider):
        if dataset_id not in inventories:
            inventories[dataset_id] = notebook_inventory(dataset_id)
        inventory = inventories[dataset_id]
        matches = inventory[inventory.provider.eq(provider) & inventory.representation.eq(REPRESENTATION)]
        pinned = pins.get(dataset_id, {}).get(provider, {}).get(REPRESENTATION)
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


def direct_scores(entries=None, cfg=None, *, providers=None, loader=None, datasets=None, bins=None):
    """Stand-alone scores and reliability bins for every selected dataset and decision model."""
    from src.data.loaders import load_catalog, load_dataset
    from src.data.selection import selected_entries
    from src.evaluation.comparison_report import report_config

    cfg = cfg or report_config()
    options = cfg["comparison"]
    providers = providers or options["providers"]
    bins = bins or options["standalone_quality"]["calibration_bins"]
    loader = loader or inventory_loader(options["artifacts"])
    if datasets is None:
        entries = entries if entries is not None else selected_entries(load_catalog(cfg["catalog"]), cfg)
        datasets = (load_dataset(e, allow_download=False) for e in entries)
    scores, curves, started = [], [], time.monotonic()
    for dataset in datasets:
        blocks = {p: loader(dataset.metadata.dataset_id, p) for p in providers}
        one, reliability_bins = dataset_scores(dataset, blocks, bins=bins)
        scores.append(one)
        if not reliability_bins.empty:
            curves.append(reliability_bins)
    scores = pd.concat(scores, ignore_index=True) if scores else pd.DataFrame(columns=COLUMNS)
    curves = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(columns=RELIABILITY_COLUMNS)
    scores.attrs["seconds"] = time.monotonic() - started
    return scores, curves


def summary_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Median and range over decision models per dataset and metric (descriptive)."""
    scored = scores[scores.status.eq("descriptive")] if not scores.empty else scores
    if scored.empty:
        return pd.DataFrame(columns=["dataset_id", "task_type", "metric", "models", "median",
                                     "minimum", "maximum", "mean_coverage"])
    return (scored.groupby(["dataset_id", "task_type", "metric"], sort=False)
            .agg(models=("provider", "nunique"), median=("value", "median"),
                 minimum=("value", "min"), maximum=("value", "max"),
                 mean_coverage=("coverage", "mean"))
            .reset_index())
