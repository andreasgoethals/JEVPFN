"""Score supplied predictions without training, threshold selection or data splitting.

Probability column order is explicit. Undefined metrics are null with a reason, not
invented scores. Binary and multiclass Brier conventions are deliberately different.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import metrics as sk

from src.utils.serialization import canonical_json, scalar

METRICS_VERSION = "prediction-metrics-v1"


def _put(result, name, value, reason="Metric is undefined for these observations"):
    if value is None or not np.isfinite(value):
        result["metrics"][name] = None
        result["undefined"][name] = reason
    else:
        result["metrics"][name] = float(value)


def _report():
    return {"version": METRICS_VERSION, "metrics": {}, "undefined": {}}


def _reliability(probability, observed, bins):
    if not isinstance(bins, int) or bins < 2:
        raise ValueError("calibration_bins must be an integer >= 2")
    index = np.minimum((probability * bins).astype(int), bins - 1)
    table, errors = [], []
    for b in range(bins):
        mask = index == b
        n = int(mask.sum())
        prediction = float(probability[mask].mean()) if n else None
        frequency = float(observed[mask].mean()) if n else None
        table.append(
            {
                "left": b / bins,
                "right": (b + 1) / bins,
                "count": n,
                "mean_probability": prediction,
                "observed_fraction": frequency,
            }
        )
        if n:
            errors.append((n / len(index), abs(prediction - frequency)))
    return table, sum(w * e for w, e in errors), max(e for _, e in errors)


def _decisions(y, predicted, k, positive=None):
    result = _report()
    labels = np.arange(k)
    matrix = sk.confusion_matrix(y, predicted, labels=labels)
    precision, recall, f1, support = sk.precision_recall_fscore_support(
        y, predicted, labels=labels, zero_division=0
    )
    _put(result, "accuracy", sk.accuracy_score(y, predicted))
    _put(
        result,
        "balanced_accuracy",
        recall.mean() if (support > 0).all() else None,
        "At least one declared class is absent from this evaluation fold",
    )
    for average in ("macro", "micro", "weighted"):
        p, r, f, _ = sk.precision_recall_fscore_support(
            y, predicted, labels=labels, average=average, zero_division=0
        )
        for metric, value in [("precision", p), ("recall", r), ("f1", f)]:
            _put(result, f"{metric}_{average}", value)
    _put(
        result,
        "mcc",
        sk.matthews_corrcoef(y, predicted)
        if len(np.unique(y)) > 1 and len(np.unique(predicted)) > 1
        else None,
        "MCC is undefined when targets or decisions are constant",
    )
    expected = float(np.dot(matrix.sum(axis=0), matrix.sum(axis=1))) / len(y) ** 2
    _put(
        result,
        "cohen_kappa",
        (np.trace(matrix) / len(y) - expected) / (1 - expected) if expected < 1 else None,
    )
    result["confusion_matrix"] = matrix.tolist()
    result["per_class"] = [
        {
            "class_index": i,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(k)
    ]
    result["zero_division_convention"] = 0
    if positive is not None:
        tp = int(matrix[positive, positive])
        fn = int(matrix[positive].sum() - tp)
        fp = int(matrix[:, positive].sum() - tp)
        tn = len(y) - tp - fn - fp
        result["counts"] = dict(tn=tn, fp=fp, fn=fn, tp=tp)
        for metric, value in [
            ("precision", precision[positive]),
            ("recall", recall[positive]),
            ("f1", f1[positive]),
        ]:
            _put(result, metric, value)
        _put(result, "specificity", tn / (tn + fp) if tn + fp else None)
        _put(result, "negative_predictive_value", tn / (tn + fn) if tn + fn else None)
    return result


def classification_metrics(
    y_true,
    probabilities,
    *,
    classes,
    positive_class=None,
    selected_threshold=None,
    calibration_bins=15,
):
    """Evaluate binary or multiclass probabilities with an explicit complete vocabulary.

    selected_threshold must have been selected on inner-validation predictions by the
    caller. This function never optimizes it or uses these targets to change predictions.
    """
    classes = [scalar(c) for c in classes]
    keys = [canonical_json(c) for c in classes]
    if len(keys) < 2 or len(set(keys)) != len(keys) or None in classes:
        raise ValueError("classes must contain >=2 distinct nonmissing values")
    lookup = {key: i for i, key in enumerate(keys)}
    try:
        y = np.array([lookup[canonical_json(scalar(v))] for v in y_true], dtype=int)
    except KeyError as exc:
        raise ValueError("Target missing or outside declared class vocabulary") from exc
    p = np.asarray(probabilities, dtype=float)
    k = len(classes)
    if not len(y) or p.shape != (len(y), k):
        raise ValueError("Expected nonempty n-by-K probabilities matching targets/classes")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Probabilities must be finite and in [0, 1]")
    if not np.allclose(p.sum(axis=1), 1, atol=1e-8, rtol=0):
        raise ValueError("Probability rows must sum to one; never silently renormalize")
    result = _report()
    result.update(
        task="binary_classification" if k == 2 else "multiclass_classification",
        n_observations=len(y),
        class_order=classes,
    )
    onehot = np.eye(k)[y]
    _put(result, "log_loss", sk.log_loss(y, p, labels=np.arange(k)))
    _put(result, "brier_multiclass_sum", np.square(p - onehot).sum(axis=1).mean())
    result["brier_multiclass_range"] = [0, 2]
    all_present = len(np.unique(y)) == k
    ranking_reason = "ROC/PR summaries require positive and negative observations for every class"
    per_class_ranking = []
    for i in range(k):
        binary = (y == i).astype(int)
        valid = 0 < binary.sum() < len(binary)
        per_class_ranking.append(
            {
                "class_index": i,
                "roc_auc": float(sk.roc_auc_score(binary, p[:, i])) if valid else None,
                "average_precision": float(sk.average_precision_score(binary, p[:, i]))
                if valid
                else None,
                "brier": float(np.square(p[:, i] - binary).mean()),
            }
        )
    result["per_class_probability_metrics"] = per_class_ranking
    if k == 2:
        if positive_class is None or canonical_json(scalar(positive_class)) not in lookup:
            raise ValueError("An explicit positive_class from classes is required")
        pos = lookup[canonical_json(scalar(positive_class))]
        positive = (y == pos).astype(int)
        score = p[:, pos]
        result["positive_class"] = classes[pos]
        _put(result, "brier", np.square(score - positive).mean())
        result["brier_range"] = [0, 1]
        _put(
            result,
            "roc_auc",
            sk.roc_auc_score(positive, score) if all_present else None,
            ranking_reason,
        )
        _put(
            result,
            "average_precision",
            sk.average_precision_score(positive, score) if all_present else None,
            ranking_reason,
        )
        if all_present:
            precision, recall, _ = sk.precision_recall_curve(positive, score)
            pr_auc = sk.auc(recall, precision)
        else:
            pr_auc = None
        _put(result, "pr_auc_trapezoidal", pr_auc, ranking_reason)
        thresholds = {"default_0_5": 0.5}
        if selected_threshold is not None:
            if not np.isfinite(selected_threshold) or not 0 <= selected_threshold <= 1:
                raise ValueError("selected_threshold must be finite and in [0, 1]")
            thresholds["validation_selected"] = float(selected_threshold)
        result["decisions"] = {}
        for name, threshold in thresholds.items():
            predicted = np.where(score >= threshold, pos, 1 - pos)
            result["decisions"][name] = {"threshold": threshold, **_decisions(y, predicted, k, pos)}
        calibration_probability, calibration_outcome = score, positive
        result["calibration_definition"] = "positive_class_equal_width"
    else:
        if positive_class is not None or selected_threshold is not None:
            raise ValueError("Binary threshold/positive class cannot be applied to multiclass")
        for strategy in ("ovr", "ovo"):
            for average in ("macro", "weighted"):
                _put(
                    result,
                    f"roc_auc_{strategy}_{average}",
                    sk.roc_auc_score(
                        y, p, multi_class=strategy, average=average, labels=np.arange(k)
                    )
                    if all_present
                    else None,
                    ranking_reason,
                )
        for average in ("macro", "weighted", "micro"):
            _put(
                result,
                f"average_precision_{average}",
                sk.average_precision_score(onehot, p, average=average) if all_present else None,
                ranking_reason,
            )
        predicted = p.argmax(axis=1)  # Ties resolve by the frozen class order.
        result["decisions"] = {"argmax": _decisions(y, predicted, k)}
        for top_k in (2, 3, 5):
            ranked = np.argsort(-p, axis=1, kind="stable")[:, :top_k]
            _put(
                result,
                f"top_{top_k}_accuracy",
                (ranked == y[:, None]).any(axis=1).mean() if top_k < k else None,
                "k >= number of classes gives a trivial score",
            )
        calibration_probability = p.max(axis=1)
        calibration_outcome = (predicted == y).astype(int)
        result["calibration_definition"] = "top_label_equal_width"
    bins, ece, mce = _reliability(calibration_probability, calibration_outcome, calibration_bins)
    result["reliability_bins"] = bins
    _put(result, "ece", ece)
    _put(result, "maximum_calibration_error", mce)
    return result


def regression_metrics(
    y_true, prediction, *, positive_ratio_scale=False, nonnegative_ratio_scale=False, quantiles=None
):
    """Point metrics on the declared target scale; optional ordered quantile predictions.

    Percentage/log metrics require an explicit meaningful ratio scale, not just positive
    numbers. Never exponentiate a log-transformed target or clip negative predictions here.
    """
    y, pred = np.asarray(y_true, dtype=float), np.asarray(prediction, dtype=float)
    if y.ndim != 1 or not len(y) or y.shape != pred.shape:
        raise ValueError("Expected matching nonempty one-dimensional targets/predictions")
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Targets/predictions must be finite")
    result = _report()
    result.update(task="regression", n_observations=len(y))
    error = pred - y
    for name, value in [
        ("mse", np.square(error).mean()),
        ("rmse", np.sqrt(np.square(error).mean())),
        ("mae", np.abs(error).mean()),
        ("median_absolute_error", np.median(np.abs(error))),
        ("max_absolute_error", np.abs(error).max()),
        ("mean_error", error.mean()),
    ]:
        _put(result, name, value)
    nonconstant = len(y) >= 2 and np.ptp(y) > 0
    _put(
        result,
        "r2",
        sk.r2_score(y, pred, force_finite=False) if nonconstant else None,
        "R2 requires >=2 observations and a nonconstant target",
    )
    _put(
        result,
        "explained_variance",
        sk.explained_variance_score(y, pred, force_finite=False) if nonconstant else None,
        "Explained variance requires nonconstant targets",
    )
    correlation_valid = nonconstant and np.ptp(pred) > 0
    _put(
        result,
        "pearson_r",
        np.corrcoef(y, pred)[0, 1] if correlation_valid else None,
        "Correlation requires >=2 observations and nonconstant targets and predictions",
    )
    _put(
        result,
        "spearman_rho",
        np.corrcoef(pd.Series(y).rank(), pd.Series(pred).rank())[0, 1]
        if correlation_valid
        else None,
        "Rank correlation requires nonconstant inputs",
    )
    _put(
        result,
        "mape_pct",
        np.mean(np.abs(error / y)) * 100 if positive_ratio_scale and (y > 0).all() else None,
        "MAPE requires a declared ratio scale and strictly positive targets",
    )
    _put(
        result,
        "rmsle",
        np.sqrt(np.mean(np.square(np.log1p(pred) - np.log1p(y))))
        if nonnegative_ratio_scale and (y >= 0).all() and (pred >= 0).all()
        else None,
        "RMSLE requires a declared ratio scale and nonnegative targets and predictions",
    )
    result["quantile_metrics"] = {}
    if quantiles:
        qlevels = sorted(quantiles)
        if any(not 0 < q < 1 for q in qlevels):
            raise ValueError("Quantile levels must lie strictly between zero and one")
        values = np.column_stack([quantiles[q] for q in qlevels])
        if values.shape != (len(y), len(qlevels)) or not np.isfinite(values).all():
            raise ValueError("Quantiles must match targets and be finite")
        if (np.diff(values, axis=1) < 0).any():
            raise ValueError("Quantile predictions cross; do not silently reorder them")
        for i, q in enumerate(qlevels):
            result["quantile_metrics"][str(q)] = {
                "pinball_loss": float(sk.mean_pinball_loss(y, values[:, i], alpha=q)),
                "observed_fraction_below": float((y <= values[:, i]).mean()),
            }
        result["interval_metrics"] = {}
        for q in qlevels:
            upper = next((v for v in qlevels if abs(v - (1 - q)) < 1e-10), None)
            if q < 0.5 and upper is not None:
                lower_values, upper_values = (
                    values[:, qlevels.index(q)],
                    values[:, qlevels.index(upper)],
                )
                width = upper_values - lower_values
                score = (
                    width
                    + np.maximum(lower_values - y, 0) / q
                    + np.maximum(y - upper_values, 0) / q
                )
                result["interval_metrics"][str(1 - 2 * q)] = {
                    "coverage": float(((y >= lower_values) & (y <= upper_values)).mean()),
                    "mean_width": float(width.mean()),
                    "interval_score": float(score.mean()),
                }
    return result
