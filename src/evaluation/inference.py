"""Dataset-level paired summaries, with optional explicitly preregistered inference.

CV folds are not independent experimental units. Never turn their SD into an SE.
Raw metrics are summarized separately by task, predictor, recipe and reference.
"""

import numpy as np
import pandas as pd

GROUP = [
    "task_type",
    "model_id",
    "checkpoint_id",
    "feature_set",
    "comparison_id",
    "reference",
    "metric",
]
DIMENSIONLESS = {
    "r2",
    "explained_variance",
    "pearson_r",
    "spearman_rho",
    "roc_auc",
    "average_precision",
    "log_loss",
    "brier",
    "brier_multiclass_sum",
    "accuracy",
    "balanced_accuracy",
    "f1",
    "f1_macro",
    "f1_micro",
    "f1_weighted",
    "precision",
    "recall",
    "mcc",
    "cohen_kappa",
    "roc_auc_ovr_macro",
    "roc_auc_ovr_weighted",
    "roc_auc_ovo_macro",
    "roc_auc_ovo_weighted",
}


def dataset_differences(paired):
    if paired.empty:
        return pd.DataFrame(columns=GROUP + ["dataset_id", "mean_difference", "defined_folds"])
    keys = GROUP + ["dataset_id", "split_id"]
    if paired.duplicated(keys + ["outer_fold"]).any():
        raise ValueError("Duplicate fold comparison; choose one campaign")
    result = (
        paired.groupby(keys, dropna=False)
        .difference.agg(
            mean_difference="mean",
            descriptive_fold_sd="std",
            defined_folds="count",
        )
        .reset_index()
    )
    # Keep incomplete/undefined cases visible; they cannot enter inferential comparisons.
    result["complete"] = result.defined_folds.eq(5)
    return result


def across_datasets(datasets):
    if datasets.empty:
        return pd.DataFrame(
            columns=GROUP
            + ["datasets", "complete_datasets", "mean_difference", "median_difference"]
        )
    records = []
    for key, group in datasets.groupby(GROUP, dropna=False):
        values = group.loc[group.complete, "mean_difference"]
        comparable = key[-1].split(".")[-1] in DIMENSIONLESS
        records.append(
            dict(zip(GROUP, key))
            | {
                "datasets": len(group),
                "complete_datasets": len(values),
                "mean_difference": float(values.mean()) if len(values) and comparable else None,
                "median_difference": float(values.median()) if len(values) and comparable else None,
                "positive_deltas": int((values > 0).sum()),
                "zero_deltas": int((values == 0).sum()),
                "negative_deltas": int((values < 0).sum()),
                "interpretation": "Descriptive; positive is better only for higher-is-better metrics. Raw magnitudes with dataset-specific units are not pooled.",
            }
        )
    return pd.DataFrame(records)


def confirmatory_tests(datasets, plan):
    """One complete family, declared before outcomes; Holm includes every planned contrast.

    Each contrast supplies GROUP identifiers, dataset IDs and direction (+1 or -1).
    Missing/undefined data blocks inference, rather than shrinking the testing family.
    """
    from scipy.stats import wilcoxon

    if (
        plan.get("approved_before_results") is not True
        or not plan.get("contrasts")
        or not plan.get("reviewed_by")
    ):
        raise ValueError("An explicit analysis plan approved before results is required")
    results, identities = [], set()
    for contrast in plan["contrasts"]:
        keys = contrast["match"]
        if set(keys) != set(GROUP) or contrast["direction"] not in {-1, 1}:
            raise ValueError("Specify one task, metric, predictor, recipe, reference and direction")
        if keys["metric"].split(".")[-1] not in DIMENSIONLESS:
            raise ValueError("Use a comparable dimensionless metric, not raw target-unit errors")
        identity = tuple(keys[k] for k in GROUP)
        if identity in identities:
            raise ValueError("Duplicate planned statistical contrast")
        identities.add(identity)
        chosen = datasets.loc[np.logical_and.reduce([datasets[k].eq(v) for k, v in keys.items()])]
        expected = contrast["datasets"]
        if (
            len(set(expected)) != len(expected)
            or len(expected) < 2
            or chosen.dataset_id.duplicated().any()
            or set(chosen.dataset_id) != set(expected)
            or not chosen.complete.all()
        ):
            raise ValueError("Every planned dataset requires exactly one complete paired result")
        differences = chosen.mean_difference.to_numpy() * contrast["direction"]
        if not np.isfinite(differences).all():
            raise ValueError("Nonfinite dataset difference")
        # Two-sided test avoids interpreting a harmful significant effect as an improvement.
        p = (
            1.0
            if np.all(differences == 0)
            else float(wilcoxon(differences, alternative="two-sided", zero_method="wilcox").pvalue)
        )
        results.append(
            {
                **keys,
                "datasets": len(chosen),
                "median_oriented_delta": float(np.median(differences)),
                "p_raw": p,
            }
        )
    order = np.argsort([r["p_raw"] for r in results])
    adjusted = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, max(adjusted, results[index]["p_raw"] * (len(results) - rank)))
        results[index]["p_holm"] = adjusted
    return pd.DataFrame(results)
