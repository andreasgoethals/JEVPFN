"""Prespecified MulTaBench-compatible headline metrics; all other scores stay saved."""

HEADLINES = {
    "binary_classification": "roc_auc",
    "multiclass_classification": "roc_auc_ovr_macro",
    "regression": "r2",
}
LABELS = {"roc_auc": "ROC-AUC", "roc_auc_ovr_macro": "Macro OvR ROC-AUC", "r2": "R²"}


def headline_metric(task):
    return HEADLINES[task]
