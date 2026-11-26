"""Preflight widths; distinguish a selected recipe's limit from hardware capacity.

Kumo's default recipe keeps 500 columns per estimator. Since 23 November 2026 (owner
approval) wider Kumo arms widen that recipe limit to their full width instead of being
excluded (`KumoBackend`); B200 memory and time are a B0 full-fold profile check.
"""

from src.evaluation.features import (
    ADAPTED,
    HIDDEN_FEATURE_SETS,
    PER_COLUMN_WITHOUT_ONE,
    SUPERVISED_PROBE,
    decision_feature_set,
    uses_embeddings,
)
from src.evaluation.models import KUMO_RECIPE_COLUMNS


def feature_width(metadata, arm):
    texts = len(metadata.text_columns)
    base = len(metadata.non_text_columns)
    name = decision_feature_set(arm["feature_set"])
    if arm["feature_set"].startswith("original_features"):
        base += texts
    if arm["feature_set"] == "original_features":
        width = base
    elif name == "non_text":
        width = base + texts
    else:
        width = base + texts  # shared missing-text flags, only once
        per_input = (
            len(metadata.classes) if metadata.task_type == "multiclass_classification" else 1
        )
        if (
            arm.get("view", "main") != "mean"
            and metadata.task_type == "regression"
        ):
            per_input = 9
        if arm.get("view") == "probabilities_plus_confidence":
            per_input += 3 if metadata.task_type == "regression" else 2
        groups = (
            texts
            if name.endswith("per_column")
            else 1
            if name.endswith("joint")
            else texts + (texts > 1)
            if name.endswith("both")
            else 0
        )
        if arm["feature_set"] == PER_COLUMN_WITHOUT_ONE:
            groups -= 1
        providers = arm.get("providers") or ([arm["provider"]] if arm.get("provider") else [])
        width += len(providers) * groups * per_input
    if arm["feature_set"] == SUPERVISED_PROBE:
        per_input = {"binary_classification": 1, "regression": 9}.get(
            metadata.task_type, len(metadata.classes)
        )
        width += per_input * (texts if arm["probe_view"] == "per_column" else 1)
    elif arm["feature_set"] in HIDDEN_FEATURE_SETS:  # Group C1: PCA of one hidden-state group
        width += arm["pca_components"] * (texts if arm["hidden_view"] == "per_column" else 1)
    elif arm["feature_set"] == ADAPTED:  # Group C2: B1's joint width
        width += {"binary_classification": 1, "regression": 9}.get(
            metadata.task_type, len(metadata.classes))
    elif uses_embeddings(arm["feature_set"]):
        width += arm["pca_components"] * (texts if arm["pca_scope"] == "per_column" else 1)
    return max(width, 1)


def width_status(metadata, arm):
    """No selected recipe now silently drops columns, so no arm is `unsupported_width`.

    Kumo arms within the default keep their historical fields (and plan identity); wider
    Kumo arms record the per-estimator limit raised to their planned width. The fit
    recomputes the limit after Kumo's category-count expansion.
    """
    width = feature_width(metadata, arm)
    status = {
        "predictor_columns": width,
        "recipe_column_limit": None,
        "width_status": "within_known_width_limit",
    }
    if arm["model"] == "kumo_tabular":
        status["recipe_column_limit"] = max(KUMO_RECIPE_COLUMNS, width)
        if width > KUMO_RECIPE_COLUMNS:
            status["kumo_max_columns_override"] = width
    return status


def pca_width_table():
    """Catalog audit counts, including non-text columns and one missing flag per text."""
    from types import SimpleNamespace

    import pandas as pd

    from src.data.loaders import load_catalog
    from src.data.selection import selected_entries
    from src.evaluation.design import pca_grid
    from src.utils.config import load_yaml

    cfg, semantics = load_yaml("B1"), load_yaml("dataset_semantics")["datasets"]
    rows = []
    for entry in selected_entries(load_catalog(), cfg):
        stats, task = entry["paper_summary"], semantics[entry["id"]]
        meta = SimpleNamespace(
            text_columns=range(stats["text_columns"]),
            non_text_columns=range(stats["non_text_columns"]),
            classes=task.get("classes", []),
            task_type=task["task_type"],
        )
        row = {
            "dataset": entry["name"],
            "text_columns": len(meta.text_columns),
            "non_text_columns": len(meta.non_text_columns),
        }
        for scope, k in pca_grid():
            arm = dict(feature_set="non_text_plus_frozen_e5", pca_scope=scope, pca_components=k)
            row[f"{scope}_{k}"] = feature_width(meta, arm)
        row["max_embedding_columns"] = max(row[f"{scope}_{k}"] for scope, k in pca_grid())
        # Kumo may add a count column for every high-cardinality categorical column.
        row["conservative_kumo_upper_bound"] = row["max_embedding_columns"] + len(meta.non_text_columns)
        rows.append(row)
    return pd.DataFrame(rows)
