"""Preflight widths; distinguish a selected recipe's limit from hardware capacity."""

from src.evaluation.features import decision_feature_set, uses_embeddings


def feature_width(metadata, arm):
    texts = len(metadata.text_columns)
    base = len(metadata.non_text_columns)
    name = decision_feature_set(arm["feature_set"])
    if arm["feature_set"].startswith("original_features"):
        base += texts
    if name == "non_text_plus_text_statistics":
        return base + texts * 3 + 3
    if name == "non_text" or arm["feature_set"] == "original_features":
        width = base
    else:
        width = base + texts  # shared missing-text flags, only once
        per_input = (
            len(metadata.classes) if metadata.task_type == "multiclass_classification" else 1
        )
        if (
            arm.get("view") in {"probabilities", "probabilities_plus_confidence"}
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
        providers = arm.get("providers") or ([arm["provider"]] if arm.get("provider") else [])
        width += len(providers) * groups * per_input
    if uses_embeddings(arm["feature_set"]):
        if name == "non_text":
            width += texts
        width += arm["pca_components"] * (texts if arm["pca_scope"] == "per_column" else 1)
    return max(width, 1)


def width_status(metadata, arm):
    width = feature_width(metadata, arm)
    limit = 500 if arm["model"] == "kumo_tabular" else None
    return {
        "predictor_columns": width,
        "recipe_column_limit": limit,
        "width_status": "unsupported_width"
        if limit is not None and width > limit
        else "within_known_width_limit",
    }


def pca_width_table():
    """Catalog audit counts, including non-text columns and one missing flag per text."""
    from types import SimpleNamespace

    import pandas as pd

    from src.data.loaders import load_catalog
    from src.data.selection import selected_entries
    from src.evaluation.design import pca_grid
    from src.utils.config import load_yaml

    cfg, semantics = load_yaml("experiment_1/main"), load_yaml("dataset_semantics")["datasets"]
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
        arm.update(
            feature_set="non_text_plus_frozen_e5_and_both",
            pca_scope="per_column",
            pca_components=32,
            provider="any_single_provider",
        )
        per = feature_width(meta, arm)
        arm.update(pca_scope="joint", pca_components=256)
        row["max_with_one_provider_both"] = max(per, feature_width(meta, arm))
        # Kumo may add a count column for every high-cardinality categorical column.
        row["conservative_kumo_upper_bound"] = row["max_with_one_provider_both"] + len(
            meta.non_text_columns
        )
        rows.append(row)
    return pd.DataFrame(rows)
