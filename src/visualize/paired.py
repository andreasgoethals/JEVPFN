"""Paired representation effects and training-only PCA diagnostics; no model ranking."""

import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.design import pca_grid
from src.visualize import layout, style
from src.visualize.research import ENCODER_LABELS

# Endpoints fixed before comparative outcomes; fold spread stays descriptive.
METRIC_VIEWS = {
    "binary_classification": ("roc_auc", 1, "AUC: augmented − reference"),
    "multiclass_classification": ("roc_auc_ovr_macro", 1, "Macro OvR AUC: augmented − reference"),
    "regression": ("r2", 1, "R²: augmented − reference"),
}
MODES = {"non_text_plus_joint": "Joint", "non_text_plus_per_column": "Per-column",
         "non_text_plus_both": "Both"}
MODE_KEYS = {"non_text_plus_joint": "joint", "non_text_plus_per_column": "per_column",
             "non_text_plus_both": "both"}
REFERENCE_LABELS = {
    "non_text": "non-text + text presence",
    "non_text_plus_presence": "non-text + missing-text flags",
    "frozen_e5": "frozen embedding alone",
    "same_decision_features_without_e5": "same decision features alone",
    "original_text_without_decision_features": "original text alone",
}


def _dataset_effects(group, keys, task, direction):
    aggregations = {"mean": ("difference", "mean"), "count": ("difference", "count"),
                    "folds": ("outer_fold", "nunique")}
    rmse = task == "regression" and "metric" in group and group.metric.eq("rmse").all()
    if rmse:
        aggregations.update(reference_mean=("reference_value", "mean"),
                            reference_count=("reference_value", "count"))
    result = group.groupby(keys, dropna=False).agg(**aggregations).reset_index()
    result = result[result["count"].eq(5) & result.folds.eq(5)].copy()
    result["effect"] = result["mean"] * direction
    if rmse:
        # Target units differ across datasets. Compare the percentage reduction in
        # mean fold RMSE, never an average of dollars, salary units and review scores.
        result = result[result.reference_count.eq(5) & result.reference_mean.gt(0)
                        & np.isfinite(result.reference_mean)].copy()
        result["effect"] *= 100 / result.reference_mean
    return result


def _effect_grid(ax, values, wins, counts, limit, cmap):
    ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
    for (i, j), n in np.ndenumerate(counts):
        if n:
            dark = abs(values[i, j]) > 0.6 * limit
            ax.text(j, i, f"{wins[i, j]:.0f}/{n:.0f}", ha="center", va="center", fontsize=7,
                    color=style.BACKGROUND_COLOR if dark else style.TEXT_COLOR)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def embedding_effect_figures(paired):
    if paired.empty:
        return
    recipes = pca_grid()
    current = np.logical_or.reduce([
        paired.pca_scope.eq(scope) & paired.pca_components.eq(count) for scope, count in recipes
    ])
    for task, (metric, direction, unit) in METRIC_VIEWS.items():
        selected = paired[current & paired.phase.eq("B2") & paired.task_type.eq(task)
                          & paired.metric.eq(metric)]
        for (model, reference), group in selected.groupby(["model_id", "reference"], sort=False):
            keys = ["dataset_id", "split_id", "checkpoint_id", "encoder", "pca_scope",
                    "pca_components", "arm_id"]
            datasets = _dataset_effects(group, keys, task, direction)
            if datasets.empty:
                continue
            matches = datasets.groupby(["dataset_id", "split_id", "checkpoint_id", "encoder"]).filter(
                lambda frame: len(frame) == len(recipes)
                and set(zip(frame.pca_scope, frame.pca_components)) == set(recipes)
            )
            if matches.empty:
                continue
            if matches.duplicated(["dataset_id", "encoder", "pca_scope", "pca_components"]).any():
                raise ValueError("Multiple splits or checkpoints match the PCA comparison")
            fig, axes = plt.subplots(1, len(recipes), figsize=style.DASHBOARD_SIZE, sharey=True)
            for ax, (scope, count) in zip(axes, recipes):
                selected_recipe = matches[
                    matches.pca_scope.eq(scope) & matches.pca_components.eq(count)
                ]
                for encoder, part in selected_recipe.groupby("encoder", sort=False):
                    position = list(ENCODER_LABELS).index(encoder)
                    color = style.EMBEDDING_COLORS[encoder]
                    values = part.effect
                    ax.scatter(position + np.linspace(-0.1, 0.1, len(values)), values,
                               color=color, alpha=0.4, s=12)
                    ax.scatter(position, values.mean(), color=color, marker="D", s=25)
                ax.axhline(0, color=style.GRID_COLOR)
                ax.set_ylabel(unit, fontsize=7)
                ax.set_xticks(range(3), ["Small E5", "Large E5", "Qwen"], fontsize=7)
                ax.set_title(f"{scope.replace('_', '-').capitalize()} PCA{count}", fontsize=8)
            fig.suptitle(f"{style.PREDICTOR_LABELS[model]} · {layout.TASK_LABELS[task]} · frozen embeddings"
                         f" · reference: {REFERENCE_LABELS.get(reference, reference)}", fontsize=9)
            yield fig, f"embedding_effect_{model}_{task}_{reference}", (
                "Paired frozen-embedding gains over the shared structured reference. For each "
                "encoder, both panels use the same datasets and splits with five defined "
                "paired folds for both recipes. Small points are dataset effects and diamonds "
                "are their means. Positive values indicate gains in AUC or R². Separate PCA32 "
                "uses 32 coordinates per text field and joint PCA64 uses 64 total. These "
                "recipes have equal width for datasets with two text fields."
            )
