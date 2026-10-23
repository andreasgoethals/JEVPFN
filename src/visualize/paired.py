"""Paired representation effects and training-only PCA diagnostics; no model ranking."""

import matplotlib.pyplot as plt
import numpy as np

from src.visualize import style
from src.visualize.decision_comparison import LABELS
from src.visualize.research import ENCODER_LABELS

# Descriptive views of recorded metrics, not approval of a confirmatory endpoint.
METRIC_VIEWS = {
    "binary_classification": ("roc_auc", 1, "AUC: augmented − reference"),
    "multiclass_classification": ("log_loss", -1, "Log loss: reference − augmented"),
    "regression": ("rmse", -1, "RMSE reduction (%)"),
}
MODES = {"non_text_plus_joint": "Joint", "non_text_plus_per_column": "Per-column",
         "non_text_plus_both": "Both"}


def _dataset_effects(group, keys, task, direction):
    aggregations = {"mean": ("difference", "mean"), "count": ("difference", "count"),
                    "folds": ("outer_fold", "nunique")}
    if task == "regression":
        aggregations.update(reference_mean=("reference_value", "mean"),
                            reference_count=("reference_value", "count"))
    result = group.groupby(keys, dropna=False).agg(**aggregations).reset_index()
    result = result[result["count"].eq(5) & result.folds.eq(5)].copy()
    result["effect"] = result["mean"] * direction
    if task == "regression":
        # Target units differ across datasets. Compare the percentage reduction in
        # mean fold RMSE, never an average of dollars, salary units and review scores.
        result = result[result.reference_count.eq(5) & result.reference_mean.gt(0)
                        & np.isfinite(result.reference_mean)].copy()
        result["effect"] *= 100 / result.reference_mean
    return result


def paired_effect_figures(paired):
    """Show dataset points and their mean; correlated folds are never error bars."""
    if paired.empty:
        return
    for task, (metric, direction, unit) in METRIC_VIEWS.items():
        sub = paired[paired.task_type.eq(task) & paired.metric.eq(metric)
                     & paired.phase.eq("experiment_1") & paired.feature_set.isin(MODES)
                     & paired.provider.isin(LABELS)]
        if sub.empty:
            continue
        for (model, reference), group in sub.groupby(["model_id", "reference"], sort=False):
            # Undefined metrics remain in tables; require all five paired defined folds
            # before a dataset contributes to the corresponding plotted contrast.
            key = ["provider", "feature_set", "dataset_id", "split_id", "arm_id"]
            means = _dataset_effects(group, key, task, direction)
            if means.empty:
                continue
            providers = sorted(means.provider.unique(), key=lambda p: list(LABELS).index(p))
            fig, axes = plt.subplots(1, 3, figsize=style.DASHBOARD_SIZE, sharey=True)
            bound = max(float(means.effect.abs().max()) * 1.15, 0.001)
            for ax, (mode, label) in zip(axes, MODES.items()):
                ax.axvline(0, color=style.GRID_COLOR)
                for pos, provider in enumerate(providers):
                    values = means.loc[means.provider.eq(provider) & means.feature_set.eq(mode), "effect"]
                    if values.empty:
                        continue
                    offsets = np.linspace(-0.15, 0.15, len(values))
                    ax.scatter(values, pos + offsets, color=style.MEDIAN_COLOR, alpha=0.35, s=10)
                    ax.scatter(values.mean(), pos, color=style.P95_COLOR, marker="D", s=25)
                ax.set_title(label)
                ax.set_xlabel(unit, fontsize=7)
                ax.set_xlim(-bound, bound)
                ax.set_yticks(range(len(providers)), [LABELS[p] for p in providers], fontsize=7)
                ax.grid(axis="x", color=style.GRID_COLOR, alpha=0.4)
                ax.tick_params(axis="x", labelsize=7)
            axes[0].invert_yaxis()
            n = means.dataset_id.nunique()
            fig.suptitle(f"{model} · {task.split('_')[0]} · {n} available datasets\nReference: {reference}", fontsize=9)
            yield fig, f"paired_{model}_{task}_{reference}", (
                "Paired representation effects within the same predictor and named reference. "
                "Small points are per-dataset averages of five defined paired outer-fold differences; "
                "diamonds are unweighted averages across the available datasets, separately per "
                "extractor/mode. Positive values indicate improvement. Regression shows percentage "
                "reduction in mean fold RMSE relative to the same dataset/reference; zero or "
                "undefined reference RMSE is excluded. Other tasks retain their named metric units. "
                "Coverage may differ; exact dataset identities/counts remain in the table. "
                "Fold spread is not a confidence interval. These descriptive metric views do not "
                "select a confirmatory endpoint or establish a cross-model leaderboard."
            )


def embedding_effect_figures(paired):
    if paired.empty:
        return
    for task, (metric, direction, unit) in METRIC_VIEWS.items():
        selected = paired[paired.phase.eq("experiment_2") & paired.task_type.eq(task)
                          & paired.metric.eq(metric)]
        for (model, reference), group in selected.groupby(["model_id", "reference"], sort=False):
            keys = ["dataset_id", "split_id", "encoder", "pca_scope", "pca_components", "arm_id"]
            datasets = _dataset_effects(group, keys, task, direction)
            if datasets.empty:
                continue
            fig, axes = plt.subplots(1, 2, figsize=style.DASHBOARD_SIZE, sharey=True,
                                      gridspec_kw={"width_ratios": [2, 1]})
            for encoder, part in datasets.groupby("encoder", sort=False):
                color = style.EMBEDDING_COLORS[encoder]
                joint = part[part.pca_scope.eq("joint")].groupby("pca_components").effect.mean()
                if not joint.empty:
                    axes[0].plot(joint.index, joint, color=color, marker="o",
                                 label=ENCODER_LABELS[encoder], markersize=4)
                separate = part[part.pca_scope.eq("per_column")].effect
                if not separate.empty:
                    position = list(ENCODER_LABELS).index(encoder)
                    axes[1].scatter(position + np.linspace(-0.1, 0.1, len(separate)), separate,
                                    color=color, alpha=0.4, s=12)
                    axes[1].scatter(position, separate.mean(), color=color, marker="D", s=25)
            for ax in axes:
                ax.axhline(0, color=style.GRID_COLOR)
                ax.set_ylabel(unit, fontsize=7)
            axes[0].set_xscale("log", base=2)
            axes[0].set_xticks([8, 16, 32, 64, 128, 256], [8, 16, 32, 64, 128, 256])
            axes[0].set_xlabel("Joint PCA: total dimensions")
            axes[0].legend(fontsize=7)
            axes[1].set_xticks(range(3), ["Small\nE5", "Large\nE5", "Qwen"], fontsize=7)
            axes[1].set_title("Per-column PCA32", fontsize=8)
            fig.suptitle(f"{model} · {task.split('_')[0]} · embedding effects\nReference: {reference}", fontsize=9)
            yield fig, f"embedding_effect_{model}_{task}_{reference}", (
                "Paired frozen-embedding effects within a fixed predictor. Each joint curve "
                "averages dataset-level five-fold differences at each requested width; per-column "
                "points are individual dataset differences and diamonds their means. Exact counts "
                "and identities remain in the tables; missing/unsupported arms are not imputed. "
                "Positive values indicate improvement. Regression is percentage reduction in mean "
                "fold RMSE against a positive same-dataset reference; other tasks use their named "
                "metric units. Per-column32 "
                "has 32 coordinates per text field, not 32 total; the panels are not equal-budget "
                "grouping comparisons. PCA is fitted within the corresponding training partition."
            )


def pca_variance_figures(table):
    if table.empty:
        return
    for encoder, part in table.groupby("encoder", sort=False):
        fig, axes = plt.subplots(1, 2, figsize=style.DASHBOARD_SIZE,
                                  gridspec_kw={"width_ratios": [2, 1]})
        joint = part[part.pca_scope.eq("joint")]
        for _, group in joint.groupby("dataset_id"):
            mean = group.groupby("requested_components").training_explained_variance_fraction.mean()
            axes[0].plot(mean.index, mean, color=style.MEDIAN_COLOR, alpha=0.45, marker="o", markersize=3)
        separate = part[part.pca_scope.eq("per_column")]
        if not separate.empty:
            means = separate.groupby(["dataset_id", "text_group"]).training_explained_variance_fraction.mean()
            axes[1].scatter(np.linspace(-0.2, 0.2, len(means)), means, color=style.P95_COLOR, s=15, alpha=0.6)
        axes[0].set_xscale("log", base=2)
        axes[0].set_xticks([8, 16, 32, 64, 128, 256], [8, 16, 32, 64, 128, 256])
        axes[0].set(xlabel="Joint PCA: requested total dimensions", ylabel="Training explained-variance fraction")
        axes[1].set(xticks=[0], xticklabels=["Per-column PCA32"], xlim=(-0.4, 0.4))
        for ax in axes:
            ax.set_ylim(0, 1.02)
        if joint.empty:
            axes[0].text(0.5, 0.5, "Joint diagnostics pending", transform=axes[0].transAxes, ha="center")
        if separate.empty:
            axes[1].text(0.5, 0.5, "Per-column\ndiagnostics pending", transform=axes[1].transAxes, ha="center")
        fig.suptitle(f"{ENCODER_LABELS.get(encoder, encoder)} · observed training-only compression")
        yield fig, f"pca_training_variance_{encoder}", (
            "Training-only cumulative PCA explained-variance fractions, averaged over available "
            "outer-training folds. Each joint curve is one dataset; each per-column point is one "
            "dataset/text column. Reuse across predictors and metric records is counted once. "
            "Requested and fitted rank remain in the table. Variance is not task information "
            "or predictive utility, and fractions across encoders do not define a common "
            "information budget. No validation/test rows fit PCA or select dimensions."
        )
