"""Paired representation effects and training-only PCA diagnostics; no model ranking."""

import matplotlib.pyplot as plt
import numpy as np

from src.visualize import layout, style
from src.visualize.decision_comparison import LABELS
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
    "non_text": "non-text",
    "non_text_plus_presence": "non-text + missing-text flags",
    "frozen_e5": "frozen embedding alone",
    "same_decision_features_without_e5": "same decision features alone",
    "original_text_without_decision_features": "original text alone",
}
JOINT_WIDTHS = (8, 16, 32, 64, 128, 256)


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
            ax.text(j, i, f"{wins[i, j]:.0f}/{n:.0f}", ha="center", va="center", fontsize=5.8,
                    color=style.BACKGROUND_COLOR if dark else style.TEXT_COLOR)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def effect_matrix_figures(paired, phase="experiment_1"):
    """Headline view: every recipe × predictor, against the prespecified non-text reference.

    Colour is the mean over datasets of the five-fold mean paired difference; the text gives
    datasets with a positive effect over datasets with five defined folds. No recipe is ranked.
    """
    if paired.empty:
        return
    cmap = plt.get_cmap(style.EFFECT_CMAP).with_extremes(bad=style.FAINT_COLOR)
    models = list(style.MAIN_PREDICTORS)
    short = {m: style.PREDICTOR_LABELS[m].split(" ")[0].split("/")[0] for m in models}
    for task, (metric, direction, unit) in METRIC_VIEWS.items():
        sub = paired[paired.phase.eq(phase) & paired.task_type.eq(task) & paired.metric.eq(metric)
                     & paired.reference.eq("non_text") & paired.model_id.isin(models)]
        if phase == "experiment_1":
            sub = sub[sub.feature_set.isin(MODES) & sub.provider.isin(LABELS)]
            rows = [p for p in LABELS if p in set(sub.provider)]
            row_keys, panels = ["provider"], list(MODES)
            row_labels = [LABELS[p] for p in rows]
        else:
            sub = sub[sub.encoder.notna()]
            recipes = [("per_column", 32), *[("joint", w) for w in sorted(JOINT_WIDTHS, reverse=True)]]
            rows = [(e, s, c) for e in ENCODER_LABELS for s, c in recipes]
            row_keys, panels = ["encoder", "pca_scope", "pca_components"], [None]
            row_labels = [f"{ENCODER_LABELS[e]} · {'P' if s == 'per_column' else 'J'}{c}"
                          for e, s, c in rows]
        if sub.empty:
            continue
        keys = row_keys + ["feature_set", "model_id", "dataset_id", "split_id", "arm_id"]
        effects = _dataset_effects(sub, keys, task, direction)
        if effects.empty:
            continue
        cells = effects.groupby(row_keys + ["feature_set", "model_id"], dropna=False).agg(
            effect=("effect", "mean"), wins=("effect", lambda v: (v > 0).sum()),
            n=("effect", "size")).reset_index()
        limit = max(float(np.nanpercentile(np.abs(cells.effect), 95)), 1e-4)
        fig, axes = plt.subplots(1, len(panels), figsize=(style.WIDTH_FULL, min(
            style.MAX_HEIGHT, 1.1 + 0.2 * len(rows))), sharey=True, squeeze=False)
        for ax, panel in zip(axes[0], panels):
            part = cells if panel is None else cells[cells.feature_set.eq(panel)]
            shape = (len(rows), len(models))
            values, wins, counts = np.full(shape, np.nan), np.zeros(shape), np.zeros(shape)
            for record in part.itertuples():
                key = tuple(getattr(record, k) for k in row_keys)
                key = key[0] if len(key) == 1 else (key[0], key[1], int(key[2]))
                if key in rows and record.model_id in models:
                    i, j = rows.index(key), models.index(record.model_id)
                    values[i, j], wins[i, j], counts[i, j] = record.effect, record.wins, record.n
            _effect_grid(ax, values, wins, counts, limit, cmap)
            ax.set_xticks(range(len(models)), [short[m] for m in models], rotation=90, fontsize=6.5)
            if panel is not None:
                ax.set_title(MODES[panel], fontsize=8, color=style.MODE_COLORS[MODE_KEYS[panel]])
            if phase != "experiment_1":
                for boundary in range(len(recipes), len(rows), len(recipes)):
                    ax.axhline(boundary - 0.5, color=style.BACKGROUND_COLOR, linewidth=2)
        axes[0, 0].set_yticks(range(len(rows)), row_labels, fontsize=6.5)
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-limit, limit))
        fig.colorbar(mappable, ax=list(axes[0]), shrink=0.8, pad=0.02,
                     label=f"{unit} (mean over datasets)")
        family = "decision features" if phase == "experiment_1" else "frozen embeddings"
        yield fig, f"effect_matrix_{phase}_{task}", (
            f"{layout.TASK_LABELS[task]} datasets: effect of adding {family} to each fixed "
            "predictor, relative to the same predictor on non-text columns, dataset and outer "
            "folds held fixed. Colour is the unweighted mean over datasets of the five-fold mean "
            "paired difference in the prespecified endpoint; text gives the datasets with a "
            "positive difference over those with five defined folds. Grey cells lack complete "
            "evidence. The colour scale is symmetric and clipped at the 95th percentile of "
            "|effect|. Descriptive; no recipe is selected and fold spread is not uncertainty."
            + (" J = joint PCA total width, P32 = 32 components per text column."
               if phase != "experiment_1" else "")
        )


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
                color = style.MODE_COLORS[MODE_KEYS[mode]]
                for pos, provider in enumerate(providers):
                    values = means.loc[means.provider.eq(provider) & means.feature_set.eq(mode), "effect"]
                    if values.empty:
                        continue
                    offsets = np.linspace(-0.15, 0.15, len(values))
                    ax.scatter(values, pos + offsets, color=color, alpha=0.3, s=10, linewidths=0)
                    ax.scatter(values.mean(), pos, color=color, marker="D", s=22, zorder=3)
                ax.set_title(label, color=color)
                ax.set_xlabel(unit, fontsize=7)
                ax.set_xlim(-bound, bound)
                ax.set_yticks(range(len(providers)), [LABELS[p] for p in providers], fontsize=7)
                ax.grid(axis="x", color=style.GRID_COLOR, alpha=0.4)
                ax.tick_params(axis="x", labelsize=7)
            axes[0].invert_yaxis()
            n = means.dataset_id.nunique()
            fig.suptitle(f"{style.PREDICTOR_LABELS[model]} · {layout.TASK_LABELS[task]} · {n} "
                         f"dataset{'s' if n != 1 else ''}"
                         f" · reference: {REFERENCE_LABELS.get(reference, reference)}", fontsize=9)
            yield fig, f"paired_{model}_{task}_{reference}", (
                "Paired representation effects within the same predictor and named reference. "
                "Small points are per-dataset averages of five defined paired outer-fold differences; "
                "diamonds are unweighted averages across the available datasets, separately per "
                "extractor/mode. Positive values indicate improvement in the prespecified "
                "AUC or R² endpoint; values retain their named metric units. "
                "Coverage may differ; exact dataset identities/counts remain in the table. "
                "Fold spread is not a confidence interval; these plots do not select a winning recipe."
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
                joint_rows = part[part.pca_scope.eq("joint")
                                  & part.pca_components.isin(JOINT_WIDTHS)]
                # A curve must compare the same dataset/split at every width.
                # Otherwise a newly completed easy dataset can look like a PCA benefit.
                complete = joint_rows.groupby(["dataset_id", "split_id"]).filter(
                    lambda frame: len(frame) == len(JOINT_WIDTHS)
                    and set(frame.pca_components) == set(JOINT_WIDTHS)
                )
                joint = complete.groupby("pca_components").effect.mean()
                if not joint.empty:
                    axes[0].plot(joint.index, joint, color=color, marker="o",
                                 label=f"{ENCODER_LABELS[encoder]} (n={complete.dataset_id.nunique()})",
                                 markersize=4)
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
            handles, _ = axes[0].get_legend_handles_labels()
            if handles:
                axes[0].legend(fontsize=7)
            else:
                axes[0].text(0.5, 0.5, "Awaiting datasets with all six widths",
                             transform=axes[0].transAxes, ha="center", fontsize=8)
            axes[1].set_xticks(range(3), ["Small\nE5", "Large\nE5", "Qwen"], fontsize=7)
            axes[1].set_title("Per-column PCA32", fontsize=8)
            fig.suptitle(f"{style.PREDICTOR_LABELS[model]} · {layout.TASK_LABELS[task]} · frozen embeddings"
                         f" · reference: {REFERENCE_LABELS.get(reference, reference)}", fontsize=9)
            yield fig, f"embedding_effect_{model}_{task}_{reference}", (
                "Paired frozen-embedding effects within a fixed predictor. Each joint curve "
                "uses only dataset/split pairs with five defined paired folds at all six joint "
                "widths, then averages their dataset-level effects. Legend n is this fixed "
                "cohort size; cohorts may differ between encoders. Per-column "
                "points are individual dataset differences and diamonds their means. Exact counts "
                "and identities remain in the tables; missing/unsupported arms are not imputed. "
                "Positive values indicate improvement in the named AUC or R² metric. Per-column32 "
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
