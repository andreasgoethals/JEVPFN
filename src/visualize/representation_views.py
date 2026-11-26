"""B2 supporting views: training-only PCA compression and row-level complementarity.

Each function yields ``(figure, name, caption)`` and nothing for an empty table.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src.visualize import layout, style
from src.visualize.group_b import ENCODER_NAMES, MARKERS, PREDICTORS, predictor_handles

SCOPES = {"per_column": "Per column, PCA32", "joint": "Joint, PCA64"}


def pca_variance(table, *, name="pca_training_variance"):
    """Training explained-variance fraction of the outer-training PCA fits per encoder and scope."""
    if table is None or table.empty:
        return
    data = table.dropna(subset=["training_explained_variance_fraction"])
    encoders = [e for e in style.EMBEDDING_COLORS if e in set(data.encoder)]
    scopes = [s for s in SCOPES if s in set(data.pca_scope)]
    fig, axes = plt.subplots(1, len(scopes), figsize=(style.WIDTH_FULL, 2.6), layout="constrained",
                             sharey=True, squeeze=False)
    rng = np.random.default_rng(0)
    for ax, scope in zip(axes[0], scopes):
        part = data[data.pca_scope.eq(scope)]
        means = part.groupby(["encoder", "dataset_id", "task_type", "text_group"],
                             as_index=False).training_explained_variance_fraction.mean()
        for i, encoder in enumerate(encoders):
            rows = means[means.encoder.eq(encoder)]
            x = i + rng.uniform(-0.22, 0.22, len(rows))
            ax.scatter(x, rows.training_explained_variance_fraction, s=8, linewidths=0, alpha=0.75,
                       c=[style.TASK_COLORS[t] for t in rows.task_type])
            median = rows.training_explained_variance_fraction.median()
            ax.hlines(median, i - 0.3, i + 0.3, color=style.EMBEDDING_COLORS[encoder], lw=2)
        ax.set_xticks(range(len(encoders)), [ENCODER_NAMES.get(e, e) for e in encoders], fontsize=7)
        ax.set_ylim(0, 1.02)
        ax.tick_params(axis="y", labelsize=6.5)
        layout.light_grid(ax, axis="y")
        unit = "dataset x text column" if scope == "per_column" else "dataset"
        ax.set_title(f"{SCOPES[scope]} (one dot per {unit})", loc="left", fontsize=7.5)
    axes[0, 0].set_ylabel("Training explained-variance fraction", fontsize=7)
    handles = layout.task_handles(marker=True) + [
        Line2D([], [], color=style.MEDIAN_COLOR, lw=2, label="Median (encoder colour)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=7)
    n = data.dataset_id.nunique()
    yield fig, name, (
        "Cumulative explained-variance fraction of the PCA fitted on outer-training rows only (mean over "
        f"the five outer folds) for each frozen encoder, {n} datasets: per-column PCA32 per dataset and "
        "text column (left), joint PCA64 per dataset (right); marker colour gives the task, the bar the "
        "median. Read from the prepared PCA stores (outer-training partitions only; inner binary "
        "partitions excluded). Variance retained is not predictive information.")


FAMILY_NAMES = {"embedding": "frozen", "tar": "TAR"}


def complementarity(summary, *, name, what):
    """Row-benefit Spearman rho and top-10% overlap above chance, per decision view x family."""
    if summary is None or summary.empty:
        return
    data = summary[summary.aggregation.eq("mode_by_family")].copy()
    if data.empty:
        return
    data["row"] = (data.decision.str.split("/").str[1].str.replace("_", " ") + " vs "
                   + data.other.map(FAMILY_NAMES).fillna(data.other))
    data["excess"] = data.mean_jaccard_top - data.mean_jaccard_chance
    rows = sorted(set(data.row), key=lambda r: (r.split(" vs ")[1], ["per column", "joint", "both"].index(
        r.split(" vs ")[0]) if r.split(" vs ")[0] in ("per column", "joint", "both") else 9))
    models = [m for m in PREDICTORS if m in set(data.model_id)]
    offsets = dict(zip(models, np.linspace(-0.3, 0.3, max(len(models), 1))))
    fig, axes = plt.subplots(1, 2, figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 1.3 + 0.36 * len(rows))),
                             layout="constrained", sharey=True)
    for model in models:
        frame = data[data.model_id.eq(model)].set_index("row").reindex(rows)
        y = np.arange(len(rows)) + offsets[model]
        axes[0].hlines(y, frame.rho_ci_low, frame.rho_ci_high, color=style.PREDICTOR_INK, lw=0.6)
        axes[0].scatter(frame.mean_spearman_rho, y, marker=MARKERS[model], s=13, facecolors="none",
                        edgecolors=style.PREDICTOR_INK, linewidths=0.6, zorder=3)
        axes[1].scatter(frame.excess, y, marker=MARKERS[model], s=13, facecolors="none",
                        edgecolors=style.PREDICTOR_INK, linewidths=0.6, zorder=3)
    for ax, label in zip(axes, ("Spearman ρ of per-row benefits", "Top-10% Jaccard minus chance")):
        ax.axvline(0, color=style.MEDIAN_COLOR, lw=0.7)
        layout.light_grid(ax)
        ax.tick_params(labelsize=6.5)
        ax.set_xlabel(label, fontsize=7)
    axes[0].set_yticks(range(len(rows)), [f"Decision {r}" for r in rows], fontsize=7)
    axes[0].set_ylim(len(rows) - 0.5, -0.5)
    axes[0].tick_params(axis="y", length=0)
    fig.legend(handles=predictor_handles(models), loc="outside lower center", ncol=4, frameon=False, fontsize=7)
    n = f"{int(data.n_datasets.min())}–{int(data.n_datasets.max())}"
    yield fig, name, (
        f"{what}: on identical held-out rows and folds of one predictor, the Spearman correlation between "
        "a decision arm's and a frozen-embedding or TAR arm's per-row benefit over the no-text baseline "
        "(left; line: 95% dataset-bootstrap interval, 2,000 draws) and the Jaccard overlap of their "
        "10% most-improved rows minus its value for independent sets (right). Values are averaged within "
        f"each dataset over decision models of one view and recipes of one family, then over {n} datasets "
        "with equal weight; one marker per predictor. Descriptive.")
