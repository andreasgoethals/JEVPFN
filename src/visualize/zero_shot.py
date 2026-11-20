"""A1 stand-alone quality and calibration figures (joint requests; reference only).

One dot per decision model and dataset; a bar marks the median over models. Reliability
diagrams show every model's top-label calibration per classification dataset. Every figure
states that decision models are not tabular predictors.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter

from src.evaluation.zero_shot import (
    CLASSIFICATION_METRICS,
    METRIC_LABELS,
    REFERENCE_LABEL,
    REGRESSION_METRIC,
)
from src.visualize import layout, style
from src.visualize.study import _ordered_datasets

DOT_COLOR = "#7A8794"
LOG_LOSS_TICKS = (0.1, 0.2, 0.5, 1, 2, 5)


def _dataset_dots(ax, datasets, frame, metric):
    for i, dataset in enumerate(datasets):
        values = frame.loc[frame.dataset_id.eq(dataset) & frame.metric.eq(metric), "value"].dropna()
        if values.empty:
            continue
        jitter = np.linspace(-0.18, 0.18, len(values)) if len(values) > 1 else np.zeros(1)
        ax.scatter(values, i + jitter, s=10, color=DOT_COLOR, alpha=0.75, linewidths=0, zorder=2)
        ax.plot([values.median()] * 2, [i - 0.32, i + 0.32], color=style.MEDIAN_COLOR, lw=1.6, zorder=3)


def _legend_handles():
    return [Line2D([], [], ls="none", marker="o", ms=3.5, color=DOT_COLOR, label="One decision model"),
            Line2D([], [], color=style.MEDIAN_COLOR, lw=1.6, label="Median over models")]


def classification_figure(scores):
    """AUC, log loss and expected calibration error per classification dataset."""
    frame = scores[scores.status.eq("descriptive") & scores.task_type.ne("regression")]
    if frame.empty:
        return
    datasets, tasks, names = _ordered_datasets(frame.dataset_id.unique())
    fig, axes = plt.subplots(1, 3, figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 1.5 + 0.26 * len(datasets))),
                             sharey=True)
    auc = frame[frame.metric.isin(CLASSIFICATION_METRICS.values())].assign(metric="auc")
    _dataset_dots(axes[0], datasets, auc, "auc")
    _dataset_dots(axes[1], datasets, frame, "log_loss")
    _dataset_dots(axes[2], datasets, frame, "ece")
    axes[0].axvline(0.5, color=style.WINDOW_COLOR, lw=0.8, ls=":", zorder=1)
    axes[0].set_title("a  ROC-AUC / macro OvR AUC", loc="left", fontsize=8.5)
    axes[1].set_title("b  Log loss (log axis)", loc="left", fontsize=8.5)
    axes[2].set_title("c  Calibration error (ECE)", loc="left", fontsize=8.5)
    axes[1].set_xscale("log")
    axes[1].xaxis.set_major_locator(FixedLocator(LOG_LOSS_TICKS))
    axes[1].xaxis.set_major_formatter(FormatStrFormatter("%g"))
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    axes[2].set_xlim(0, None)
    layout.dataset_axis(axes[0], [names[d] for d in datasets], tasks, fontsize=7.5)
    for ax in axes:
        layout.light_grid(ax)
        layout.task_rules(ax, tasks)
    for ax in axes[1:]:
        ax.tick_params(axis="y", length=0)
    fig.suptitle(REFERENCE_LABEL, fontsize=9, style="italic")
    fig.legend(handles=_legend_handles(), loc="outside lower center", ncol=2, frameon=False, fontsize=7.5)
    yield fig, "standalone_quality_classification", (
        "Classification datasets (rows, grouped by task). Each grey dot is one decision model's stored joint-request "
        "class probabilities scored directly against the target, without fitting or recalibration, on every row that "
        "has a joint request (rows without text are not scored); the dark bar is the median over models. (a) ROC-AUC "
        "for binary and macro one-vs-rest ROC-AUC for multiclass tasks (dotted line: 0.5). (b) Log loss on a "
        "logarithmic axis. (c) Expected calibration error of the top-label confidence over ten equal-width bins."
    )


def regression_figure(scores):
    """Spearman correlation of the expected evidence level with the target, per dataset."""
    frame = scores[scores.status.eq("descriptive") & scores.metric.eq(REGRESSION_METRIC)]
    if frame.empty:
        return
    datasets, tasks, names = _ordered_datasets(frame.dataset_id.unique())
    fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 1.55 + 0.26 * len(datasets))))
    _dataset_dots(ax, datasets, frame, REGRESSION_METRIC)
    ax.axvline(0, color=style.WINDOW_COLOR, lw=0.8, ls=":", zorder=1)
    ax.set_xlim(-0.5, 1)
    ax.set_xlabel(f"{METRIC_LABELS[REGRESSION_METRIC]}: expected evidence level vs target")
    layout.dataset_axis(ax, [names[d] for d in datasets], tasks)
    layout.light_grid(ax)
    fig.suptitle(f"{REFERENCE_LABEL}; rank correlation only, not R²", fontsize=9, style="italic")
    fig.legend(handles=_legend_handles(), loc="outside lower center", ncol=2, frameon=False, fontsize=7.5)
    yield fig, "standalone_quality_regression", (
        "Regression datasets (rows). Each grey dot is the Spearman rank correlation, over every row with a joint "
        "request, between one decision model's expected evidence level (nine ordered levels −4 to +4 weighted by "
        "their stored probabilities) and the target; the dark bar is the median over models (dotted line: no rank "
        "association; axis from −0.5). Nothing is fitted; the level has no target scale, so this describes ranking "
        "only and is not comparable with the benchmark's R²."
    )


def reliability_figure(reliability):
    """Top-label reliability diagrams: one panel per classification dataset, one line per model."""
    if reliability.empty:
        return
    datasets, tasks, names = _ordered_datasets(reliability.dataset_id.unique())
    columns = 3
    rows = int(np.ceil(len(datasets) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 0.4 + 1.55 * rows)),
                             sharex=True, sharey=True, squeeze=False)
    for ax, dataset, task in zip(axes.flat, datasets, tasks):
        frame = reliability[reliability.dataset_id.eq(dataset) & reliability["count"].gt(0)]
        ax.plot([0, 1], [0, 1], color=style.WINDOW_COLOR, lw=0.8, ls=":")
        for provider, curve in frame.groupby("provider"):
            curve = curve.sort_values("bin")
            ax.plot(curve.mean_confidence, curve.accuracy, lw=0.8, marker="o", ms=1.6,
                    color=style.DECISION_COLORS.get(provider, DOT_COLOR), alpha=0.7)
        ax.set_title(names[dataset], loc="left", fontsize=7.5, color=style.TASK_COLORS[task])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        layout.light_grid(ax, axis="both")
    for ax in axes.flat[len(datasets):]:
        ax.set_visible(False)
    fig.supxlabel("Mean top-label confidence in bin", fontsize=8)
    fig.supylabel("Share correct", fontsize=8)
    fig.suptitle(REFERENCE_LABEL, fontsize=9, style="italic")
    yield fig, "standalone_reliability", (
        "Reliability diagrams of the decision models' stored joint-request probabilities, one panel per classification "
        "dataset and one line per model (model colours as in the other figures). Rows with a joint request are grouped "
        "into ten equal-width bins of the top-label confidence; each point shows a non-empty bin's mean confidence "
        "and the share of rows whose most probable class is the target (exact ties share credit). Points below the "
        "dotted diagonal are over-confident. No recalibration is applied."
    )


def direct_score_figures(scores, reliability=None):
    yield from classification_figure(scores)
    yield from regression_figure(scores)
    if reliability is not None:
        yield from reliability_figure(reliability)
