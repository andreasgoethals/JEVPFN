"""Exp 4 appendix candidate: decision models scored directly on outer-test rows (reference only).

One dot per decision model and dataset; a bar marks the median over models. Saved Exp 4
outer-test references (best no-text predictor, best structured arm) are drawn when they
exist. Every figure states that decision models are not tabular predictors.
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
NO_TEXT_COLOR = "#B35806"
STRUCTURED_COLOR = "#1B7837"
LOG_LOSS_TICKS = (0.1, 0.2, 0.5, 1, 2, 5)


def _dataset_dots(ax, datasets, frame, metric):
    for i, dataset in enumerate(datasets):
        values = frame.loc[frame.dataset_id.eq(dataset) & frame.metric.eq(metric), "value"].dropna()
        if values.empty:
            continue
        jitter = np.linspace(-0.18, 0.18, len(values)) if len(values) > 1 else np.zeros(1)
        ax.scatter(values, i + jitter, s=10, color=DOT_COLOR, alpha=0.75, linewidths=0, zorder=2)
        ax.plot([values.median()] * 2, [i - 0.32, i + 0.32], color=style.MEDIAN_COLOR, lw=1.6, zorder=3)


def _legend_handles(references_shown):
    handles = [
        Line2D([], [], ls="none", marker="o", ms=3.5, color=DOT_COLOR, label="One decision model"),
        Line2D([], [], color=style.MEDIAN_COLOR, lw=1.6, label="Median over models"),
    ]
    if references_shown:
        handles += [
            Line2D([], [], ls="none", marker="s", ms=4.5, mfc="none", mec=NO_TEXT_COLOR,
                   label="Best no-text predictor (Exp 4)"),
            Line2D([], [], ls="none", marker="D", ms=4.5, mfc="none", mec=STRUCTURED_COLOR,
                   label="Best Exp 4 arm"),
        ]
    return handles


def classification_figure(scores, references=None):
    """ROC-AUC / macro OvR AUC and log loss per classification dataset; yields (fig, name, caption)."""
    frame = scores[scores.status.eq("descriptive") & scores.task_type.ne("regression")]
    if frame.empty:
        return
    datasets, tasks, names = _ordered_datasets(frame.dataset_id.unique())
    labels = [names[d] for d in datasets]
    fig, (left, right) = plt.subplots(1, 2, figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 1.45 + 0.26 * len(datasets))),
                                      sharey=True, gridspec_kw={"width_ratios": [1.15, 1]})
    auc = frame[frame.metric.isin(CLASSIFICATION_METRICS.values())].assign(metric="auc")
    _dataset_dots(left, datasets, auc, "auc")
    _dataset_dots(right, datasets, frame, "log_loss")
    shown = references is not None and not references.empty
    if shown:
        for i, dataset in enumerate(datasets):
            ref = references[references.dataset_id.eq(dataset)]
            if ref.empty:
                continue
            left.scatter(ref.no_text_best, [i], marker="s", s=22, facecolors="none",
                         edgecolors=NO_TEXT_COLOR, zorder=4)
            left.scatter(ref.best_structured, [i], marker="D", s=22, facecolors="none",
                         edgecolors=STRUCTURED_COLOR, zorder=4)
    left.axvline(0.5, color=style.WINDOW_COLOR, lw=0.8, ls=":", zorder=1)
    left.set_xlabel("ROC-AUC (binary) / macro OvR ROC-AUC")
    right.set_xlabel("Log loss (lower is better; log axis)")
    right.set_xscale("log")
    right.xaxis.set_major_locator(FixedLocator(LOG_LOSS_TICKS))
    right.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    right.xaxis.set_minor_formatter(NullFormatter())
    layout.dataset_axis(left, labels, tasks)
    right.tick_params(axis="y", length=0)
    layout.task_rules(right, tasks)
    for ax in (left, right):
        layout.light_grid(ax)
    fig.suptitle(REFERENCE_LABEL, fontsize=9, style="italic")
    fig.legend(handles=_legend_handles(shown), loc="outside lower center", ncol=4 if shown else 2,
               frameon=False, fontsize=7.5)
    caption = (
        "Classification datasets (rows, grouped by task). Each grey dot is one decision model's "
        "stored joint-request class probabilities scored directly, without fitting, on the outer-test "
        "rows of the shared five-fold splits (mean over folds); the dark bar is the median over models. "
        "Left: ROC-AUC for binary and macro one-vs-rest ROC-AUC for multiclass tasks (dotted line: 0.5). "
        "Right: log loss on a logarithmic axis. Rows without text receive the uniform distribution. "
        + ("Open markers show the best saved Exp 4 no-text predictor and best Exp 4 arm on the same rows, "
           "both selected after the fact for description only."
           if shown else "Exp 4 predictor references are pending (no completed outer-test results).")
    )
    yield fig, "direct_decision_scores_classification", caption


def regression_figure(scores):
    """Spearman rank correlation of the expected evidence score with the target, per dataset."""
    frame = scores[scores.status.eq("descriptive") & scores.metric.eq(REGRESSION_METRIC)]
    if frame.empty:
        return
    datasets, tasks, names = _ordered_datasets(frame.dataset_id.unique())
    fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 1.55 + 0.26 * len(datasets))))
    _dataset_dots(ax, datasets, frame, REGRESSION_METRIC)
    ax.axvline(0, color=style.WINDOW_COLOR, lw=0.8, ls=":", zorder=1)
    ax.set_xlim(-0.5, 1)
    ax.set_xlabel(f"{METRIC_LABELS[REGRESSION_METRIC]}: expected evidence score vs target")
    layout.dataset_axis(ax, [names[d] for d in datasets], tasks)
    layout.light_grid(ax)
    fig.suptitle(f"{REFERENCE_LABEL}; rank correlation only, not R²", fontsize=9, style="italic")
    fig.legend(handles=_legend_handles(False), loc="outside lower center", ncol=2, frameon=False,
               fontsize=7.5)
    caption = (
        "Regression datasets (rows). Each grey dot is the Spearman rank correlation, on the outer-test "
        "rows of the shared five-fold splits (mean over folds), between one decision model's expected "
        "evidence score (nine ordered levels −4 to 4 weighted by their stored joint-request probabilities) "
        "and the target; the dark bar is the median over models (dotted line: no rank association; "
        "axis from −0.5). "
        "Nothing is fitted; the score has no target scale, so this describes ranking only and is not "
        "comparable with the benchmark's R²."
    )
    yield fig, "direct_decision_scores_regression", caption


def direct_score_figures(scores, references=None):
    yield from classification_figure(scores, references)
    yield from regression_figure(scores)
