"""B synthesis figures (B/B_synthesis): every Group B arm side by side, without per-dataset panels.

Inputs come from `group_b_views` (ladder, pairwise win rates, compute costs). Each function
yields ``(figure, name, caption)`` and nothing for an empty table.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src.visualize import layout, style
from src.visualize.group_b import _label, color_of, gain_forest, normalized_score_forest

LADDER_GROUPS = ("decision", "decision+frozen", "frozen", "probe", "tar", "native")


def _ladder_rows(summary):
    from src.evaluation.group_b_views import LADDER

    present = set(summary.representation)
    rows = [r for r in LADDER if r in present]
    rows += sorted(r for r in present if r.startswith("native/"))
    separators = [i for i in range(1, len(rows))
                  if rows[i].split("/")[0] != rows[i - 1].split("/")[0]]
    return rows, separators


def representation_ladder(summary, *, name, what, share=False):
    """Every Group B arm (rows) by predictor: raw gain per task family, or the pooled share."""
    if summary is None or summary.empty:
        return
    rows, separators = _ladder_rows(summary)
    yield from gain_forest(summary, rows, name=name, what=what, share=share, separators=separators)


def normalized_ladder(summary, *, name, what):
    """The ladder on the MulTaBench-style normalised score (no-text baseline as the first row)."""
    if summary is None or summary.empty:
        return
    rows, separators = _ladder_rows(summary)
    if "baseline" in set(summary.representation):
        rows, separators = ["baseline", *rows], [1, *(s + 1 for s in separators)]
    yield from normalized_score_forest(summary, rows, name=name, what=what, separators=separators)


def win_rate_matrix(pairwise, *, name, what):
    """Fold-level win rate of each row arm against each column arm, predictors pooled."""
    if pairwise is None or pairwise.empty:
        return
    from src.evaluation.group_b_views import MAIN_ARMS

    arms = [a for a in MAIN_ARMS if a in set(pairwise.learner) | set(pairwise.versus)]
    table = pairwise.pivot_table(index="learner", columns="versus", values="win_rate").reindex(
        index=arms, columns=arms)
    fig, ax = plt.subplots(figsize=(style.WIDTH_HALF, 3.0), layout="constrained")
    image = ax.imshow(table.to_numpy(float), cmap=style.AGREEMENT_DIVERGING, vmin=0, vmax=1)
    for (i, j), value in np.ndenumerate(table.to_numpy(float)):
        if np.isfinite(value):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(value - 0.5) > 0.3 else style.TEXT_COLOR)
    short = [_label(a).replace(" (median model)", "").replace("Label-trained classifier", "Classifier")
             .replace(" (same predictor)", "") for a in arms]
    ax.set_xticks(range(len(arms)), short, rotation=50, ha="right", fontsize=6)
    ax.set_yticks(range(len(arms)), short, fontsize=6)
    for tick, arm in zip([*ax.get_xticklabels()], arms):
        tick.set_color(color_of(arm))
    for tick, arm in zip([*ax.get_yticklabels()], arms):
        tick.set_color(color_of(arm))
    ax.tick_params(length=0)
    bar = fig.colorbar(image, ax=ax, shrink=0.75)
    bar.set_label("Row arm's fold win rate", fontsize=7)
    bar.ax.tick_params(labelsize=6)
    cells = int(pairwise.fold_cells.max())
    yield fig, name, (
        f"{what}: share of (dataset, predictor, outer fold) cells, up to {cells}, on which the row arm's "
        "headline score exceeds the column arm's (same predictor, same fold; ties count as not won). "
        "Median-model rows use the per-fold median over the complete decision-model cohort. Descriptive; "
        "intervals (dataset-cluster bootstrap) are in the printed table.")


def quality_vs_compute(points, *, name, what):
    """Utility of the both-view decision features against extraction GPU time per audited request."""
    if points is None or points.empty:
        return
    cost = "gpu_hours_per_1000_requests"
    data = points.loc[points.representation.str.startswith("decision/both/")].dropna(subset=[cost, "value"])
    data = data[data[cost].gt(0)]
    if data.empty:
        return
    single = data[data.representation.str.match(r"^decision/[a-z_]+/(?!median)")]
    named = data.drop(index=single.index)
    fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, 3.0), layout="constrained")
    ax.scatter(single[cost], single.value, s=10, color=style.DECISION_POOL_COLOR,
               linewidths=0, zorder=2)
    handles = [Line2D([], [], ls="none", marker="o", ms=3, color=style.DECISION_POOL_COLOR,
                      label="One decision model")] if not single.empty else []
    for _, row in named.iterrows():
        color = color_of(row.representation)
        ax.vlines(row[cost], row.low, row.high, color=color, lw=0.7)
        ax.scatter(row[cost], row.value, s=22, marker="D", color=color, zorder=3)
        handles.append(Line2D([], [], ls="none", marker="D", ms=4, color=color,
                              label=_label(row.representation).replace(" (median model)", " (median)")))
    ax.set_xscale("log")
    ax.axhline(0, color=style.MEDIAN_COLOR, lw=0.7)
    layout.light_grid(ax, axis="both")
    ax.tick_params(labelsize=6.5)
    ax.set_xlabel("Recorded extraction GPU-hours per 1,000 audited distinct requests (log scale)", fontsize=7)
    ax.set_ylabel("Mean headroom share (all datasets)", fontsize=7)
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=6.5)
    cohort = (
        "The diamond pairs the full cohort’s median extraction rate with the median-model utility arm. "
        "Its line spans predictors. " if not named.empty else
        "The cohort median is unavailable with the current coverage. "
    )
    yield fig, name, (
        "Each dot pairs one decision model’s both-view gain with the GPU-hours recorded for its "
        "successful extraction jobs per 1,000 distinct requests in the audited workload. The vertical "
        "coordinate is the median across predictors of the equal-dataset mean headroom gain. "
        f"{cohort}The table identifies missing rates and reports predictor fit time separately "
        "per 1,000 training rows.")
