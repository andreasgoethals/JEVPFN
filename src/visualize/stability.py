"""A2 (exp02_robustness): does averaging over wordings reduce wording sensitivity?"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src.visualize import style
from src.visualize.decision_diagnostics import _models, short

SERIES = (("replay_tv", "Unchanged replay", "#9AA3AD", "o"),
          ("single_tv", "Single wordings", "#B35806", "s"),
          ("averaged_tv", "Averages of two wordings", "#2166AC", "D"))
MODES = (("per_column", "a  Per-column requests"), ("joint", "b  Joint requests"))


def wording_averaging_figure(summary):
    """Per model: replay noise, single-wording distance and averaged-pair distance."""
    if summary.empty:
        return
    names = _models(summary.provider)
    fig, axes = plt.subplots(1, 2, sharey=True, sharex=True,
                             figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 1.3 + 0.2 * len(names))))
    for ax, (mode, title) in zip(axes, MODES):
        frame = summary[summary["mode"].eq(mode)].set_index("provider").reindex(names)
        for column, _, color, marker in SERIES:
            ax.scatter(frame[column], np.arange(len(names)), s=14, color=color, marker=marker, zorder=3,
                       linewidths=0)
        for i, row in enumerate(frame.itertuples()):
            if np.isfinite([row.single_tv, row.averaged_tv]).all():
                ax.plot([row.single_tv, row.averaged_tv], [i, i], color=style.GRID_COLOR, lw=1.2, zorder=2)
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.grid(axis="x", color=style.GRID_COLOR, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlim(0, None)
        ax.set_xlabel("Mean total variation distance", fontsize=8)
    axes[0].set_yticks(range(len(names)), [short(n) for n in names], fontsize=7.5)
    axes[0].set_ylim(len(names) - 0.5, -0.5)
    for tick, name in zip(axes[0].get_yticklabels(), names):
        tick.set_color(style.DECISION_COLORS.get(name, style.TEXT_COLOR))
    axes[1].tick_params(axis="y", length=0)
    fig.legend(handles=[Line2D([], [], ls="none", marker=m, ms=4, color=c, label=label) for _, label, c, m in SERIES],
               loc="outside lower center", ncol=3, frameon=False, fontsize=7.5)
    yield fig, "wording_averaging", (
        "Wording sensitivity of each decision model with and without averaging, from responses already collected "
        "for the stability tests (no additional model calls). Orange squares: mean total variation distance between "
        "single wordings across the pairs {original, paraphrase 1} and {paraphrase 2, paraphrase 3}; blue diamonds: "
        "distance between the averages of those two pairs; grey circles: distance between the original and its "
        "unchanged replays. Within-dataset means are combined with equal dataset weight; (a) per-column and "
        "(b) joint requests."
    )
