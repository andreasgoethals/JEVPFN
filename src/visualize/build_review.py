"""Extraction costs and main feature counts, using measured dataset statistics."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from src.visualize import style


def extraction_figure(estimates, summary, price):
    """Unique requests split by mode, and Jev input cost with its tokenizer uncertainty."""
    from src.visualize import layout

    order = layout.ordered(summary[["dataset_id", "dataset", "task_type"]])
    central = estimates[estimates.characters_per_token.eq(4)]
    per_column = central[central["mode"].eq("per_column")].groupby("dataset_id") \
        .unique_requests_within_group.sum()
    combined = central.groupby("dataset_id").requests_for_combined_build.sum()
    tokens = estimates.pivot_table(index="dataset_id", columns="characters_per_token",
                                   values="wire_tokens_combined_approx", aggfunc="sum")
    order = order.assign(
        per_column=order.dataset_id.map(per_column).to_numpy(),
        joint_extra=(order.dataset_id.map(combined) - order.dataset_id.map(per_column)).to_numpy(),
    )
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(style.WIDTH_FULL, 3.9), sharey=True)
    axes[0].barh(y, order.per_column, height=0.62, color=style.MODE_COLORS["per_column"],
                 label="Per-column")
    axes[0].barh(y, order.joint_extra, left=order.per_column, height=0.62,
                 color=style.MODE_COLORS["joint"], label="Joint, not already per-column")
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k"))
    axes[0].set_xlabel("Unique requests after exact reuse")
    axes[0].legend(loc="lower right", fontsize=7, frameon=False)
    if price is not None:
        cost = tokens.reindex(order.dataset_id) * price / 1e6
        axes[1].hlines(y, cost[5], cost[3], color=style.P95_COLOR, linewidth=1)
        axes[1].scatter(cost[4], y, s=14, color=style.MEDIAN_COLOR, zorder=3)
        totals = cost.sum()
        # "\$" keeps matplotlib from reading a pair of dollar signs as mathtext.
        axes[1].text(0.98, 0.02, f"All {len(order)} datasets: \\${totals[4]:,.2f}\n"
                     f"(range \\${totals[5]:,.2f}–\\${totals[3]:,.2f})", ha="right", va="bottom",
                     fontsize=7, transform=axes[1].transAxes)
        axes[1].set_xlim(0, cost[3].max() * 1.08)
        axes[1].set_xlabel("Jev input cost per dataset (USD)\n(dot: 4 chars/token; line: 5–3)")
    for ax in axes:
        layout.light_grid(ax)
        layout.task_rules(ax, order.task_type)
    layout.dataset_axis(axes[0], list(order.dataset), order.task_type, fontsize=7, separators=False)
    return (
        fig,
        "extraction_workload_and_cost",
        f"Left: unique requests per extractor for the {len(order)} selected datasets after exact "
        "reuse; joint requests equal to a single-field request are counted once, as per-column. "
        "Right: Jev input cost of building both representations at the configured price, with "
        "request bodies tokenized at 4 characters per token (range: 5 to 3). Costs exclude retries "
        "and unknown billing overhead; open-weight extractors incur GPU time instead.",
    )


def regression_scale_figure(labels):
    from textwrap import fill

    from matplotlib.colors import to_rgb

    items = sorted(((int(level), wording) for level, wording in labels.items()))
    fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, 3.0))
    ax.set_axis_off()
    ax.set(xlim=(0, 1), ylim=(len(items) - 0.5, -0.5))
    white = np.ones(3)
    for row, (level, wording) in enumerate(items):
        side = "lower" if level < 0 else "higher"
        tint = white if level == 0 else white + abs(level) / 4 * (np.array(
            to_rgb(style.DIRECTION_COLORS[side])) - white)
        ax.add_patch(plt.Rectangle((0, row - 0.42), 0.055, 0.84, color=tint))
        ax.text(0.0275, row, f"{level:+d}" if level else "0", ha="center", va="center", fontsize=7.5,
                weight="bold", color="white" if abs(level) >= 3 else style.TEXT_COLOR)
        name, _, criterion = wording.partition(":")
        ax.text(0.07, row, name, va="center", fontsize=7, weight="bold")
        ax.text(0.29, row, fill(criterion.strip(), 98), va="center", fontsize=6.8,
                linespacing=1.1)
    ax.axhline(3.5, xmin=0.07, color=style.GRID_COLOR, linewidth=0.6)
    ax.axhline(4.5, xmin=0.07, color=style.GRID_COLOR, linewidth=0.6)
    return (
        fig,
        "regression_evidence_levels",
        "The fixed nine-level rubric used for every regression question. Each level describes "
        "the strength and specificity of textual evidence for a relatively lower or higher "
        "target value; colour intensity grows with |level|. The extracted feature is the "
        "probability-weighted mean level; the full nine-level distribution is also stored. "
        "Levels are defined without target statistics or row targets and are not target units.",
    )
