"""What each prepared request contains: fixed task context versus the row's own text."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.visualize import layout, style

CHARACTERS_PER_TOKEN = 4


def request_composition_figure(table):
    """Context and text sizes (left) and the text share of each request (right)."""
    order = layout.ordered(table.drop_duplicates("dataset_id"), task="task")[["dataset_id", "dataset", "task"]]
    position = {d: i for i, d in enumerate(order.dataset_id)}
    fig, axes = plt.subplots(1, 2, figsize=(style.WIDTH_FULL, 3.9), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1.15]})
    shift = {"per_column": -0.17, "joint": 0.17}
    for row in table.itertuples():
        y = position[row.dataset_id] + shift[row.mode]
        color = style.MODE_COLORS[row.mode]
        axes[0].scatter(row.text_characters_median / CHARACTERS_PER_TOKEN, y, s=14,
                        color=color, zorder=3)
        axes[0].scatter(row.context_characters_median / CHARACTERS_PER_TOKEN, y, s=14,
                        marker="s", color=style.QUANTITY_COLOR, zorder=3)
        axes[0].plot([row.text_characters_median / CHARACTERS_PER_TOKEN,
                      row.context_characters_median / CHARACTERS_PER_TOKEN], [y, y],
                     color=style.GRID_COLOR, linewidth=1, zorder=1)
        axes[1].hlines(y, row.text_share_p10, row.text_share_p90, color=color, linewidth=0.8)
        axes[1].add_patch(plt.Rectangle((row.text_share_p25, y - 0.13),
                                        row.text_share_p75 - row.text_share_p25, 0.26,
                                        facecolor=style.BACKGROUND_COLOR, edgecolor=color,
                                        linewidth=0.9, zorder=2))
        axes[1].vlines(row.text_share_p50, y - 0.13, y + 0.13, color=color, linewidth=1.5,
                       zorder=3)
    axes[0].set_xscale("log")
    axes[0].set_xticks([3, 10, 30, 100, 300, 1000], ["3", "10", "30", "100", "300", "1,000"])
    axes[0].set_xlabel("Median size per unique request\n(approximate tokens, log)")
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Row text as share of the request (%)\n(box p25–p75, line p10–p90)")
    for ax in axes:
        layout.light_grid(ax)
        layout.task_rules(ax, order.task)
    layout.dataset_axis(axes[0], list(order.dataset), order.task, fontsize=7, separators=False)
    handles = [
        Line2D([], [], linestyle="none", marker="s", markersize=4, color=style.QUANTITY_COLOR,
               label="Task context, question and options"),
        Line2D([], [], linestyle="none", marker="o", markersize=4,
               color=style.MODE_COLORS["per_column"], label="Single-field request"),
        Line2D([], [], linestyle="none", marker="o", markersize=4,
               color=style.MODE_COLORS["joint"], label="Joint request (≥ 2 fields)"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncols=3, fontsize=7, frameon=False)
    return (
        fig,
        "request_composition",
        "Composition of the unique provider-neutral requests saved for the 18 selected "
        "datasets. Left: median length of the fixed part (task context, question and answer "
        "options, as canonical JSON) and of the row text, for requests with one named field and "
        "for joint requests with two or more fields; joint requests that contain a single "
        "available field equal a single-field request and are counted there. Right: "
        "distribution of the row-text share of each request, including JSON escaping. "
        "Lengths are serialized characters/4, not information content; exact "
        "per-provider counts, which also include native prompt framing, come from the "
        "Experiment 0 token audit.",
    )
