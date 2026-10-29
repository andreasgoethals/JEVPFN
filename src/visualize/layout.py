"""One dataset axis grammar for every figure: same order, task colours and group rules.

Rows follow the catalog's task grouping (binary, multiclass, regression), so a dataset sits
at the same height in every figure and readers can compare panels without re-reading labels.
"""

from __future__ import annotations

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.visualize import style

TASK_ORDER = ("binary_classification", "multiclass_classification", "regression")
TASK_LABELS = {
    "binary_classification": "Binary",
    "multiclass_classification": "Multiclass",
    "regression": "Regression",
}


def ordered(table, task="task_type"):
    """Stable task grouping that keeps catalog order within each task."""
    rank = {name: i for i, name in enumerate(TASK_ORDER)}
    return table.assign(_rank=table[task].map(rank)).sort_values("_rank", kind="stable").drop(
        columns="_rank"
    )


def dataset_axis(ax, labels, tasks, *, removed=(), fontsize=None, separators=True):
    """Label rows top-down, colour them by task and rule between task groups."""
    ax.set_yticks(range(len(labels)), labels, fontsize=fontsize)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    removed = set(removed)
    for tick, task, label in zip(ax.get_yticklabels(), tasks, labels):
        tick.set_color(style.TASK_COLORS[task])
        if label in removed:
            tick.set_color(style.REMOVED_COLOR)
            tick.set_fontstyle("italic")
    if separators:
        task_rules(ax, tasks)
    ax.tick_params(axis="y", length=0)


def task_rules(ax, tasks):
    tasks = list(tasks)
    for position in range(1, len(tasks)):
        if tasks[position] != tasks[position - 1]:
            ax.axhline(position - 0.5, color=style.GRID_COLOR, linewidth=0.8, zorder=0)


def task_handles(*, marker=False):
    """Legend entries for the task colours used on dataset labels and markers."""
    if marker:
        return [
            Line2D([], [], linestyle="none", marker="o", markersize=4,
                   color=style.TASK_COLORS[t], label=TASK_LABELS[t])
            for t in TASK_ORDER
        ]
    return [Patch(color=style.TASK_COLORS[t], label=TASK_LABELS[t]) for t in TASK_ORDER]


def light_grid(ax, axis="x"):
    ax.grid(axis=axis, color=style.GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
