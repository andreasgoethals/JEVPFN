"""Question-schema coverage; all plotting style comes from the shared style module."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from src.visualize import style


def question_schema_figure(table):
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=style.DASHBOARD_SIZE)
    y = np.arange(len(table))
    colors = [style.TASK_COLORS[t] for t in table.task]
    for ax, column, title in zip(
        axes,
        ["text_columns", "classes"],
        ["Text fields with documentary definitions", "Original classification labels"],
    ):
        bars = ax.barh(y, table[column], color=colors)
        ax.bar_label(bars, padding=2)
        ax.set_xlim(0, max(table[column]) * 1.2)
        ax.set_title(title)
        ax.set_xlabel("Count")
        ax.grid(axis="x", color=style.GRID_COLOR, alpha=0.5)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, table.dataset)
    axes[0].invert_yaxis()
    axes[1].legend(
        handles=[
            Patch(color=style.TASK_COLORS[task], label=label)
            for task, label in [
                ("binary_classification", "Binary"),
                ("multiclass_classification", "Multiclass"),
                ("regression", "Regression: continuous target"),
            ]
        ],
        loc="lower right",
    )
    fig.suptitle("The 18 question schemas before feature creation")
    return (
        fig,
        "question_schema_coverage",
        "Source-backed question schemas for the 18 selected datasets. Left: official text fields "
        "with stored descriptions, including explicitly unresolved source definitions. Right: "
        "original target-class vocabulary size; regression has no class vocabulary and uses the "
        "shared nine-level directional scale. Colors distinguish binary, multiclass and regression "
        "tasks. This is a large-format inspection overview.",
    )


def prepared_reuse_figure(table):
    """Report measured saved inputs, not a rows-times-fields workload estimate."""
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=style.DASHBOARD_SIZE)
    y = np.arange(len(table))
    axes[0].barh(y, table.unique_inputs, color=style.MEDIAN_COLOR, label="Unique inputs")
    axes[0].barh(
        y,
        table.reused_slots,
        left=table.unique_inputs,
        color=style.TOTAL_COLOR,
        label="Repeated slots avoided",
    )
    axes[0].set_xlabel("Nonempty per-column + joint slots")
    axes[0].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[0].legend(loc="lower right")
    bars = axes[1].barh(y, table.avoided_percent, color=style.TOTAL_COLOR)
    axes[1].bar_label(
        bars,
        labels=[
            f"{pct:.1f}% ({count:,})"
            for pct, count in zip(table.avoided_percent, table.reused_slots)
        ],
        padding=2,
    )
    axes[1].set_xlim(0, 145)
    axes[1].set_xticks([0, 25, 50, 75, 100])
    axes[1].set_xlabel("Nonempty slots avoided (%) and count")
    axes[0].set_yticks(y, table.dataset)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(axis="x", color=style.GRID_COLOR, alpha=0.5)
        ax.set_axisbelow(True)
    fig.suptitle("Exact reuse in the 18 saved input files")
    return (
        fig,
        "prepared_input_reuse",
        (
            "Measured deduplication in the 18 prepared dataset files. Left: distinct complete "
            "inputs and repeated nonempty per-column/joint slots. Right: avoided slots as a "
            "percentage and absolute count. Empty slots are excluded; a joint input identical "
            "to a per-column input shares the same saved input. Counts apply separately to each "
            "extractor configuration. This is a large-format inspection overview."
        ),
    )
