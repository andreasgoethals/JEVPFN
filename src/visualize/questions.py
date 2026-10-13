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
