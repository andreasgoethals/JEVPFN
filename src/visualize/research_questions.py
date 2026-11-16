"""Design figures distinguish retained information/width from unmeasured performance."""

import matplotlib.pyplot as plt
import numpy as np

from src.visualize import style


def regression_width_figure(table):
    frame = table.sort_values("per_column_probabilities")
    fig, ax = plt.subplots(figsize=style.figsize(ratio=0.67))
    y = np.arange(len(frame))
    ax.hlines(
        y, frame.per_column_mean, frame.per_column_probabilities, color=style.GRID_COLOR, linewidth=2
    )
    ax.scatter(frame.per_column_mean, y, facecolors="white", edgecolors=style.FAMILY_COLORS["experiment_1"], label="Mean per text field", zorder=3)
    ax.scatter(
        frame.per_column_probabilities,
        y,
        color=style.FAMILY_COLORS["experiment_1"],
        marker="s",
        label="Nine probabilities per text field",
        zorder=3,
    )
    ax.set_yticks(y, frame.dataset)
    ax.set_xlabel("Total predictor columns (non-text + availability + decision features)")
    ax.set_title("Regression readouts: information retained versus input width")
    ax.set_xlim(0, 150)
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    return fig
