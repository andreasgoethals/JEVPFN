"""Paper-width class-probability feature counts, including declared removals."""

import matplotlib.pyplot as plt
import numpy as np

from src.visualize import style


def multiclass_dimensions_figure(dimensions):
    table = dimensions.loc[dimensions.task.eq("multiclass_classification")].sort_values(
        "combined_features", ascending=False
    )
    fig, axes = plt.subplots(2, 1, figsize=style.DASHBOARD_SIZE,
                              gridspec_kw={"height_ratios": [1, 4]})
    for ax, part in zip(axes, [table[table.status.eq("REMOVED")], table[table.status.ne("REMOVED")]]):
        y = np.arange(len(part))
        for column, offset, color, label in (
            ("per_column_features", -0.25, style.MODE_COLORS["per_column"], "Per-column: T × K"),
            ("joint_features", 0, style.MODE_COLORS["joint"], "Joint: K"),
            ("combined_features", 0.25, style.MODE_COLORS["both"], "Both"),
        ):
            bars = ax.barh(y + offset, part[column], height=0.23, color=color, label=label)
            ax.bar_label(bars, padding=2, fontsize=7)
        labels = [f"{r.dataset}\nT={r.text_columns}, K={int(r.classes)}" +
                  (" · removed" if r.status == "REMOVED" else "") for r in part.itertuples()]
        ax.set_yticks(y, labels, fontsize=7)
        ax.invert_yaxis()
        if not part.empty:
            ax.set_xlim(0, part.combined_features.max() * 1.18)
        ax.grid(axis="x", color=style.GRID_COLOR, alpha=0.5)
        ax.set_axisbelow(True)
    axes[0].set_title("Removed high-class-count task", fontsize=8)
    axes[1].set_title("Six included multiclass datasets · separate horizontal scale", fontsize=8)
    axes[1].set_xlabel("Added numeric probability columns per row")
    fig.legend(*axes[1].get_legend_handles_labels(), loc="outside upper center", ncols=3, fontsize=7)
    return (
        fig,
        "multiclass_numeric_feature_counts",
        "Number of added class-probability columns for every core multiclass dataset, using "
        "separate horizontal scales for the removed Spotify task and the six included datasets. "
        "T is the number of official text columns and K the number of classes. Separate "
        "extraction yields T × K columns and joint extraction K columns; combined concatenates "
        "them, removing the identical joint block when T=1. Spotify is retained for exploration "
        "and marked REMOVED from feature creation and predictive experiments. These are feature "
        "counts, not API-call counts; each request returns the complete class vector.",
    )
