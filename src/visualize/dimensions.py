"""Paper-width class-probability feature counts, including declared removals."""

import matplotlib.pyplot as plt
import numpy as np

from src.visualize import style


def multiclass_dimensions_figure(dimensions):
    table = dimensions.loc[dimensions.task.eq("multiclass_classification")].sort_values(
        "combined_features", ascending=False
    )
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, style.AUDIT_RATIO))
    y = np.arange(len(table))
    for column, offset, color, label in (
        ("per_column_features", -0.25, style.MEDIAN_COLOR, "Separate: T × K"),
        ("joint_features", 0, style.P95_COLOR, "Joint: K"),
        ("combined_features", 0.25, style.TOTAL_COLOR, "Combined"),
    ):
        bars = ax.barh(y + offset, table[column], height=0.23, color=color, label=label)
        ax.bar_label(bars, padding=2)
    labels = [
        f"{r.dataset}\nT={r.text_columns}, K={int(r.classes)}"
        + (" · REMOVED" if r.status == "REMOVED" else "")
        for r in table.itertuples()
    ]
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, table.combined_features.max() * 1.18)
    ax.set_xlabel("Added numeric probability features per row")
    ax.set_title("Multiclass text conversion: input groups × classes")
    ax.legend(loc="lower right")
    ax.grid(axis="x", color=style.GRID_COLOR, alpha=0.5)
    ax.set_axisbelow(True)
    return (
        fig,
        "multiclass_numeric_feature_counts",
        "Number of added class-probability columns for every core multiclass dataset. "
        "T is the number of official text columns and K the number of classes. Separate "
        "extraction yields T × K columns and joint extraction K columns; combined concatenates "
        "them, removing the identical joint block when T=1. Spotify is retained for exploration "
        "and marked REMOVED from feature creation and predictive experiments. These are feature "
        "counts, not API-call counts; each request returns the complete class vector.",
    )
