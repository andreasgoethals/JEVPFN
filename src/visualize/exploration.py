"""Measured dataset overview and text diagnostics; no model inference."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from src.visualize import style

TASK_LABELS = {
    "binary_classification": "Binary",
    "multiclass_classification": "Multiclass",
    "regression": "Regression",
}


def dataset_overview(audit):
    """One row per dataset; means pool nonempty text cells, not row targets."""
    table = audit.summary.copy()
    n = table.per_column_nonempty_inputs.replace(0, np.nan)
    table["mean_characters_per_nonempty_text"] = table.total_text_characters / n
    table["mean_tokens_per_nonempty_text"] = table.total_cell_text_tokens / n
    return table[
        [
            "dataset",
            "task_type",
            "target",
            "n_classes",
            "rows",
            "columns_including_target",
            "features",
            "text_columns",
            "non_text_columns",
            "mean_characters_per_nonempty_text",
            "mean_tokens_per_nonempty_text",
            "per_column_missing_or_empty_pct",
            "per_column_unique_inputs",
            "per_column_repeated_inputs",
            "per_column_reuse_savings_pct",
            "joint_missing_or_empty_pct",
            "joint_unique_inputs",
            "joint_repeated_inputs",
            "joint_reuse_savings_pct",
            "combined_unique_inputs",
            "combined_tokens_p95",
        ]
    ]


def overview_figure(table, characters_per_token=4.0):
    """Annotated, independently shaded columns: values retain their own units."""
    specs = [
        ("task_type", "Task", "task"),
        ("n_classes", "Target\nclasses", "int"),
        ("rows", "Rows", "int"),
        ("columns_including_target", "All\ncolumns¹", "int"),
        ("features", "Input\nfeatures", "int"),
        ("non_text_columns", "Non-text\nfeatures", "int"),
        ("text_columns", "Text\nfeatures", "int"),
        ("mean_characters_per_nonempty_text", "Mean chars\n/ text cell²", "float"),
        ("mean_tokens_per_nonempty_text", "Mean tokens\n/ text cell²", "float"),
        ("per_column_missing_or_empty_pct", "Empty text\ncells (%)", "pct"),
        ("per_column_repeated_inputs", "Per-column calls\navoided³: N / %", "per_column"),
        ("joint_repeated_inputs", "Joint calls\navoided³: N / %", "joint"),
        ("combined_unique_inputs", "Unique calls\nboth modes⁴", "int"),
        ("combined_tokens_p95", "Joint text tokens\n95th percentile", "float"),
    ]
    values = np.zeros((len(table), len(specs)))
    for j, (col, _, fmt) in enumerate(specs):
        if fmt != "task":
            arr = table[col].fillna(0).to_numpy(dtype=float)
            values[:, j] = arr / max(arr.max(), 1)
    fig, ax = plt.subplots(figsize=style.OVERVIEW_SIZE, layout="none")
    fig.subplots_adjust(left=0.16, right=0.99, top=0.83, bottom=0.12)
    ax.imshow(values, cmap=style.OVERVIEW_CMAP, vmin=0, vmax=2, aspect="auto")
    for i, (_, row) in enumerate(table.iterrows()):
        for j, (col, _, fmt) in enumerate(specs):
            value = row[col]
            if fmt == "task":
                label = TASK_LABELS[value]
            elif fmt in {"per_column", "joint"}:
                label = f"{value:,.0f}\n{row[fmt + '_reuse_savings_pct']:.1f}%"
            elif fmt == "pct":
                label = f"{value:.1f}%"
            elif fmt == "float":
                label = f"{value:,.1f}"
            else:
                label = (
                    "—" if value != value or (col == "n_classes" and not value) else f"{value:,.0f}"
                )
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=style.OVERVIEW_FONT,
                color=style.TASK_COLORS[row.task_type] if fmt == "task" else style.TEXT_COLOR,
            )
    ax.set_xticks(range(len(specs)), [s[1] for s in specs], fontsize=style.OVERVIEW_FONT)
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", length=0, pad=9)
    ax.set_yticks(range(len(table)), table.dataset, fontsize=style.OVERVIEW_FONT)
    ax.set_xticks(np.arange(-0.5, len(specs), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(table), 1), minor=True)
    ax.grid(which="minor", color=style.GRID_COLOR, linewidth=0.7)
    ax.tick_params(which="minor", length=0)
    fig.suptitle("20 core text datasets · dataset size, text burden and exact reuse", y=0.96)
    fig.text(
        0.16,
        0.91,
        "Darker cells indicate larger values within that column; compare the printed values across columns.",
    )
    fig.text(
        0.16,
        0.05,
        f"¹ Includes the target. Input features exclude it.  ² Nonempty cells only; tokens ≈ ceil(characters / {characters_per_token:g}).\n"
        "³ Additional nonempty inputs reused: absolute count and percentage.  ⁴ Includes overlap between per-column and joint inputs.",
    )
    caption = (
        "Large-format inspection overview of all 20 core datasets. Columns give task type, class count "
        "(not applicable to regression), dimensions, pooled nonempty-text means, missing-text percentage, "
        "exact reuse savings as counts and percentages of nonempty inputs, distinct requests for both modes, "
        "and the 95th percentile of combined text tokens per row. Shading is independently scaled by each "
        "column's maximum. Tokens use the character approximation; request metadata is excluded."
    )
    return fig, "dataset_overview", caption


def text_column_figures(audit):
    data = audit.text.merge(audit.summary[["dataset_id", "dataset", "task_type"]], on="dataset_id")
    datasets = list(audit.summary.dataset)
    positions = {name: i for i, name in enumerate(datasets)}
    fig, axes = plt.subplots(1, 2, figsize=style.DASHBOARD_SIZE, sharey=True)
    for dataset, group in data.groupby("dataset", sort=False):
        y = (
            positions[dataset] + np.linspace(-0.27, 0.27, len(group))
            if len(group) > 1
            else [positions[dataset]]
        )
        color = style.TASK_COLORS[group.task_type.iloc[0]]
        axes[0].scatter(group.nonempty_characters_mean, y, color=color, s=style.AUDIT_MARKER_SIZE)
        axes[1].scatter(group.missing_or_empty_pct, y, color=color, s=style.AUDIT_MARKER_SIZE)
    axes[0].set_yticks(range(len(datasets)), datasets)
    axes[0].invert_yaxis()
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Mean characters per nonempty text cell (log scale)")
    axes[0].set_title("Length of each text feature")
    axes[1].set_xlim(-2, 102)
    axes[1].set_xlabel("Missing or empty cells (%)")
    axes[1].set_title("Missingness of each text feature")
    for task, color in style.TASK_COLORS.items():
        axes[1].scatter([], [], color=color, label=TASK_LABELS[task])
    axes[1].legend(loc="lower right")
    yield (
        fig,
        "text_length_and_missingness",
        (
            "Each point represents one of the 90 official text columns. Left: mean character count "
            "among nonempty cells on a logarithmic axis. Right: null, nonfinite or whitespace-only "
            "cells as a percentage of all rows. Within-dataset offsets separate text columns; "
            "colours identify task types. This is a large-format inspection figure."
        ),
    )
    fig, ax = plt.subplots(figsize=style.figsize())
    for task, group in data.groupby("task_type"):
        ax.scatter(
            group.nonempty_characters_mean,
            group.unique_nonempty_ratio * 100,
            color=style.TASK_COLORS[task],
            label=TASK_LABELS[task],
            alpha=0.75,
            s=style.AUDIT_MARKER_SIZE,
        )
    ax.set_xscale("log")
    ax.set_ylim(-3, 103)
    ax.set_xlabel("Mean characters per nonempty text cell (log scale)")
    ax.set_ylabel("Unique inputs / nonempty cells (%)")
    ax.legend()
    yield (
        fig,
        "text_length_and_uniqueness",
        (
            "Mean nonempty character count versus the percentage of distinct nonempty inputs for "
            "each of the 90 text columns. Lower values on the vertical axis imply greater exact "
            "reuse within that field. The horizontal axis is logarithmic; colours indicate task type."
        ),
    )


def reuse_figure(summary):
    ordered = summary.sort_values("combined_unique_inputs")
    fig, axes = plt.subplots(1, 2, figsize=style.DASHBOARD_SIZE, sharey=True)
    y = np.arange(len(ordered))
    for mode, offset, color, label in [
        ("per_column", -0.17, style.MEDIAN_COLOR, "Per-column"),
        ("joint", 0.17, style.P95_COLOR, "Joint"),
    ]:
        for ax, suffix in zip(axes, ["reuse_savings_pct", "repeated_inputs"]):
            ax.barh(y + offset, ordered[f"{mode}_{suffix}"], height=0.32, color=color, label=label)
    axes[0].set_yticks(y, ordered.dataset)
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("Nonempty request slots avoided (%)")
    axes[1].set_xlabel("Nonempty request slots avoided (absolute count)")
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    axes[1].tick_params(axis="x", labelrotation=25)
    axes[0].legend(loc="lower right")
    return (
        fig,
        "exact_text_reuse",
        (
            "Exact reuse savings for per-column and joint requests on all 20 datasets. Left: "
            "avoided requests divided by nonempty slots, as percentages. Right: the same avoided "
            "requests as absolute counts. Missing inputs and cross-mode overlap are excluded. "
            "Axes start at zero. This is a large-format inspection figure."
        ),
    )
