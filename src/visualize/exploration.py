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
    """Counts follow exact request equality within each named text column."""
    table = audit.summary.copy()
    from src.utils.config import load_config

    removed = set(load_config("feature_creation")["dataset_selection"]["exclude"])
    table["experiment_status"] = np.where(table.dataset_id.isin(removed), "REMOVED", "Included")
    n = table.per_column_nonempty_inputs.replace(0, np.nan)
    table["text_cells"] = table.rows * table.text_columns
    table["nonempty_text_cells"] = table.per_column_nonempty_inputs
    table["empty_text_cells"] = table.text_cells - table.nonempty_text_cells
    table["distinct_text_cells"] = table.per_column_unique_inputs
    table["distinct_text_pct"] = (100 * table.distinct_text_cells / n).fillna(0)
    table["mean_characters_per_nonempty_text"] = table.total_text_characters / n
    table["mean_tokens_per_nonempty_text"] = table.total_cell_text_tokens / n
    table["joint_empty_rows"] = table.rows - table.joint_nonempty_inputs
    table["combined_avoided_requests"] = (
        table.combined_nonempty_inputs - table.combined_unique_inputs
    )
    return table[
        [
            "dataset",
            "experiment_status",
            "task_type",
            "n_classes",
            "rows",
            "text_columns",
            "non_text_columns",
            "text_cells",
            "per_column_missing_or_empty_pct",
            "distinct_text_cells",
            "distinct_text_pct",
            "mean_tokens_per_nonempty_text",
            "joint_nonempty_inputs",
            "joint_missing_or_empty_pct",
            "joint_unique_inputs",
            "joint_reuse_savings_pct",
            "combined_tokens_median",
            "combined_tokens_p95",
            "combined_tokens_max",
            "combined_unique_inputs",
            "combined_avoided_requests",
            "combined_reuse_savings_pct",
        ]
    ]


OVERVIEW_LABELS = {
    "dataset": "Dataset",
    "experiment_status": "Experiment status",
    "task_type": "Task",
    "n_classes": "Classes",
    "rows": "Rows",
    "text_columns": "Text columns",
    "non_text_columns": "Non-text columns",
    "text_cells": "All text cells",
    "nonempty_text_cells": "Nonempty text cells",
    "empty_text_cells": "Empty text cells",
    "per_column_missing_or_empty_pct": "Empty text cells (%)",
    "distinct_text_cells": "Distinct texts (within columns)",
    "distinct_text_pct": "Distinct / nonempty texts (%)",
    "per_column_repeated_inputs": "Per-column requests avoided",
    "per_column_reuse_savings_pct": "Per-column avoided (%)",
    "mean_characters_per_nonempty_text": "Mean characters / nonempty text",
    "mean_tokens_per_nonempty_text": "Mean tokens / nonempty text",
    "joint_nonempty_inputs": "Rows with text",
    "joint_empty_rows": "Rows without text",
    "joint_missing_or_empty_pct": "Rows without text (%)",
    "joint_unique_inputs": "Distinct joint texts",
    "joint_repeated_inputs": "Joint requests avoided",
    "joint_reuse_savings_pct": "Joint avoided (%)",
    "combined_tokens_median": "Joint tokens median",
    "combined_tokens_p95": "Joint tokens p95",
    "combined_tokens_max": "Joint tokens maximum",
    "combined_nonempty_inputs": "Nonempty slots (both modes)",
    "combined_unique_inputs": "Unique requests (both modes)",
    "combined_avoided_requests": "Requests avoided (both modes)",
    "combined_reuse_savings_pct": "Avoided (both modes, %)",
}


def overview_figure(table, characters_per_token=4.0):
    """A single overview with separate header/body/footer axes, also safe in Jupyter."""
    specs = [
        ("task_type", "Task /\nclasses", "task"),
        ("rows", "Rows", "int"),
        ("text_columns", "Text\ncolumns", "int"),
        ("non_text_columns", "Non-text\ncolumns", "int"),
        ("text_cells", "All text\ncells", "int"),
        ("per_column_missing_or_empty_pct", "Empty cells\n(%)", "float"),
        ("distinct_text_cells", "Distinct texts", "int"),
        ("mean_tokens_per_nonempty_text", "Mean tokens\n/ text", "float"),
        ("joint_nonempty_inputs", "Rows\nwith text", "int"),
        ("joint_unique_inputs", "Distinct\njoint texts", "int"),
        ("joint_reuse_savings_pct", "Joint reuse\n(%)", "float"),
        ("combined_tokens_median", "Joint tokens\nmedian", "float"),
        ("combined_tokens_p95", "Joint tokens\np95 / max", "tail"),
        ("combined_unique_inputs", "Unique requests\nboth modes", "int"),
        ("combined_avoided_requests", "Both modes saved\nN / %", "combined_reuse_savings_pct"),
    ]
    values = np.zeros((len(table), len(specs)))
    for j, (col, _, fmt) in enumerate(specs):
        if fmt != "task":
            arr = table[col].fillna(0).to_numpy(dtype=float)
            values[:, j] = arr / max(arr.max(), 1)
    fig = plt.figure(figsize=style.OVERVIEW_SIZE, layout="constrained")
    grid = fig.add_gridspec(3, 1, height_ratios=style.OVERVIEW_ROW_RATIOS)
    header, ax, footer = [fig.add_subplot(grid[i]) for i in range(3)]
    for annotation_ax in (header, footer):
        annotation_ax.set_axis_off()
    header.text(
        0,
        0.8,
        "20 core datasets: text availability, repetition and Jev workload",
        transform=header.transAxes,
        fontsize=style.OVERVIEW_TITLE_FONT,
        va="top",
    )
    header.text(
        0,
        0.3,
        "Darker shading means a larger value within that column. Counts and percentages are printed explicitly.",
        transform=header.transAxes,
        fontsize=style.OVERVIEW_FONT,
        va="top",
    )
    ax.imshow(values, cmap=style.OVERVIEW_CMAP, vmin=0, vmax=2, aspect="auto")
    for i, (_, row) in enumerate(table.iterrows()):
        for j, (col, _, fmt) in enumerate(specs):
            value = row[col]
            if fmt == "task":
                label = TASK_LABELS[value]
                if row.n_classes > 0:
                    label += f"\n{row.n_classes:.0f} classes"
            elif fmt.endswith("pct"):
                label = f"{value:,.0f}\n{row[fmt]:.1f}%"
            elif fmt == "tail":
                label = f"{value:,.1f}\n{row.combined_tokens_max:,.0f}"
            elif fmt == "float":
                label = f"{value:,.1f}"
            else:
                label = f"{value:,.0f}"
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
    labels = [
        f"{row.dataset} [REMOVED]"
        if getattr(row, "experiment_status", "Included") == "REMOVED"
        else row.dataset
        for row in table.itertuples()
    ]
    ax.set_yticks(range(len(table)), labels, fontsize=style.OVERVIEW_FONT)
    ax.set_xticks(np.arange(-0.5, len(specs), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(table), 1), minor=True)
    ax.grid(which="minor", color=style.GRID_COLOR, linewidth=0.7)
    ax.tick_params(which="minor", length=0)
    notes = (
        "[REMOVED] = audit-only: Zomato (extreme text burden) and Spotify (114-class representation size); all 20 remain in exploration.\n"
        "Text cells = rows × text columns. Distinct texts are counted within each named column, then summed. Empty = null or whitespace-only.\n"
        "Empty-cell percentages use all text cells; distinct percentages use nonempty text cells. Joint reuse uses rows with text.\n"
        "Both-mode savings include within-mode duplicates and matching per-column/joint requests. Means exclude empty text cells; joint token summaries include all rows.\n"
        f"Tokens ≈ ceil(characters / {characters_per_token:g}); task/question overhead is excluded. The p95/max column highlights long inputs needing a policy."
    )
    footer.text(
        0,
        0.95,
        notes,
        transform=footer.transAxes,
        fontsize=style.OVERVIEW_FONT,
        va="top",
        linespacing=1.5,
    )
    caption = (
        "Text-focused overview of all 20 core datasets; [REMOVED] identifies Spotify and Zomato as excluded from extraction and experiments, with raw data retained. Dataset/task dimensions are followed by text-cell availability, "
        "distinct inputs and avoided requests, mean nonempty text length, joint-input availability and reuse, "
        "joint token median/p95/maximum, and total unique requests and reuse across both modes. "
        "Percentages use the denominators stated below the table; distinct texts are counted per named field. "
        "Joint token summaries include empty rows. Shading is scaled independently per column. Token counts "
        "are character-based estimates excluding questions and metadata. This is a large-format inspection figure."
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
