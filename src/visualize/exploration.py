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


def overview_figures(table, characters_per_token=4.0):
    """Two print-sized overview panels; the full audit remains in the notebook table."""
    from textwrap import fill

    specs = [("rows", "Rows"), ("text_columns", "Text /\nnon-text"),
             ("per_column_missing_or_empty_pct", "Empty\ncells %"),
             ("mean_tokens_per_nonempty_text", "Mean tokens\n/ text"),
             ("combined_unique_inputs", "Unique\nrequests"),
             ("combined_avoided_requests", "Avoided\nN / %")]
    for page, start in enumerate(range(0, len(table), 10), 1):
        part = table.iloc[start:start + 10]
        values = np.column_stack([part[col].fillna(0).to_numpy(dtype=float) /
                                  max(table[col].max(), 1) for col, _ in specs])
        fig, ax = plt.subplots(figsize=style.figsize(ratio=style.AUDIT_RATIO))
        ax.imshow(values, cmap=style.OVERVIEW_CMAP, vmin=0, vmax=1.8, aspect="auto")
        for i, row in enumerate(part.itertuples()):
            texts = [f"{row.rows:,}", f"{row.text_columns} / {row.non_text_columns}",
                     f"{row.per_column_missing_or_empty_pct:.1f}",
                     f"{row.mean_tokens_per_nonempty_text:,.1f}",
                     f"{row.combined_unique_inputs:,}",
                     f"{row.combined_avoided_requests:,}\n{row.combined_reuse_savings_pct:.1f}%"]
            for j, text in enumerate(texts):
                ax.text(j, i, text, ha="center", va="center", fontsize=7)
        labels = [fill(r.dataset, 23) + "\n" + TASK_LABELS[r.task_type] +
                  (f" · K={r.n_classes:.0f}" if r.n_classes > 0 else "") +
                  (" · removed" if r.experiment_status == "REMOVED" else "")
                  for r in part.itertuples()]
        ax.set_yticks(range(len(part)), labels, fontsize=7)
        for label, removed in zip(ax.get_yticklabels(), part.experiment_status.eq("REMOVED")):
            if removed:
                label.set_color(style.REMOVED_COLOR)
                label.set_fontstyle("italic")
        ax.set_xticks(range(len(specs)), [label for _, label in specs], fontsize=7)
        ax.xaxis.tick_top()
        ax.tick_params(which="both", length=0)
        ax.set_title(f"Text extraction overview · {page}/2", pad=12)
        yield fig, f"dataset_overview_{page}", (
            "Core text datasets in catalog order. Counts exclude the target. Empty cells are null or whitespace-only. "
            "Mean tokens exclude empty cells. Unique requests combine separate and joint inputs after exact reuse; "
            "avoided requests are shown as a count and percentage of nonempty slots. K denotes target classes. "
            "Spotify and Zomato are marked removed from experiments. Shading scales each metric independently. "
            f"Tokens use ceil(characters/{characters_per_token:g}), excluding request overhead. Full counts and quantiles appear in the companion table."
        )


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
            "colours identify task types."
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
    ax.set_title("Field length and exact-input uniqueness")
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
        ("per_column", -0.17, style.MODE_COLORS["per_column"], "Per-column"),
        ("joint", 0.17, style.MODE_COLORS["joint"], "Joint"),
    ]:
        for ax, suffix in zip(axes, ["reuse_savings_pct", "repeated_inputs"]):
            ax.barh(y + offset, ordered[f"{mode}_{suffix}"], height=0.32, color=color, label=label)
    axes[0].set_yticks(y, ordered.dataset)
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("Nonempty request slots avoided (%)")
    axes[1].set_xlabel("Nonempty request slots avoided (absolute count)")
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    axes[1].tick_params(axis="x", labelrotation=25)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncols=2)
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}k"))
    axes[1].tick_params(axis="x", labelrotation=0)
    return (
        fig,
        "exact_text_reuse",
        (
            "Exact reuse savings for per-column and joint requests on all 20 datasets. Left: "
            "avoided requests divided by nonempty slots, as percentages. Right: the same avoided "
            "requests as absolute counts. Missing inputs and cross-mode overlap are excluded. "
            "Axes start at zero."
        ),
    )
