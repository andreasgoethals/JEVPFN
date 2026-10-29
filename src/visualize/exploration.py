"""Measured dataset overview and text diagnostics; no model inference."""

import matplotlib.pyplot as plt
import numpy as np

from src.visualize import layout, style


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


def _removed():
    from src.utils.config import load_config

    return set(load_config("feature_creation")["dataset_selection"]["exclude"])


def overview_figures(table, characters_per_token=4.0):
    """All 20 datasets on one page; shading ranks each column separately."""
    from matplotlib.colors import to_rgb

    specs = [
        ("rows", "Rows", "{:,.0f}", True),
        ("text_columns", "Text\ncolumns", "{:.0f}", False),
        ("non_text_columns", "Non-text\ncolumns", "{:.0f}", False),
        ("n_classes", "Classes\nK", "{:.0f}", False),
        ("per_column_missing_or_empty_pct", "Empty text\ncells (%)", "{:.1f}", False),
        ("mean_tokens_per_nonempty_text", "Tokens per\ntext cell", "{:,.1f}", True),
        ("distinct_text_pct", "Distinct\ntexts (%)", "{:.0f}", False),
        ("combined_unique_inputs", "Unique\nrequests", "{:,.0f}", True),
        ("combined_reuse_savings_pct", "Slots\nreused (%)", "{:.0f}", False),
    ]
    table = table.reset_index(drop=True)
    removed = table.experiment_status.eq("REMOVED").to_numpy()
    shades = []
    for column, _, _, log in specs:
        values = table[column].fillna(0).to_numpy(dtype=float)
        values = np.log10(np.maximum(values, 1)) if log else values
        kept = values[~removed]
        span = max(kept.max() - kept.min(), 1e-9)
        shades.append(np.clip((values - kept.min()) / span, 0, 1))
    shades = np.column_stack(shades)
    base = np.array(to_rgb(style.OVERVIEW_TINT))
    rgb = 1 - shades[..., None] * (1 - base) * 0.75
    rgb[removed] = to_rgb(style.FAINT_COLOR)
    fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, 4.55))
    ax.imshow(rgb, aspect="auto", interpolation="nearest")
    for i, row in table.iterrows():
        for j, (column, _, pattern, _) in enumerate(specs):
            value = row[column]
            text = "–" if column == "n_classes" and not value > 0 else pattern.format(value)
            ax.text(j, i, text, ha="center", va="center", fontsize=6.5,
                    color=style.REMOVED_COLOR if removed[i] else style.TEXT_COLOR)
    labels = list(table.dataset)
    layout.dataset_axis(ax, labels, table.task_type,
                        removed=set(table.dataset[removed]), fontsize=7)
    ax.set_xticks(range(len(specs)), [label for _, label, _, _ in specs], fontsize=7)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0)
    for position in range(1, len(specs)):
        ax.axvline(position - 0.5, color="white", linewidth=1.2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.legend(handles=layout.task_handles(), loc="outside lower center", ncols=3, fontsize=7,
               frameon=False)
    yield fig, "dataset_overview", (
        "The 20 core MulTaBench text datasets, grouped by task. Empty cells are null, nonfinite "
        "or whitespace-only. Tokens per text cell exclude empty cells and use "
        f"ceil(characters/{characters_per_token:g}). Distinct texts count exact duplicates within "
        "a column. Unique requests and reused slots combine per-column and joint inputs. Shading "
        "ranks each column across the 18 retained datasets (log scale for counts and tokens). "
        "Grey rows (Spotify, Zomato) are audited but excluded from extraction and benchmarks."
    )


def text_column_figures(audit):
    """Length, emptiness and exact repetition of every official text column."""
    order = layout.ordered(audit.summary)
    removed = _removed()
    data = audit.text.merge(order[["dataset_id", "dataset", "task_type"]], on="dataset_id")
    positions = {name: i for i, name in enumerate(order.dataset)}
    panels = [
        ("nonempty_characters_mean", "Mean characters per\nnonempty cell (log)", True, None),
        ("missing_or_empty_pct", "Empty cells (%)", False, (-3, 103)),
        ("unique_nonempty_ratio", "Distinct texts among\nnonempty cells (%)", False, (-3, 103)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(style.WIDTH_FULL, 4.5), sharey=True,
                             gridspec_kw={"width_ratios": [1.25, 1, 1]})
    for dataset, group in data.groupby("dataset", sort=False):
        offsets = np.linspace(-0.28, 0.28, len(group)) if len(group) > 1 else np.zeros(1)
        y = positions[dataset] + offsets
        color = (style.REMOVED_COLOR if group.dataset_id.iloc[0] in removed
                 else style.TASK_COLORS[group.task_type.iloc[0]])
        for ax, (column, _, _, _) in zip(axes, panels):
            values = group[column] * (100 if column == "unique_nonempty_ratio" else 1)
            ax.scatter(values, y, s=10, color=color, linewidths=0, alpha=0.85)
    for ax, (_, label, log, limits) in zip(axes, panels):
        ax.set_xlabel(label)
        if log:
            ax.set_xscale("log")
        if limits:
            ax.set_xlim(*limits)
        layout.light_grid(ax)
        layout.task_rules(ax, order.task_type)
    layout.dataset_axis(axes[0], list(order.dataset), order.task_type,
                        removed={d for i, d in zip(order.dataset_id, order.dataset) if i in removed},
                        fontsize=7, separators=False)
    yield fig, "text_field_profile", (
        f"Each point is one of the {len(data)} official text columns; columns of a dataset are "
        "offset vertically. Left: mean character count of nonempty cells (logarithmic). Middle: "
        "share of rows whose cell is null, nonfinite or whitespace-only; such cells produce no "
        "request and missing features. Right: exact distinct values as a share of nonempty "
        "cells; low values mean a field behaves like a categorical string and most requests are "
        "reused. Grey points belong to the two excluded datasets."
    )


def reuse_figure(summary, long_datasets):
    """Per-row text volume, total volume and exact request reuse on one dataset axis."""
    order = layout.ordered(summary).merge(
        long_datasets.drop(columns=["dataset", "text_columns"]), on="dataset_id", how="left"
    )
    removed = _removed()
    is_removed = order.dataset_id.isin(removed).to_numpy()
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 3, figsize=(style.WIDTH_FULL, 4.5), sharey=True,
                             gridspec_kw={"width_ratios": [1.45, 0.8, 1]})
    ink = np.where(is_removed, style.REMOVED_COLOR, style.MEDIAN_COLOR)
    ax = axes[0]
    ax.hlines(y, order.joint_tokens_p01, order.joint_tokens_p99, color=ink, linewidth=0.8)
    for i, row in enumerate(order.itertuples()):
        ax.add_patch(plt.Rectangle((row.joint_tokens_p25, i - 0.28),
                                   row.joint_tokens_p75 - row.joint_tokens_p25, 0.56,
                                   facecolor=style.FAINT_COLOR, edgecolor=ink[i], linewidth=0.8))
    ax.vlines(order.joint_tokens_median, y - 0.28, y + 0.28, color=ink, linewidth=1.4)
    ax.scatter(order.joint_tokens_max, y, marker="D", s=9, color=style.P95_COLOR, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("Text tokens per row, all fields\n(box p25–p75, line p01–p99)")
    axes[1].scatter(order.total_text_tokens / 1e6, y, s=12, color=ink, zorder=3)
    axes[1].hlines(y, 1e-3, order.total_text_tokens / 1e6, color=style.GRID_COLOR, linewidth=0.8)
    axes[1].set_xscale("log")
    axes[1].set_xlim(0.03, order.total_text_tokens.max() / 1e6 * 3)
    axes[1].set_xticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
    axes[1].set_xlabel("Dataset text volume\n(million tokens, log)")
    ax = axes[2]
    for mode, column in (("per_column", "per_column_reuse_savings_pct"),
                         ("joint", "joint_reuse_savings_pct")):
        ax.scatter(order[column], y, s=14, color=style.MODE_COLORS[mode], zorder=3,
                   label={"per_column": "Per-column request", "joint": "Joint request"}[mode])
    ax.hlines(y, order.joint_reuse_savings_pct, order.per_column_reuse_savings_pct,
              color=style.GRID_COLOR, linewidth=1.2)
    ax.set_xlim(-3, 103)
    ax.set_xlabel("Nonempty request slots\nserved by exact reuse (%)")
    handles = [*ax.get_legend_handles_labels()[0],
               plt.Line2D([], [], linestyle="none", marker="D", markersize=3.5,
                          color=style.P95_COLOR, label="Longest row")]
    fig.legend(handles=handles, loc="outside upper center", ncols=3, fontsize=7, frameon=False)
    for axis in axes:
        layout.light_grid(axis)
        layout.task_rules(axis, order.task_type)
    layout.dataset_axis(axes[0], list(order.dataset), order.task_type,
                        removed=set(order.dataset[is_removed]), fontsize=7, separators=False)
    return fig, "text_volume_and_reuse", (
        "Left: distribution of the combined text length of each row (all named fields, empty "
        "fields omitted; tokens approximated as ceil(characters/4)). Middle: total text volume "
        "of the dataset before reuse. Right: share of nonempty request slots that repeat an "
        "earlier identical request and therefore need no new inference, separately for "
        "per-column and joint inputs. Grey rows are the two excluded datasets."
    )
