"""Visual explanations and measured feature-build dimensions; no invented Jev outputs."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from src.visualize import style

MODE_LABELS = {"per_column": "Per-column", "joint": "Joint", "per_column_and_joint": "Both"}


def input_design_figure():
    fig, ax = plt.subplots(figsize=style.COMPACT_FLOW_SIZE)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    boxes = [
        (0.12, 0.75, "Text column A", style.MEDIAN_COLOR),
        (0.12, 0.5, "Text column B", style.MEDIAN_COLOR),
        (0.12, 0.22, "Text columns A + B", style.P95_COLOR),
        (0.48, 0.75, "Fixed question + A\n→ decision model", style.MEDIAN_COLOR),
        (0.48, 0.5, "Fixed question + B\n→ decision model", style.MEDIAN_COLOR),
        (0.48, 0.22, "Same question + A + B\n→ decision model", style.P95_COLOR),
        (0.84, 0.75, "Features from A", style.MEDIAN_COLOR),
        (0.84, 0.5, "Features from B", style.MEDIAN_COLOR),
        (0.84, 0.22, "Joint features", style.P95_COLOR),
    ]
    for x, y, text, color in boxes:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.8",
                "facecolor": style.BACKGROUND_COLOR,
                "edgecolor": color,
            },
        )
    for y in (0.75, 0.5, 0.22):
        for start, end in ((0.22, 0.36), (0.60, 0.74)):
            ax.annotate(
                "",
                xy=(end, y),
                xytext=(start, y),
                arrowprops={"arrowstyle": "->", "color": style.TOTAL_COLOR},
            )
    ax.set_title(
        "Two text columns → three input groups before empty-input skipping and exact reuse"
    )
    ax.text(
        0.5,
        0.02,
        "Task metadata is shared. Non-text features and row targets are excluded. Each model: 1 binary probability, K class probabilities, or 1 expected direction per input.",
        ha="center",
    )
    return (
        fig,
        "per_column_and_joint_inputs",
        (
            "Conceptual request construction for a row with two nonempty text columns. Two separate "
            "inputs and one joint input use the same deterministic task question. Each distinct complete "
            "request is cached once. This diagram shows the design, not executed API calls or outputs."
        ),
    )


def request_counts_figure(summary):
    before = [
        summary.per_column_nonempty_inputs.sum(),
        summary.joint_nonempty_inputs.sum(),
        summary.combined_nonempty_inputs.sum(),
    ]
    after = [
        summary.per_column_unique_inputs.sum(),
        summary.joint_unique_inputs.sum(),
        summary.combined_unique_inputs.sum(),
    ]
    fig, ax = plt.subplots(figsize=style.figsize())
    x = np.arange(3)
    ax.bar(x - 0.2, before, width=0.4, color=style.P95_COLOR, label="Nonempty slots")
    ax.bar(x + 0.2, after, width=0.4, color=style.MEDIAN_COLOR, label="Unique requests")
    for i, (total, unique) in enumerate(zip(before, after)):
        ax.text(
            i,
            total,
            f"{total - unique:,} avoided\n({100 * (total - unique) / total:.1f}%)",
            ha="center",
            va="bottom",
        )
    ax.set_xticks(x, ["Per-column", "Joint", "Both (shared cache)"])
    ax.set_ylabel("Request slots / unique requests")
    ax.set_ylim(0, max(before) * 1.3)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1e6:g}M"))
    ax.legend(loc="upper left")
    return (
        fig,
        "requests_before_and_after_reuse",
        (
            f"Nonempty request slots and distinct complete inputs across the {len(summary)} inspected text datasets. "
            "Labels show avoided calls in absolute counts and percentages. Both modes use a shared "
            "cache, including single-available-field overlap; the three alternatives must not be added."
        ),
    )


def feature_dimensions_figure(plan):
    fig, axes = plt.subplots(1, 2, figsize=style.DASHBOARD_SIZE, sharey=True)
    datasets = plan.dataset.drop_duplicates().tolist()
    for mode, offset, color in [
        ("per_column", -0.17, style.MEDIAN_COLOR),
        ("joint", 0.17, style.P95_COLOR),
    ]:
        table = plan[plan["mode"] == mode].set_index("dataset").loc[datasets]
        for ax, col in zip(axes, ["main_numeric_columns", "retained_numeric_columns"]):
            ax.barh(
                np.arange(len(table)) + offset,
                table[col],
                height=0.32,
                color=color,
                label=MODE_LABELS[mode],
            )
    axes[0].set_yticks(range(len(datasets)), datasets)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Numeric columns in the agreed main view")
    axes[1].set_xlabel("Numeric columns retained in the cached table")
    axes[0].legend(loc="lower right")
    return (
        fig,
        "numeric_feature_dimensions",
        (
            "Planned numeric column counts for per-column and joint tables. Compact views use one "
            "binary probability, K multiclass probabilities, or one expected regression direction per "
            "input. Retained views additionally keep Choice confidence, or all nine Score probabilities "
            "and confidence. These are schema dimensions, not generated features or API request counts."
        ),
    )


def workload_figure(totals):
    """One axis for token sensitivity; costs label the same bars, not a duplicate panel."""
    fig, ax = plt.subplots(figsize=style.figsize())
    tokens = totals.wire_tokens_combined_approx / 1e6
    bars = ax.barh(
        [f"{n:g} chars/token" for n in totals.characters_per_token],
        tokens,
        color=style.MEDIAN_COLOR,
    )
    labels = [
        f"{v:,.1f}M tokens" + (f" · ${cost:,.2f}" if np.isfinite(cost) else "")
        for v, cost in zip(tokens, totals.cost_usd.fillna(np.nan))
    ]
    ax.bar_label(bars, labels=labels, padding=4)
    ax.set_xlim(0, tokens.max() * 1.55)
    ax.set_xlabel("Approximate input tokens (millions)")
    ax.set_title("Jev cost sensitivity after exact reuse · both input modes")
    return (
        fig,
        "deduplicated_budget",
        (
            "Character-based input-token scenarios for both representations after exact reuse. "
            "Each bar is labelled with its token volume and configured-price cost. These are "
            "alternative estimates, not additive workloads or provider billing measurements. "
            "Metadata and question text are included; unknown billing overhead and retries are excluded."
        ),
    )
