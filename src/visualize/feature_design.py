"""Visual explanations and measured feature-build dimensions; no invented Jev outputs."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from src.visualize import style

MODE_LABELS = {"per_column": "Per-column", "joint": "Joint", "per_column_and_joint": "Both"}


def input_design_figure():
    fig, ax = plt.subplots(figsize=style.FLOW_SIZE)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    boxes = [
        (0.12, 0.75, "Text column A", style.MEDIAN_COLOR),
        (0.12, 0.5, "Text column B", style.MEDIAN_COLOR),
        (0.12, 0.22, "Text columns A + B", style.P95_COLOR),
        (0.48, 0.75, "Fixed question + A\n→ Jev request", style.MEDIAN_COLOR),
        (0.48, 0.5, "Fixed question + B\n→ Jev request", style.MEDIAN_COLOR),
        (0.48, 0.22, "Same question + A + B\n→ Jev request", style.P95_COLOR),
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
        "Task metadata is shared. Non-text features and row targets are excluded. Both = concatenate the saved outputs.",
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


def task_outputs_figure():
    fig, axes = plt.subplots(1, 3, figsize=style.FLOW_SIZE)
    descriptions = [
        (
            "Binary · Noul",
            "One probability\np(positive class), from 0 to 1\n\n1 compact feature per input",
            style.MEDIAN_COLOR,
        ),
        (
            "Multiclass · Choice",
            "One probability per class\np₁, p₂, …, pK\n\nK compact features per input",
            style.P95_COLOR,
        ),
        (
            "Regression · Score",
            "Nine probabilities: p(−4), …, p(+4)\nExpected direction = Σ level × p(level)\n\n1 compact feature, or 9 probabilities",
            style.TOTAL_COLOR,
        ),
    ]
    for ax, (title, text, color) in zip(axes, descriptions):
        ax.set_axis_off()
        ax.set_title(title, color=color)
        ax.text(0.5, 0.6, text, ha="center", va="center", transform=ax.transAxes, linespacing=2)
    fig.suptitle("Numeric output shapes · compact views are proposals; full responses are retained")
    return (
        fig,
        "task_specific_feature_shapes",
        (
            "Schematic of proposed numeric Jev feature views by task. Choice and Score responses also "
            "retain their documented confidence values. No probability values have been simulated or "
            "measured. The regression expectation is a directional score, not a prediction in target units."
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
            "Nonempty request slots and distinct complete inputs across the 20 core text datasets. "
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
        for ax, col in zip(axes, ["proposed_compact_numeric_columns", "retained_numeric_columns"]):
            ax.barh(
                np.arange(len(table)) + offset,
                table[col],
                height=0.32,
                color=color,
                label=MODE_LABELS[mode],
            )
    axes[0].set_yticks(range(len(datasets)), datasets)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Numeric columns in the proposed compact view")
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


def workload_figure(totals, *, deduplicated=False):
    token_col = "wire_tokens_combined_approx" if deduplicated else "input_tokens_approx"
    cost_col = "cost_usd" if deduplicated else "input_cost_estimate"
    fig, axes = plt.subplots(1, 2, figsize=style.FLOW_SIZE)
    labels = (
        [f"{n:g} chars/token" for n in totals.characters_per_token]
        if deduplicated
        else [MODE_LABELS.get(m, m) for m in totals["mode"]]
    )
    axes[0].barh(labels, totals[token_col] / 1e6, color=style.MEDIAN_COLOR)
    axes[0].set_xlabel("Approximate input tokens (millions)")
    if totals[cost_col].notna().all():
        axes[1].barh(labels, totals[cost_col], color=style.P95_COLOR)
        for i, cost in enumerate(totals[cost_col]):
            axes[1].text(cost, i, f" ${cost:,.2f}", va="center")
        axes[1].set_xlim(0, totals[cost_col].max() * 1.25)
        axes[1].set_xlabel("Estimated input cost (USD; configured price)")
    else:
        axes[1].set_axis_off()
        axes[1].text(0.5, 0.5, "Set a verified input price to display costs.", ha="center")
    scope = "after exact reuse" if deduplicated else "before exact reuse"
    fig.suptitle(
        f"Workload {scope} · {'both modes; tokenizer sensitivity' if deduplicated else 'three alternative modes'}"
    )
    return (
        fig,
        "deduplicated_budget" if deduplicated else "logical_workload_budget",
        (
            f"Approximate input tokens and configured-price costs {scope}. "
            + (
                "Bars compare 3, 4 and 5 characters per token for both input modes with exact reuse. "
                if deduplicated
                else "Bars compare per-column, joint and both input modes. "
            )
            + "Both is the complete alternative, not a third workload to add to the others. "
            "Character-based estimates include request metadata; unknown billing overhead, retries "
            "and unresolved over-limit input handling prevent interpreting these as a quote."
        ),
    )
