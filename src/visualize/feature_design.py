"""Visual explanations and measured feature-build dimensions; no invented Jev outputs."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from src.visualize import style

MODE_LABELS = {"per_column": "Per-column", "joint": "Joint", "per_column_and_joint": "Both"}


def benchmark_configurations_figure(table):
    """Binary design matrix: no performance measurements or invented results."""
    from matplotlib.colors import ListedColormap

    columns = ["Non-text", "E5 / PCA", "Joint", "Per-column"]
    values = table[columns].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=style.figsize(ratio=style.AUDIT_RATIO))
    ax.imshow(
        values,
        cmap=ListedColormap([style.BACKGROUND_COLOR, style.MEDIAN_COLOR]),
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    ax.set_xticks(range(len(columns)), columns)
    ax.set_yticks(range(len(table)), table.configuration)
    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(table), 1), minor=True)
    ax.grid(which="minor", color=style.GRID_COLOR, linewidth=1)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0)
    for (i, j), included in np.ndenumerate(values):
        ax.text(
            j,
            i,
            "Yes" if included else "–",
            ha="center",
            va="center",
            color=style.BACKGROUND_COLOR if included else style.TEXT_COLOR,
        )
    ax.set_title("Eight feature configurations for each of six predictors")
    return (
        fig,
        "benchmark_feature_configurations",
        (
            "Feature inclusion in the eight main benchmark configurations. Non-text includes all "
            "original numerical and categorical columns. E5 is encoded separately per text column, "
            "with PCA fitted only on each training partition (32 output columns per text column, "
            "plus a missingness flag). Joint and per-column decision features use one extractor "
            "at a time. The two configurations without decision features are fitted once per "
            "dataset and predictor; the other six are repeated for each selected extractor. "
            "TabSTAR and ConTextTab are separate references using all original features."
        ),
    )


def input_design_figure():
    fig, ax = plt.subplots(figsize=style.COMPACT_FLOW_SIZE)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    boxes = [
        (0.14, 0.80, "Text column A", style.MEDIAN_COLOR),
        (0.14, 0.55, "Text column B", style.MEDIAN_COLOR),
        (0.14, 0.30, "A + B text", style.P95_COLOR),
        (0.50, 0.80, "Fixed question\n+ A → model", style.MEDIAN_COLOR),
        (0.50, 0.55, "Fixed question\n+ B → model", style.MEDIAN_COLOR),
        (0.50, 0.30, "Fixed question\n+ A + B → model", style.P95_COLOR),
        (0.86, 0.80, "A features", style.MEDIAN_COLOR),
        (0.86, 0.55, "B features", style.MEDIAN_COLOR),
        (0.86, 0.30, "Joint features", style.P95_COLOR),
    ]
    for x, y, text, color in boxes:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=8,
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": style.BACKGROUND_COLOR,
                "edgecolor": color,
            },
        )
    for y in (0.80, 0.55, 0.30):
        for start, end in ((0.26, 0.36), (0.65, 0.76)):
            ax.annotate(
                "",
                xy=(end, y),
                xytext=(start, y),
                arrowprops={"arrowstyle": "->", "color": style.TOTAL_COLOR},
            )
    ax.set_title(
        "Separate and joint text inputs"
    )
    ax.text(
        0.5,
        0.02,
        "Shared task metadata; no non-text features or row targets.\nOutputs per input: one binary probability, K class probabilities, or one direction score.",
        ha="center",
        fontsize=7,
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
