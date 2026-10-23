"""Length tails at dataset level, plus the columns with the largest p99 lengths."""

from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np

from src.visualize import style


def _box(row, prefix, label):
    return {
        "label": label,
        "med": row[f"{prefix}_median"],
        "q1": row[f"{prefix}_p25"],
        "q3": row[f"{prefix}_p75"],
        "whislo": row[f"{prefix}_p01"],
        "whishi": row[f"{prefix}_p99"],
        "fliers": [],
    }


def _draw(ax, rows, prefix, labels):
    keep = rows[f"{prefix}_median"].notna().to_numpy()
    rows = rows.loc[keep]
    labels = np.asarray(labels)[keep]
    if rows.empty:
        ax.text(0.5, 0.5, "No nonempty text", transform=ax.transAxes, ha="center")
        ax.set_axis_off()
        return
    records = rows.to_dict("records")
    boxes = [_box(r, prefix, label) for r, label in zip(records, labels)]
    ax.bxp(
        boxes,
        orientation="horizontal",
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": style.MEDIAN_COLOR, "alpha": 0.35},
        medianprops={"color": style.TEXT_COLOR},
        manage_ticks=True,
    )
    ax.scatter(
        [r[f"{prefix}_max"] for r in records],
        np.arange(1, len(records) + 1),
        color=style.P95_COLOR,
        marker="D",
        s=style.AUDIT_MARKER_SIZE,
        label="Maximum",
    )
    ax.set_xscale("log")
    ax.set_ylim(len(records) + 0.5, 0.5)  # Idempotent with shared y axes.
    ax.grid(axis="x", color=style.GRID_COLOR, alpha=0.5)
    ax.tick_params(axis="y", labelsize=style.LONG_TEXT_LABEL_FONT)


def long_text_figures(datasets, columns, *, all_columns=False):
    """The full column table is retained; the main view prioritizes the 12 longest p99s.

    all_columns=True is an explicit appendix view, with the same selection-independent
    axis and no new measurements. Neither view filters extraction or benchmark rows.
    """
    xmax = float(max(datasets.cell_tokens_max.max(), datasets.joint_tokens_max.max())) * 1.5
    for page, start in enumerate(range(0, len(datasets), 10), 1):
        part = datasets.iloc[start:start + 10]
        fig, axes = plt.subplots(1, 2, figsize=style.LONG_TEXT_SIZE, sharey=True)
        _draw(axes[0], part, "cell_tokens", part.dataset)
        _draw(axes[1], part, "joint_tokens", part.dataset)
        axes[0].set_title("Individual text cells")
        axes[1].set_title("All text in each row")
        for ax in axes:
            ax.set(xlabel="Approximate text tokens (log scale)", xlim=(0.7, xmax))
        fig.legend(*axes[1].get_legend_handles_labels(), loc="outside lower center")
        fig.suptitle(f"Text-length tails · catalog panel {page}")
        yield (
            fig,
            f"long_text_datasets_{page}",
            f"Nonempty text lengths for datasets {start + 1}–{min(start + 10, len(datasets))} "
            f"of {len(datasets)} in catalog order: pooled individual cells (left), joint row text "
            "(right). Boxes span p25–p75, with median lines and p01–p99 whiskers; diamonds mark "
            "maxima. Both catalog panels share the same logarithmic axis. Tokens use "
            "ceil(characters/4); these are text screens, not complete-request context checks.",
        )
    ordered = columns.sort_values(["tokens_p99", "dataset", "text_column"], ascending=[False, True, True])
    if not all_columns:
        ordered = ordered.head(12)
    pages = [ordered.iloc[i:i + 12] for i in range(0, len(ordered), 12)]
    xmax = float(columns.tokens_max.max()) * 1.5
    for page, part in enumerate(pages, start=1):
        fig, ax = plt.subplots(figsize=style.LONG_TEXT_PAGE_SIZE)
        labels = [f"{row.dataset}\n{fill(row.text_column, 29)}" for row in part.itertuples()]
        _draw(ax, part, "tokens", labels)
        ax.tick_params(axis="y", labelsize=style.LONG_TEXT_PAGE_FONT)
        ax.set_xlim(0.7, xmax)
        active = part[part.tokens_median.notna()]
        for position in range(1, len(active)):
            if active.dataset.iloc[position] != active.dataset.iloc[position - 1]:
                ax.axhline(position + 0.5, color=style.GRID_COLOR, linewidth=0.6)
        ax.set_xlabel("Approximate nonempty text tokens (log scale)")
        ax.set_title(f"Official text columns · page {page}" if all_columns else
                     "The 12 text columns with the largest p99 lengths")
        fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncols=2)
        yield (
            fig,
            f"long_text_columns_page_{page}" if all_columns else "longest_column_tails",
            ("All official text columns, ordered by descending p99 length. " if all_columns else
             "Twelve official text columns selected by descending nonempty p99 length, independent of targets. ") +
            "Dataset and column names label each row. Boxes show p25–p75 and medians; whiskers "
            "span p01–p99; diamonds mark maxima. All pages share the same logarithmic axis. "
            "The tables retain all columns and exact p99/max values. No provider context limit "
            "is inferred from character counts; no input is removed or shortened.",
        )
