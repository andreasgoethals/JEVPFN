"""Readable boxplots with p99 whiskers and separately marked observed maxima."""

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


def _draw(ax, rows, prefix, labels, limit):
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
    ax.axvline(
        limit, color=style.MISSING_COLOR, linestyle="--", label=f"{limit:,} text-token screen"
    )
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.grid(axis="x", color=style.GRID_COLOR, alpha=0.5)
    ax.tick_params(axis="y", labelsize=style.LONG_TEXT_LABEL_FONT)


def long_text_figures(datasets, columns, limit):
    fig, axes = plt.subplots(1, 2, figsize=style.LONG_TEXT_SIZE, sharey=True)
    _draw(axes[0], datasets, "cell_tokens", datasets.dataset, limit)
    _draw(axes[1], datasets, "joint_tokens", datasets.dataset, limit)
    axes[0].set_title("Individual nonempty text cells")
    axes[1].set_title("All available text in each nonempty row")
    for ax in axes:
        ax.set_xlabel("Approximate text tokens (log scale)")
    axes[1].legend(loc="lower right")
    fig.suptitle("Long-text overview: distributions and extreme inputs")
    yield (
        fig,
        "long_text_datasets",
        (
            "Nonempty text lengths across all 20 core datasets, for pooled individual cells (left) "
            "and joint row text (right). Boxes span p25–p75 with median lines; whiskers are p01 and p99, "
            "not Tukey whiskers. Diamonds mark observed maxima. Tokens use ceil(characters/4). "
            "The dashed 32,000-token screen excludes request metadata/questions and JSON escaping."
        ),
    )
    ordered = columns.reset_index(drop=True)
    pages = np.array_split(np.arange(len(ordered)), 2)
    xmax = max(float(ordered.tokens_max.max()), limit) * 1.3
    for page, indices in enumerate(pages, start=1):
        part = ordered.iloc[indices]
        fig, ax = plt.subplots(figsize=style.LONG_TEXT_PAGE_SIZE)
        labels = part.dataset + " · " + part.text_column
        _draw(ax, part, "tokens", labels, limit)
        ax.tick_params(axis="y", labelsize=style.LONG_TEXT_PAGE_FONT)
        ax.set_xlim(0.7, xmax)
        active = part[part.tokens_median.notna()]
        for position in range(1, len(active)):
            if active.dataset.iloc[position] != active.dataset.iloc[position - 1]:
                ax.axhline(position + 0.5, color=style.GRID_COLOR, linewidth=0.6)
        ax.set_xlabel("Approximate nonempty text tokens (log scale)")
        ax.set_title(f"All text columns — page {page} of 2")
        fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncols=2)
        yield (
            fig,
            f"long_text_columns_page_{page}",
            "Nonempty text lengths for half of the 90 official text columns, in catalog order. "
            "Dataset and column names label each row. Boxes show p25–p75 and medians; whiskers "
            "span p01–p99; diamonds mark maxima. The two pages share the same logarithmic axis. "
            "The dashed line marks 32,000 approximate text tokens. Exact p99/max values remain in the tables.",
        )
