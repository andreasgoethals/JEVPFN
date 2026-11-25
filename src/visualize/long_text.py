"""The longest text columns against the encoder and decision-model context windows."""

from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np

from src.visualize import layout, style


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
        boxprops={"facecolor": style.FAINT_COLOR, "edgecolor": style.MEDIAN_COLOR},
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
    ax.tick_params(axis="y", labelsize=style.LONG_TEXT_LABEL_FONT)


def context_windows():
    """Reference windows: frozen encoders and the smallest/hosted decision-model contexts."""
    from src.utils.config import load_yaml

    encoders = load_yaml("feature_creation/embedding_models")["models"]
    candidates = load_yaml("extractors")["candidates"]
    smallest = min(candidates, key=lambda provider: candidates[provider]["context_tokens"])
    encoder = style.MODE_COLORS["per_column"]
    decision = style.WINDOW_COLOR
    return [
        (encoders["e5_small_v2"]["max_tokens"], "E5 chunk", encoder, ":"),
        (encoders["qwen3_embedding_0_6b"]["max_tokens"], "Qwen chunk", encoder, ":"),
        (candidates[smallest]["context_tokens"], f"{style.DECISION_SHORT.get(smallest, smallest)} request", decision, "--"),
        (candidates["jev"]["context_tokens"], "Jev request", decision, "--"),
    ]


def long_text_figures(datasets, columns, *, all_columns=False):
    """The 12 longest p99 columns; all_columns=True is the explicit appendix view.

    Dataset-level tails appear in the exploration volume figure. No view filters
    extraction or benchmark rows.
    """
    ordered = columns.sort_values(["tokens_p99", "dataset", "text_column"], ascending=[False, True, True])
    if not all_columns:
        ordered = ordered.head(12)
    pages = [ordered.iloc[i:i + 12] for i in range(0, len(ordered), 12)]
    xmax = float(columns.tokens_max.max()) * 1.5
    tasks = dict(zip(datasets.dataset_id, datasets.task_type)) if "task_type" in datasets else {}
    windows = context_windows()
    for page, part in enumerate(pages, start=1):
        fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, 3.9))
        labels = [f"{row.dataset} · {fill(row.text_column, 28)}" for row in part.itertuples()]
        _draw(ax, part, "tokens", labels)
        ax.tick_params(axis="y", labelsize=style.LONG_TEXT_PAGE_FONT, length=0)
        from src.visualize.exploration import _removed

        removed = _removed()
        for tick, dataset_id in zip(ax.get_yticklabels(), part.dataset_id):
            tick.set_color(style.REMOVED_COLOR if dataset_id in removed
                           else style.TASK_COLORS.get(tasks.get(dataset_id), style.TEXT_COLOR))
            if dataset_id in removed:
                tick.set_fontstyle("italic")
        ax.set_xlim(0.7, xmax)
        for value, label, color, pattern in windows:
            ax.axvline(value, color=color, linestyle=pattern, linewidth=0.8,
                       zorder=0)
            # Inside the empty lower-right region, along the line, so labels never collide.
            ax.text(value * 1.08, 0.02, f"{label} · {value:,}", transform=ax.get_xaxis_transform(),
                    ha="left", va="bottom", rotation=90, fontsize=7, color=color)
        layout.light_grid(ax)
        ax.set_xlabel("Estimated tokens per nonempty cell (log scale)")
        if not all_columns:
            ax.set_title(f"Longest {len(part)} of {len(columns)} text columns by 99th percentile", fontsize=9)
        ax.legend(loc="upper right", bbox_to_anchor=(1, 0.93), fontsize=7)
        yield (
            fig,
            f"long_text_columns_page_{page}" if all_columns else "longest_column_tails",
            ("All official text columns, ordered by their 99th percentile. " if all_columns else
             f"Of {len(columns)} official text columns across {len(datasets)} datasets, the {len(part)} with "
             "the largest 99th percentiles are shown to make long inputs visible. ") +
            "One row represents a column. Boxes show the middle 50% of nonempty cells, with the median "
            "marked. Whiskers reach the 1st and 99th percentiles. Diamonds mark individual maxima. "
            "Lengths use one token per four characters, rounded up. Blue dotted lines mark encoder "
            "chunk sizes. Longer text is split and all chunks contribute to its embedding. "
            "Red dashed lines mark decision-model limits for complete requests, including the question "
            "and options. Experiment 0A checks the actual token counts. Grey italic labels mark "
            "datasets outside the benchmark.",
        )
