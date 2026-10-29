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
    smallest = min(candidates.values(), key=lambda item: item["context_tokens"])
    return [
        (encoders["e5_small_v2"]["max_tokens"], "E5 window"),
        (encoders["qwen3_embedding_0_6b"]["max_tokens"], "Qwen chunk"),
        (smallest["context_tokens"], "smallest decision window"),
        (candidates["jev"]["context_tokens"], "Jev window"),
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
        for tick, dataset_id in zip(ax.get_yticklabels(), part.dataset_id):
            tick.set_color(style.TASK_COLORS.get(tasks.get(dataset_id), style.TEXT_COLOR))
        ax.set_xlim(0.7, xmax)
        for value, label in windows:
            ax.axvline(value, color=style.WINDOW_COLOR, linestyle=(0, (3, 2)), linewidth=0.8,
                       zorder=0)
            # Inside the empty lower-right region, along the line, so labels never collide.
            ax.text(value * 1.08, 0.02, f"{label} · {value:,}", transform=ax.get_xaxis_transform(),
                    ha="left", va="bottom", rotation=90, fontsize=6, color=style.WINDOW_COLOR)
        layout.light_grid(ax)
        ax.set_xlabel("Nonempty cell length (approximate tokens, log scale)")
        ax.legend(loc="upper right", bbox_to_anchor=(1, 0.93), fontsize=7)
        yield (
            fig,
            f"long_text_columns_page_{page}" if all_columns else "longest_column_tails",
            ("All official text columns, ordered by descending p99 length. " if all_columns else
             "The twelve official text columns with the largest nonempty p99 length, selected "
             "without targets. ") +
            "Boxes span p25–p75 with the median; whiskers span p01–p99; diamonds mark maxima. "
            "Dashed lines mark the E5 window, the configured Qwen chunk, the smallest "
            "decision-model request window (Strands) and the Jev window. Lengths are "
            "ceil(characters/4) for the text alone, whereas decision windows must also hold the "
            "question. Longer encoder inputs are chunked; decision requests are never truncated.",
        )
