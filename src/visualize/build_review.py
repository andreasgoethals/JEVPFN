"""Measured extraction compute per model and the fixed regression rubric."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.visualize import style


def extraction_figure(workload, rates, *, jev_charges=None):
    """Measured extraction compute per decision model and frozen encoder.

    Grey dots show the inference time per 1,000 timed inputs for one dataset. A diamond
    pools timed inputs and an open circle uses the whole-job Slurm ledger when available.
    Percentages on recorded hours compare timed input counts with the audited workload.
    """
    from src.visualize import layout

    labels = {**style.DECISION_SHORT, "e5_small_v2": "E5-small-v2", "e5_large_v2": "E5-large-v2",
              "qwen3_embedding_0_6b": "Qwen3-Emb-0.6B"}
    data = workload.reset_index(drop=True)
    y = np.arange(len(data))
    fig, axes = plt.subplots(1, 2, figsize=(style.WIDTH_FULL, 4.3), sharey=True,
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    for i, row in data.iterrows():
        part = rates[rates.model_id.eq(row.model_id)]
        if not part.empty:
            jitter = np.linspace(-0.18, 0.18, len(part))
            ax.scatter(part.gpu_hours_per_1000, i + jitter, s=7, color=style.DECISION_POOL_COLOR,
                       linewidths=0, zorder=2)
    measured = data.inference_gpu_hours_per_1000.notna()
    ax.scatter(data.inference_gpu_hours_per_1000[measured], y[measured], marker="D", s=20,
               color=style.MEDIAN_COLOR, zorder=3)
    ledger = data.ledger_gpu_hours_per_1000.notna()
    if ledger.any():
        ax.scatter(data.ledger_gpu_hours_per_1000[ledger], y[ledger], s=26, facecolors="none",
                   edgecolors=style.MEDIAN_COLOR, linewidths=0.9, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("GPU-hours per 1,000 inputs (log)")
    has_hours = data.inference_gpu_hours.notna()
    axes[1].barh(y[has_hours], data.inference_gpu_hours[has_hours], height=0.6,
                 color=style.QUANTITY_COLOR)
    top = float(data.inference_gpu_hours[has_hours].max()) if has_hours.any() else 1.0
    axes[1].set_xlim(0, max(top * 1.3, 1e-9))
    axes[1].set_xlabel("Recorded inference GPU-hours")
    for i, row in data.iterrows():
        if row.kind == "hosted":
            note = "hosted API" + (f" · \\${jev_charges:.2f} charged" if jev_charges is not None else "")
        elif pd.isna(row.inference_gpu_hours):
            note = "no local timing manifest"
        else:
            if pd.notna(row.distinct_requests) and row.distinct_requests > 0:
                percent = 100 * row.timed_requests / row.distinct_requests
                label = f"{percent:.2f}".rstrip("0").rstrip(".") + "%"
                if label == "100%" and row.timed_requests != row.distinct_requests:
                    label = "<100%" if row.timed_requests < row.distinct_requests else ">100%"
                axes[1].text(row.inference_gpu_hours + top * 0.02, i, label, va="center", fontsize=7,
                             color=style.MEDIAN_COLOR)
            continue
        axes[1].text(0.02, i, note, va="center", fontsize=7, color=style.REMOVED_COLOR,
                     transform=axes[1].get_yaxis_transform())
    axes[0].set_yticks(y, [labels.get(m, m) for m in data.model_id], fontsize=7)
    axes[0].set_ylim(len(data) - 0.5, -0.5)
    for tick, model in zip(axes[0].get_yticklabels(), data.model_id):
        tick.set_color(style.DECISION_COLORS.get(model, style.EMBEDDING_COLORS.get(model, style.TEXT_COLOR)))
    boundary = int(data.kind.ne("encoder").sum())
    for axis in axes:
        layout.light_grid(axis)
        axis.tick_params(axis="y", length=0)
        if boundary < len(data):
            axis.axhline(boundary - 0.5, color=style.GRID_COLOR, linewidth=0.9)
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=3, color=style.DECISION_POOL_COLOR,
                      label="one dataset"),
               Line2D([], [], linestyle="none", marker="D", markersize=4, color=style.MEDIAN_COLOR,
                      label="pooled over datasets")]
    if ledger.any():
        handles.append(Line2D([], [], linestyle="none", marker="o", markersize=5, markerfacecolor="none",
                              color=style.MEDIAN_COLOR, label="Slurm ledger (whole jobs)"))
    fig.legend(handles=handles, loc="outside upper center", ncols=len(handles), fontsize=7, frameon=False)
    requests = data.loc[data.kind.eq("decision"), "distinct_requests"].dropna().unique()
    texts = data.loc[data.kind.eq("encoder"), "distinct_requests"].dropna().unique()
    caption = [
        "Recorded extraction time for open-weight decision models and frozen encoders on one B200. "
        "The left panel shows inference GPU-hours per 1,000 timed inputs. Grey dots represent datasets. "
        "Each diamond divides total recorded inference time by the pooled input count.",
    ]
    if ledger.any():
        caption.append("Open circles divide successful Slurm job time, including overhead, by the "
                       "audited workload in thousands of inputs.")
    caption.append("The right panel shows total recorded inference GPU-hours. Percentages compare "
                   "the number of input items in local timing records with the audited number of "
                   "distinct inputs. Completion is tracked separately in the feature inventory.")
    if len(requests) == 1:
        caption.append(f"The decision-model workload is {int(requests[0]):,} distinct requests per model.")
    if len(texts) == 1:
        caption.append(f"The encoder workload is {int(texts[0]):,} distinct texts per encoder.")
    caption.append("Jev uses a hosted API.")
    return (
        fig, "extraction_compute", " ".join(caption),
    )


def regression_scale_figure(labels):
    from textwrap import fill

    from matplotlib.colors import to_rgb

    items = sorted(((int(level), wording) for level, wording in labels.items()))
    fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, 3.0))
    ax.set_axis_off()
    ax.set(xlim=(0, 1), ylim=(len(items) - 0.5, -0.5))
    white = np.ones(3)
    for row, (level, wording) in enumerate(items):
        side = "lower" if level < 0 else "higher"
        tint = white if level == 0 else white + abs(level) / 4 * (np.array(
            to_rgb(style.DIRECTION_COLORS[side])) - white)
        ax.add_patch(plt.Rectangle((0, row - 0.42), 0.055, 0.84, color=tint))
        ax.text(0.0275, row, f"{level:+d}" if level else "0", ha="center", va="center", fontsize=7.5,
                weight="bold", color="white" if abs(level) >= 3 else style.TEXT_COLOR)
        name, _, criterion = wording.partition(":")
        ax.text(0.07, row, name, va="center", fontsize=7, weight="bold")
        ax.text(0.29, row, fill(criterion.strip(), 98), va="center", fontsize=7,
                linespacing=1.1)
    ax.axhline(3.5, xmin=0.07, color=style.GRID_COLOR, linewidth=0.6)
    ax.axhline(4.5, xmin=0.07, color=style.GRID_COLOR, linewidth=0.6)
    return (
        fig,
        "regression_evidence_levels",
        "The fixed nine-level rubric used for every regression question. Each level describes "
        "the strength and specificity of textual evidence for a relatively lower or higher "
        "target value; colour intensity grows with |level|. All nine level probabilities enter the "
        "predictors as features (no mean readout). "
        "Levels are defined without target statistics or row targets and are not target units.",
    )
