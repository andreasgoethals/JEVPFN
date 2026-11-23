"""Measured extraction compute per model and the fixed regression rubric."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.visualize import style


def extraction_figure(workload, rates, *, jev_charges=None):
    """Measured extraction compute per decision model and frozen encoder.

    Left: GPU-hours per 1,000 distinct requests (inference only), one grey dot per dataset
    and a diamond for the pooled rate; an open circle adds the Slurm-ledger rate (whole jobs)
    when a ledger exists. Right: recorded inference GPU-hours and the share of distinct
    requests they cover. Models without local timing evidence are labelled, never estimated.
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
    ax.set_xlabel("GPU-hours per 1,000 distinct requests (log)")
    axes[1].barh(y, data.inference_gpu_hours.fillna(0), height=0.6, color=style.QUANTITY_COLOR)
    top = float(np.nanmax(data.inference_gpu_hours)) if measured.any() else 1.0
    axes[1].set_xlim(0, top * 1.55)
    axes[1].set_xlabel("Recorded inference GPU-hours")
    for i, row in data.iterrows():
        if row.kind == "hosted":
            note = "hosted API" + (f" · \\${jev_charges:.2f} charged" if jev_charges is not None else "")
        elif pd.isna(row.inference_gpu_hours):
            note = "no local timing manifest"
        else:
            share = row.timed_requests / row.distinct_requests if row.distinct_requests else np.nan
            note = f"{share:.0%} of requests timed" if np.isfinite(share) else ""
            axes[1].text(row.inference_gpu_hours + top * 0.03, i, note, va="center", fontsize=6.5,
                         color=style.MEDIAN_COLOR)
            continue
        for axis in axes:
            axis.text(0.02, i, note, va="center", fontsize=6.5, color=style.REMOVED_COLOR,
                      transform=axis.get_yaxis_transform())
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
    requests = data.loc[data.kind.eq("decision"), "distinct_requests"].dropna()
    texts = data.loc[data.kind.eq("encoder"), "distinct_requests"].dropna()
    return (
        fig,
        "extraction_compute",
        "Measured extraction compute per decision model (top) and frozen encoder (bottom). Left: GPU-hours "
        "per 1,000 distinct requests from the timing manifests of the feature-creation jobs (inference "
        "wall time on one B200; model loading, staging and publication excluded); grey dots are datasets, "
        "the diamond pools all timed requests"
        + ("; open circles use successful Slurm job time over audited distinct requests" if ledger.any() else "")
        + ". Right: total recorded inference GPU-hours and the share of distinct requests they cover; "
        "the remainder was served from earlier runs whose timings are not held locally. "
        + (f"Each open-weight decision model answers {int(requests.iloc[0]):,} distinct requests "
           if len(requests) and requests.nunique() == 1 else "")
        + (f"and each encoder embeds {int(texts.iloc[0]):,} distinct texts. " if len(texts) and texts.nunique() == 1 else ". ")
        + "Models without a local timing manifest are labelled, not estimated; Jev is a hosted API "
        "charged per token.",
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
