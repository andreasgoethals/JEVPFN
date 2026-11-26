"""A3 figures: do texts close in frozen-encoder space get similar decisions?

Rows are decision models, one marker per frozen encoder (colour and shape), equal-dataset means
with dataset-bootstrap intervals. A small lower panel gives the encoder-encoder agreement on the
same field samples as a reference scale; it is not a second prediction study.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.evaluation import group_a
from src.visualize import layout, style
from src.visualize.decision_diagnostics import _models, short

ENCODER_LABELS = {"e5_small_v2": "E5-small", "e5_large_v2": "E5-large", "qwen3_embedding_0_6b": "Qwen3-Emb-0.6B"}
ENCODER_MARKERS = {"e5_small_v2": "o", "e5_large_v2": "s", "qwen3_embedding_0_6b": "^"}
METRICS = {
    "spearman": ("distance_correlation", "Spearman correlation of pair distances",
                 "Permutation null (range of model means)"),
    "neighbour_overlap": ("neighbour_overlap", "Neighbour overlap (k nearest texts)",
                          "Chance overlap k/(n−1) (range)"),
}


def _label(encoder):
    return ENCODER_LABELS.get(encoder, encoder)


def geometry_figure(comparisons, references, *, metric="spearman", order=None, n_boot=group_a.N_BOOT):
    """One figure for one metric: models × encoders, plus the encoder-encoder reference panel."""
    summary = group_a.geometry_summary(comparisons, metric, n_boot=n_boot)
    if summary.empty:
        return
    name, xlabel, null_label = METRICS[metric]
    names = [n for n in (list(order) if order else _models(summary.provider)) if n in set(summary.provider)]
    encoders = [e for e in ENCODER_LABELS if e in set(summary.encoder)] + sorted(
        set(summary.encoder) - set(ENCODER_LABELS))
    pairs = group_a.encoder_reference_summary(references, metric)
    fig, (ax, ref) = plt.subplots(2, 1, sharex=True, figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 2.0 + 0.17 * len(names))),
                                  gridspec_kw={"height_ratios": [len(names), max(len(pairs), 1) + 0.6]})
    offsets = dict(zip(encoders, np.linspace(-0.25, 0.25, len(encoders)) if len(encoders) > 1 else [0.0]))
    nulls = summary.reference.to_numpy(float)
    nulls = nulls[np.isfinite(nulls)]
    if len(nulls):
        for axis in (ax, ref):
            axis.axvspan(nulls.min(), max(nulls.max(), nulls.min() + 1e-3), color=style.GRID_COLOR, zorder=0)
            axis.axvline(float(np.median(nulls)), color=style.MISSING_COLOR, lw=0.8, zorder=1)
    for encoder in encoders:
        rows = summary[summary.encoder.eq(encoder)].set_index("provider").reindex(names)
        y = np.arange(len(names)) + offsets[encoder]
        color = style.EMBEDDING_COLORS.get(encoder, style.PREDICTOR_INK)
        ax.hlines(y, rows.ci_low, rows.ci_high, color=color, lw=0.9, alpha=0.8, zorder=2)
        ax.scatter(rows["mean"], y, s=12, marker=ENCODER_MARKERS.get(encoder, "D"), color=color, linewidths=0,
                   zorder=3)
    ax.set_yticks(range(len(names)), [short(n) for n in names], fontsize=7.5)
    for tick, provider in zip(ax.get_yticklabels(), names):
        tick.set_color(style.DECISION_COLORS.get(provider, style.TEXT_COLOR))
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    datasets = int(summary.datasets.max())
    ax.set_title(f"a  Decision models vs frozen encoders, per-column requests, N = {datasets} datasets",
                 loc="left", fontsize=8)
    layout.light_grid(ax)
    if not pairs.empty:
        labels = [f"{_label(r.left_encoder)} vs {_label(r.right_encoder)}" for r in pairs.itertuples()]
        ref.scatter(pairs["mean"], range(len(pairs)), s=14, marker="D", color=style.PREDICTOR_INK, zorder=3,
                    linewidths=0)
        ref.set_yticks(range(len(pairs)), labels, fontsize=7)
        ref.set_ylim(len(pairs) - 0.4, -0.6)
    ref.tick_params(axis="y", length=0)
    ref.set_title("b  Encoder vs encoder on the same samples (reference)", loc="left", fontsize=8)
    ref.set_xlabel(xlabel, fontsize=8)
    layout.light_grid(ref)
    handles = [Line2D([], [], ls="none", marker=ENCODER_MARKERS.get(e, "D"), ms=4,
                      color=style.EMBEDDING_COLORS.get(e, style.PREDICTOR_INK), label=_label(e)) for e in encoders]
    handles.append(Patch(color=style.GRID_COLOR, label=null_label))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, fontsize=7.5)
    if metric == "spearman":
        what = ("Spearman correlation between the cosine distances of the raw encoder vectors (before PCA) and the "
                "total variation distances of the decision probabilities (nine ordered levels for regression) over "
                "all pairs of up to 500 distinct non-empty texts per text field")
        null = ("the shaded band spans the model-level means of the permutation null (text identities permuted "
                "within the field, 19 permutations), the grey line its median")
    else:
        what = ("tie-aware overlap between each text's k nearest neighbours in encoder space and in decision space "
                "(k = min(10, (n − 1)/2)) over up to 500 distinct non-empty texts per text field")
        null = "the shaded band spans the chance overlap k/(n − 1) of the samples, the grey line its median"
    yield fig, f"geometry_{name}", (
        f"(a) {what}, per decision model and frozen encoder (marker). Undefined correlations (constant outputs) "
        "are left out, never set to zero; fields are averaged within a dataset, then datasets with equal weight. "
        "Lines: 95% percentile intervals over 2,000 bootstrap resamples of datasets (pairs are not independent and "
        f"are never resampled); {null}. (b) the same measure between two frozen encoders on the same samples."
    )


def geometry_figures(comparisons, references, *, order=None):
    yield from geometry_figure(comparisons, references, metric="spearman", order=order)
    yield from geometry_figure(comparisons, references, metric="neighbour_overlap", order=order)
