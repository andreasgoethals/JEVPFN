"""Cross-study overview figures for `general/5_synthesis` (no letter-synthesis panels redrawn)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.visualize import layout, style

ARM_COLORS = {
    "Decision, median model (both)": style.MODE_COLORS["both"],
    "Decision + E5, median model": style.MODE_COLORS["both"],
    "Frozen E5-small, per-column PCA32": style.EMBEDDING_COLORS["e5_small_v2"],
    "Label-trained classifier (E5-small)": style.PROBE_COLOR,
    "TAR E5-small, per-column PCA32": style.TAR_COLOR,
    "Strands LoRA + head (C2)": style.STRANDS_COLORS["lora_head"],
}
SCOPES = (("pooled", "Headroom-normalised gain\n(all datasets)"),
          ("normalized", "Normalised score, 0–1\n(all datasets)"),
          ("classification", "ΔAUC, classification"), ("regression", "ΔR², regression"))
#: Reference arm of the normalised scope, drawn as a vertical line (the gain panels draw 0).
NO_TEXT = "No text (same predictor)"


def study_at_a_glance(table):
    """Mean (circle, 95% CI) and median (diamond) per arm: pooled normalised gain, the pooled
    MulTaBench-style normalised score (when fold tables exist) and raw gains per task family in
    separate panels with separate axes."""
    arms = [a for a in dict.fromkeys(table.arm) if a != NO_TEXT]
    if not arms:
        return None
    scopes = [s for s in SCOPES if s[0] in set(table.scope)]
    fig, axes = plt.subplots(1, len(scopes), figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 1.3 + 0.32 * len(arms))),
                             sharey=True, squeeze=False)
    axes = axes[0]
    y = np.arange(len(arms))
    for ax, (scope, label) in zip(axes, scopes):
        part = table[table.scope.eq(scope)]
        for i, arm in enumerate(arms):
            color = ARM_COLORS.get(arm, style.MEDIAN_COLOR)
            hollow = arm.startswith("Decision + E5")
            for statistic, offset, marker in (("mean", -0.13, "o"), ("median", 0.13, "D")):
                row = part[part.arm.eq(arm) & part.statistic.eq(statistic)]
                if row.empty:
                    continue
                row = row.iloc[0]
                if np.isfinite(row.low) and np.isfinite(row.high):
                    ax.hlines(i + offset, row.low, row.high, color=color, linewidth=1)
                ax.scatter(row.estimate, i + offset, marker=marker, s=18, zorder=3,
                           facecolors="none" if hollow else color, edgecolors=color, linewidths=0.9)
        reference = part[part.arm.eq(NO_TEXT) & part.statistic.eq("mean")]
        if scope == "normalized":
            if not reference.empty:
                ax.axvline(reference.estimate.iloc[0], color=style.NON_TEXT_COLOR, linewidth=0.8, zorder=0)
            ax.set_xlim(-0.02, 1.02)
        else:
            ax.axvline(0, color=style.NON_TEXT_COLOR, linewidth=0.8, zorder=0)
        n = int(part.datasets.max()) if len(part) else 0
        ax.set_xlabel(f"{label}\n(N = {n} datasets)", fontsize=7)
        layout.light_grid(ax)
    axes[0].set_yticks(y, arms, fontsize=7)
    axes[0].set_ylim(len(arms) - 0.5, -0.5)
    axes[0].tick_params(axis="y", length=0)
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=4, color=style.MEDIAN_COLOR,
                      label="mean, dataset-bootstrap 95% CI"),
               Line2D([], [], linestyle="none", marker="D", markersize=4, color=style.MEDIAN_COLOR,
                      label="median (CI on the pooled panels)"),
               Line2D([], [], color=style.NON_TEXT_COLOR, label="no text (same predictor)")]
    fig.legend(handles=handles, loc="outside lower center", ncols=3, fontsize=7, frameon=False)
    predictors = int(table.predictors.min())
    normalized = ("normalized" in set(table.scope))
    return fig, "study_at_a_glance", (
        "Gain of each prespecified arm over the same predictor without text. Per dataset, the "
        f"five-fold mean gain is averaged over the {predictors} predictors; circles give the "
        "equal-dataset mean with a dataset-bootstrap 95% percentile interval (2,000 resamples), "
        "diamonds the median (interval on the pooled panels only). Left: headroom-normalised gain "
        "(gain divided by 1 − baseline score), pooled over all datasets. "
        + ("Second: MulTaBench-style normalised score, per dataset and fold the headline score "
           "(negative R² clipped at 0) min-max normalised across every predictor x arm shown plus the "
           "no-text baseline (grey line: its mean), averaged over predictors; relative to this "
           "comparison set. " if normalized else "")
        + "Right: raw gains within each task family (binary and multiclass ROC-AUC; regression R²), "
        "never averaged across families. Decision rows use the per-dataset median over all 17 decision "
        "models; the Strands row is the prespecified LoRA + head cell, not a selected best arm."
    )


def confirmatory_forest(results):
    """Declared confirmatory contrasts: estimate with 95% CI, Holm-adjusted p and the
    Bayesian signed-rank posterior (left / ROPE / right) as a bar in the margin.

    ``results`` columns: contrast, estimate, low, high, p_holm, p_left, p_rope, p_right.
    """
    if results is None or results.empty:
        return None
    data = results.reset_index(drop=True)
    y = np.arange(len(data))
    fig, axes = plt.subplots(1, 2, figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 1.2 + 0.3 * len(data))),
                             sharey=True, gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    significant = data.p_holm.lt(0.05)
    for i, row in data.iterrows():
        color = style.CD_SIGNIFICANT_COLOR if significant[i] else style.MEDIAN_COLOR
        ax.hlines(i, row.low, row.high, color=color, linewidth=1)
        ax.scatter(row.estimate, i, s=18, color=color, zorder=3)
        ax.text(1.0, i, f" p(Holm) = {row.p_holm:.3f}", transform=ax.get_yaxis_transform(), fontsize=6.5,
                va="center", ha="left", color=color)
    ax.axvline(0, color=style.NON_TEXT_COLOR, linewidth=0.8)
    ax.set_yticks(y, data.contrast, fontsize=7)
    ax.set_ylim(len(data) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Paired effect (positive favours the first arm)", fontsize=7)
    layout.light_grid(ax)
    bar = axes[1]
    # The two ends of EFFECT_CMAP (BrBG): brown = second arm better, teal = first arm better.
    colors = ("#A6611A", style.MISSING_COLOR, "#018571")
    left = np.zeros(len(data))
    for column, color in zip(("p_left", "p_rope", "p_right"), colors):
        bar.barh(y, data[column], left=left, height=0.55, color=color)
        left += data[column].to_numpy(float)
    bar.set_xlim(0, 1)
    bar.set_xlabel("Posterior probability", fontsize=7)
    bar.tick_params(axis="y", length=0)
    fig.legend(handles=[Patch(color=colors[0], label="second arm better"), Patch(color=colors[1], label="ROPE"),
                        Patch(color=colors[2], label="first arm better")],
               loc="outside lower center", ncols=3, fontsize=7, frameon=False)
    return fig, "confirmatory_family", (
        f"All {len(data)} declared confirmatory contrasts. Left: median paired per-dataset effect with "
        "dataset-bootstrap "
        "95% interval; red marks Holm-adjusted two-sided Wilcoxon p < 0.05 over the whole family. "
        "Right: Bayesian signed-rank posterior probabilities that the second arm is better, that the "
        "difference lies within the region of practical equivalence (ROPE), or that the first arm is better."
    )
