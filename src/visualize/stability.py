"""A2 (exp02_robustness) figures: change beyond replay noise, averaging skill, wording averaging."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.visualize import layout, style
from src.visualize.decision_diagnostics import _models, short

DOT_COLOR = "#7A8794"

SERIES = (("replay_tv", "Unchanged replay", "#9AA3AD", "o"),
          ("single_tv", "Single wordings", "#B35806", "s"),
          ("averaged_tv", "Averages of two wordings", "#2166AC", "D"))
MODES = (("per_column", "a  Per-column requests"), ("joint", "b  Joint requests"))


def wording_averaging_figure(summary):
    """Per model: replay noise, single-wording distance and averaged-pair distance."""
    if summary.empty:
        return
    names = _models(summary.provider)
    fig, axes = plt.subplots(1, 2, sharey=True, sharex=True,
                             figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 1.3 + 0.2 * len(names))))
    for ax, (mode, title) in zip(axes, MODES):
        frame = summary[summary["mode"].eq(mode)].set_index("provider").reindex(names)
        for column, _, color, marker in SERIES:
            ax.scatter(frame[column], np.arange(len(names)), s=14, color=color, marker=marker, zorder=3,
                       linewidths=0)
        for i, row in enumerate(frame.itertuples()):
            if np.isfinite([row.single_tv, row.averaged_tv]).all():
                ax.plot([row.single_tv, row.averaged_tv], [i, i], color=style.GRID_COLOR, lw=1.2, zorder=2)
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.grid(axis="x", color=style.GRID_COLOR, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlim(0, None)
        ax.set_xlabel("Mean total variation distance", fontsize=8)
    axes[0].set_yticks(range(len(names)), [short(n) for n in names], fontsize=7.5)
    axes[0].set_ylim(len(names) - 0.5, -0.5)
    for tick, name in zip(axes[0].get_yticklabels(), names):
        tick.set_color(style.DECISION_COLORS.get(name, style.TEXT_COLOR))
    axes[1].tick_params(axis="y", length=0)
    fig.legend(handles=[Line2D([], [], ls="none", marker=m, ms=4, color=c, label=label) for _, label, c, m in SERIES],
               loc="outside lower center", ncol=3, frameon=False, fontsize=7.5)
    yield fig, "wording_averaging", (
        "Wording sensitivity of each decision model with and without averaging, from responses already collected "
        "for the stability tests (no additional model calls). Orange squares: mean total variation distance between "
        "single wordings across the pairs {original, paraphrase 1} and {paraphrase 2, paraphrase 3}; blue diamonds: "
        "distance between the averages of those two pairs; grey circles: distance between the original and its "
        "unchanged replays. Within-dataset means are combined with equal dataset weight; (a) per-column and "
        "(b) joint requests."
    )


FACTORS = (("wording", "Wording"), ("field_order", "Field order"), ("option_order", "Option order"))
FACTOR_SCOPE = {"wording": "per-column and joint", "field_order": "joint, ≥ 2 text fields",
                "option_order": "multiclass"}
MODE_MARKERS = {"per_column": "o", "joint": "s"}
MODE_NAMES = {"per_column": "Per-column requests", "joint": "Joint requests"}


def _model_axis(ax, names):
    ax.set_yticks(range(len(names)), [short(n) for n in names], fontsize=7.5)
    ax.set_ylim(len(names) - 0.5, -0.5)
    for tick, name in zip(ax.get_yticklabels(), names):
        tick.set_color(style.DECISION_COLORS.get(name, style.TEXT_COLOR))
    ax.tick_params(axis="y", length=0)


def _datasets_per_factor(summary):
    return {f: int(summary.loc[summary.factor.eq(f), "datasets"].max()) if summary.factor.eq(f).any() else 0
            for f, _ in FACTORS}


def excess_change_figure(summary, *, order=None):
    """A2 summary: per model, the change caused by one presentation factor beyond replay noise.

    ``summary`` is `group_a.excess_summary` (equal-dataset mean and dataset-bootstrap interval).
    """
    if summary.empty:
        return
    names = list(order) if order else _models(summary.provider)
    names = [n for n in names if n in set(summary.provider)]
    counts = _datasets_per_factor(summary)
    factors = [(f, label) for f, label in FACTORS if counts[f]]
    fig, axes = plt.subplots(1, len(factors), sharey=True, squeeze=False,
                             figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 1.45 + 0.19 * len(names))))
    axes = axes[0]
    for k, (ax, (factor, label)) in enumerate(zip(axes, factors)):
        frame = summary[summary.factor.eq(factor)]
        modes = [m for m in ("per_column", "joint") if frame["mode"].eq(m).any()]
        offsets = {"per_column": -0.17, "joint": 0.17} if len(modes) == 2 else {modes[0]: 0.0}
        for mode in modes:
            rows = frame[frame["mode"].eq(mode)].set_index("provider").reindex(names)
            y = np.arange(len(names)) + offsets[mode]
            color = style.MODE_COLORS[mode]
            ax.hlines(y, rows.ci_low, rows.ci_high, color=color, lw=1.0, alpha=0.8, zorder=2)
            ax.scatter(rows["mean"], y, s=13, marker=MODE_MARKERS[mode], color=color, linewidths=0, zorder=3)
        ax.axvline(0, color=style.MEDIAN_COLOR, lw=0.8, zorder=1)
        ax.set_title(f"{'abc'[k]}  {label}, N = {counts[factor]}\n({FACTOR_SCOPE[factor]})", loc="left", fontsize=7.5)
        ax.set_xlabel("Excess TV over replay", fontsize=7.5)
        ax.tick_params(axis="x", labelsize=7)
        layout.light_grid(ax)
    _model_axis(axes[0], names)
    for ax in axes[1:]:
        ax.tick_params(axis="y", length=0)
    fig.legend(handles=[Line2D([], [], ls="none", marker=MODE_MARKERS[m], ms=4, color=style.MODE_COLORS[m],
                               label=MODE_NAMES[m]) for m in ("per_column", "joint")]
               + [Line2D([], [], color=style.MEDIAN_COLOR, lw=0.8, label="Replay noise (0)")],
               loc="outside lower center", ncol=3, frameon=False, fontsize=7.5)
    yield fig, "excess_change", (
        "Change in each decision model's probabilities caused by one presentation factor, beyond the change between "
        "unchanged replays of the same requests. Per dataset and request: total variation distance between the fresh "
        "original and the changed presentation (full probability vector; nine ordered levels for regression) minus "
        "the mean total variation of the two unchanged replays; wording averages its three rewordings. Markers: "
        "equal-dataset mean per request mode (circles per-column, squares joint requests); lines: 95% percentile "
        "intervals from 2,000 bootstrap resamples of datasets. N is the number of datasets with the factor "
        "(field order: joint requests with at least two non-empty fields; option order: multiclass, realigned by "
        "class identity). 0 = replay noise."
    )


def _margin(ax, rows, names):
    """Bayesian posterior bar (worse | equivalent | better) and Holm-adjusted p per model."""
    rows = rows.set_index("provider").reindex(names)
    colors = (style.WTL_COLORS["loss"], style.WTL_COLORS["tie"], style.WTL_COLORS["win"])
    for i, row in enumerate(rows.itertuples()):
        left = 0.0
        for value, color in zip((row.bayes_p_worse, row.bayes_p_equivalent, row.bayes_p_better), colors):
            if np.isfinite(value):
                ax.barh(i, value, left=left, height=0.55, color=color, linewidth=0)
                left += value
        if np.isfinite(row.p_holm):
            ax.text(1.05, i, f"{row.p_holm:.3f}" if row.p_holm >= 0.001 else "<.001", va="center", ha="left",
                    fontsize=6.5, color=style.TEXT_COLOR)
    ax.set_xlim(0, 1.7)
    ax.set_xticks([0, 1], ["0", "1"], fontsize=6.5)
    ax.set_title("P(worse|≈|better)\nHolm p", loc="left", fontsize=6.5)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)


def averaging_skill_forest(summary, by_dataset, *, order=None, tests=None):
    """A2: proper-score skill of the averaged over the standard presentation, per model and factor.

    ``summary`` is `group_a.averaging_skill_summary`, ``by_dataset`` the per-dataset skills. The
    margin (Bayesian posterior bar and Holm-adjusted p) is drawn only when ``tests`` is given, i.e.
    after the owner approves the statistical framework (columns provider, factor, p_holm,
    bayes_p_worse, bayes_p_equivalent, bayes_p_better).
    """
    if summary.empty:
        return
    names = list(order) if order else _models(summary.provider)
    names = [n for n in names if n in set(summary.provider)]
    counts = _datasets_per_factor(summary)
    factors = [(f, label) for f, label in FACTORS if counts[f]]
    ratios = []
    for _ in factors:
        ratios += [1, 0.42] if tests is not None else [1]
    fig, axes = plt.subplots(1, len(ratios), sharey=True, squeeze=False,
                             figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 1.45 + 0.19 * len(names))),
                             gridspec_kw={"width_ratios": ratios})
    axes = list(axes[0])
    panels = axes[::2] if tests is not None else axes
    y = np.arange(len(names))
    for k, (ax, (factor, label)) in enumerate(zip(panels, factors)):
        rows = summary[summary.factor.eq(factor)].set_index("provider").reindex(names)
        dots = by_dataset[by_dataset.factor.eq(factor)]
        for i, name in enumerate(names):
            v = dots.loc[dots.provider.eq(name), "skill"].dropna().to_numpy(float)
            if len(v):
                ax.scatter(v, i + np.linspace(-0.25, 0.25, len(v)) if len(v) > 1 else [i], s=4, color=DOT_COLOR,
                           alpha=0.45, linewidths=0, zorder=1)
        ax.hlines(y, rows.ci_low, rows.ci_high, color=style.MEDIAN_COLOR, lw=1.1, zorder=3)
        ax.scatter(rows["mean"], y, s=16, marker="D", color=style.MEDIAN_COLOR, linewidths=0, zorder=4)
        ax.scatter(rows["median"], y, s=16, marker="|", color=style.WINDOW_COLOR, linewidths=1.2, zorder=4)
        ax.axvline(0, color=style.MEDIAN_COLOR, lw=0.8, zorder=2)
        values = np.concatenate([dots.skill.dropna().to_numpy(float), rows[["ci_low", "ci_high"]].to_numpy(float).ravel()])
        values = values[np.isfinite(values)]
        if len(values):
            bound = max(np.abs(values).max(), 1e-3) * 1.08
            ax.set_xlim(-bound, bound)
        ax.set_title(f"{'abc'[k]}  {label}, N = {counts[factor]}", loc="left", fontsize=7.5)
        ax.set_xlabel("Proper-score skill", fontsize=7.5)
        ax.tick_params(axis="x", labelsize=7)
        layout.light_grid(ax)
        if tests is not None:
            _margin(axes[2 * k + 1], tests[tests.factor.eq(factor)], names)
    _model_axis(axes[0], names)
    for ax in axes[1:]:
        ax.tick_params(axis="y", length=0)
    handles = [Line2D([], [], ls="none", marker="o", ms=3, color=DOT_COLOR, label="One dataset"),
               Line2D([], [], ls="-", marker="D", ms=4, color=style.MEDIAN_COLOR, label="Mean, 95% CI"),
               Line2D([], [], ls="none", marker="|", ms=6, mew=1.2, color=style.WINDOW_COLOR, label="Median")]
    if tests is not None:
        handles += [Patch(color=style.WTL_COLORS[k], label=label) for k, label in
                    (("loss", "P(worse)"), ("tie", "P(equivalent)"), ("win", "P(better)"))]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, fontsize=7.5,
               columnspacing=1.0, handletextpad=0.4)
    margin = (" Right of each panel: Bayesian signed-rank posterior probabilities that averaging is worse, "
              "practically equivalent (ROPE ±0.01) or better than the standard presentation, and the Holm-adjusted "
              "two-sided Wilcoxon p over all declared contrasts." if tests is not None else "")
    yield fig, "averaging_skill", (
        "Stand-alone quality of each decision model's averaged presentation relative to the standard (production) "
        "presentation on the sampled joint requests: proper-score skill 1 − S_averaged / S_standard per dataset, "
        "where S is log loss for the nine classification datasets and the ranked probability score over the nine "
        "ordered levels for the nine regression datasets (observed level: nine equal-frequency target bins); "
        "positive = averaging scored better. Averaged presentation: the mean of the four wordings, of original and "
        "reversed field order, or of original and reversed option order (realigned by class identity). Grey dots: "
        "datasets; diamond: equal-dataset mean with 95% percentile interval over 2,000 dataset bootstrap resamples; "
        "bar: median. Labels are used only for scoring. N = datasets with the factor." + margin
    )
