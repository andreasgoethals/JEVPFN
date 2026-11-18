"""Headline normalised-gain forest plots and predictor ranking agreement; no winner selected."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src.evaluation.aggregate import MEDIAN, headline_rows
from src.visualize import layout, style
from src.visualize.research import ENCODER_LABELS

_FAMILY_PHASE = {"decision": "experiment_1", "embedding": "experiment_2", "tar": "experiment_3"}
_FAMILY_NAMES = {"decision": "Decision features (median over models)", "embedding": "Frozen embedding",
                 "tar": "Supervised TAR"}
_MODE_LABELS = {"per_column": "Per-column", "joint": "Joint", "both": "Both"}
_SCOPE = {"per_column": "P", "joint": "J"}


def _label(representation):
    family, first, second = representation.split("/")
    if family == "decision":
        return f"Decision · {_MODE_LABELS[first]}"
    scope = next(s for s in _SCOPE if second.startswith(s))
    name = ENCODER_LABELS.get(first, first)
    return f"{name} · {_SCOPE[scope]}{second[len(scope):]}"


def _order(representations):
    """Decision modes, then encoders in their reviewed order, per-column before joint PCA."""
    modes, encoders = list(_MODE_LABELS), list(ENCODER_LABELS)

    def key(representation):
        family, first, second = representation.split("/")
        if family == "decision":
            return (0, modes.index(first), 0, "")
        scope = next(s for s in _SCOPE if second.startswith(s))
        rank = encoders.index(first) if first in encoders else len(encoders)
        return (1 if family == "embedding" else 2, rank, list(_SCOPE).index(scope), second)

    return sorted(set(representations), key=key)


def _forest(rows, *, colors):
    representations = _order(rows.representation)
    models = [m for m in style.MAIN_PREDICTORS if m in set(rows.model_id)]
    columns = min(3, len(models))
    lines = int(np.ceil(len(models) / columns))
    height = min(style.MAX_HEIGHT, 0.7 + lines * (0.35 + 0.2 * len(representations)))
    fig, axes = plt.subplots(lines, columns, figsize=(style.WIDTH_FULL, height), sharex=True,
                             sharey=True, squeeze=False)
    for ax, model in zip(axes.flat, models):
        part = rows[rows.model_id.eq(model)].set_index("representation")
        ax.axvline(0, color=style.MEDIAN_COLOR, linewidth=0.8, zorder=1)
        for position, representation in enumerate(representations):
            if representation not in part.index:
                continue
            row = part.loc[representation]
            color = colors(representation)
            if np.isfinite(row.ci_low) and np.isfinite(row.ci_high):
                ax.hlines(position, row.ci_low, row.ci_high, color=color, linewidth=1.2, zorder=2)
            ax.scatter(row.mean_gain, position, color=color, s=14, zorder=3)
        ax.set_title(style.PREDICTOR_LABELS[model], fontsize=8)
        layout.light_grid(ax)
        ax.tick_params(axis="x", labelsize=7)
    for ax in axes.flat[len(models):]:
        ax.set_axis_off()
    axes[0, 0].set_yticks(range(len(representations)), [_label(r) for r in representations], fontsize=7)
    axes[0, 0].set_ylim(len(representations) - 0.5, -0.5)
    for ax in axes[:, 0]:
        for tick, representation in zip(ax.get_yticklabels(), representations):
            tick.set_color(colors(representation))
    fig.supxlabel("Normalised gain (share of the baseline's headroom closed)", fontsize=7)
    return fig


def gain_forest_figures(summary):
    """Equal-dataset mean normalised gain and dataset-bootstrap 95% interval, one panel per predictor."""
    rows = headline_rows(summary)
    if not rows.empty:
        fig = _forest(rows, colors=lambda r: style.FAMILY_COLORS[_FAMILY_PHASE[r.split("/")[0]]])
        families = [f for f in _FAMILY_NAMES if f in {r.split("/")[0] for r in rows.representation}]
        fig.legend(handles=[Line2D([], [], color=style.FAMILY_COLORS[_FAMILY_PHASE[f]], marker="o",
                                   markersize=4, label=_FAMILY_NAMES[f]) for f in families],
                   loc="outside upper center", ncols=len(families), fontsize=7, frameon=False)
        datasets = f"{rows.n_datasets.min()}–{rows.n_datasets.max()}"
        yield fig, "normalized_gain_headline", (
            "Mean normalised gain of each representation family over the same predictor's no-text "
            "baseline, with equal weight per dataset; horizontal lines are 95% percentile intervals "
            f"from {int(rows.n_boot.iloc[0])} bootstrap resamples of datasets ({datasets} datasets per "
            "point). Gain is the five-fold mean difference in ROC-AUC, macro one-vs-rest ROC-AUC or R² "
            "divided by the baseline's distance from 1, pooled across task types. Decision rows take "
            "the median over all decision models per dataset; embedding and TAR rows are separate "
            "encoder and PCA recipes (P32 = 32 components per text column, J64 = 64 joint). Arms "
            "without five defined paired folds are excluded. Descriptive; not a confirmatory test."
        )
    modes = summary[summary.dataset_scope.eq("multi_text") & summary.aggregation.eq(MEDIAN)] if not summary.empty else summary
    if not modes.empty:
        fig = _forest(modes, colors=lambda r: style.MODE_COLORS[r.split("/")[1]])
        yield fig, "normalized_gain_decision_modes", (
            "Mean normalised gain of per-column, joint and both decision-feature modes, restricted "
            "to datasets with at least two text columns, where the modes differ by construction. "
            "Points are equal-dataset means of the per-dataset median over decision models; lines "
            "are 95% dataset-bootstrap percentile intervals. Descriptive; not a confirmatory test."
        )


def ranking_agreement_figures(pairs):
    """Predictor × predictor Kendall tau-b of the headline representation rankings."""
    if pairs.empty or pairs.kendall_tau_b.isna().all():
        return
    models = [m for m in style.MAIN_PREDICTORS if m in set(pairs.model_a) | set(pairs.model_b)]
    matrix = np.full((len(models), len(models)), np.nan)
    np.fill_diagonal(matrix, 1.0)
    for row in pairs.itertuples():
        i, j = models.index(row.model_a), models.index(row.model_b)
        matrix[i, j] = matrix[j, i] = row.kendall_tau_b
    cmap = plt.get_cmap(style.AGREEMENT_DIVERGING).with_extremes(bad=style.FAINT_COLOR)
    fig, ax = plt.subplots(figsize=(style.WIDTH_HALF + 0.6, style.WIDTH_HALF))
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=-1, vmax=1)
    for (i, j), value in np.ndenumerate(matrix):
        if np.isfinite(value):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                    color=style.BACKGROUND_COLOR if abs(value) > 0.6 else style.TEXT_COLOR)
    labels = [style.PREDICTOR_LABELS[m] for m in models]
    ax.set_xticks(range(len(models)), labels, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(models)), labels, fontsize=7)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(image, ax=ax, shrink=0.8, label="Kendall tau-b")
    n = f"{pairs.n_representations.min()}–{pairs.n_representations.max()}"
    yield fig, "ranking_agreement", (
        "Kendall tau-b between two predictors' orderings of the headline representation families "
        f"by equal-dataset mean normalised gain, over the {n} families both predictors share. "
        "One means identical orderings, zero no association and minus one reversed orderings; "
        "grey cells have fewer than two shared families or a constant ordering."
    )
