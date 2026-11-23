"""A1 (exp01_outputs) figures at final A4 width: model profile strip, output shape, pairwise agreement.

Every view uses distinct inputs (``unique_inputs`` weighting). Per-column text fields are
averaged within a dataset first, so each dataset counts once per request mode. The former
consensus heatmap is folded into the profile strip (panel c), 23 November 2026.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from src.evaluation.decision_comparison import dataset_balanced
from src.visualize import layout, style
from src.visualize.decision_diagnostics import _ink, _models, short

TASKS = layout.TASK_LABELS
MODE_LABELS = {"joint": "joint requests", "per_column": "per-column requests"}
DOT_COLOR = "#7A8794"


def per_dataset(table, mode, metrics, weighting="unique_inputs"):
    """One value per dataset, model and mode: text fields averaged with equal weight."""
    if table.empty:
        return table
    frame = table[table.representation.eq(mode) & table.weighting.eq(weighting)]
    keys = ["dataset_id", "dataset", "task", "provider"]
    return frame.groupby(keys, sort=False)[metrics].mean().reset_index()


def _dataset_order(frame):
    rows = layout.ordered(frame.drop_duplicates("dataset")[["dataset", "task"]].sort_values("dataset"),
                          task="task")
    return rows.dataset.tolist(), rows.task.tolist()


def _dots(ax, order, frame, metric):
    for i, dataset in enumerate(order):
        values = frame.loc[frame.dataset.eq(dataset), metric].dropna()
        if values.empty:
            continue
        jitter = np.linspace(-0.2, 0.2, len(values)) if len(values) > 1 else np.zeros(1)
        ax.scatter(values, i + jitter, s=9, color=DOT_COLOR, alpha=0.75, linewidths=0, zorder=2)
        ax.plot([values.median()] * 2, [i - 0.34, i + 0.34], color=style.MEDIAN_COLOR, lw=1.6, zorder=3)


def output_shape_figure(shape, mode="joint"):
    """How concentrated are each model's answers? One dot per model and dataset."""
    metrics = ["mean_top_probability", "confident_share", "mean_normalized_entropy"]
    frame = per_dataset(shape, mode, metrics)
    if frame.empty:
        return
    order, tasks = _dataset_order(frame)
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 1.4 + 0.2 * len(order))))
    titles = ("a  Mean top probability", "b  Share top p > 0.95", "c  Normalized entropy")
    for ax, metric, title in zip(axes, metrics, titles):
        _dots(ax, order, frame, metric)
        ax.set_xlim(0, 1.02)
        ax.set_title(title, loc="left", fontsize=8.5)
        layout.light_grid(ax)
        layout.task_rules(ax, tasks)
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    layout.dataset_axis(axes[0], order, tasks, fontsize=7.5)
    for ax in axes[1:]:
        ax.tick_params(axis="y", length=0)
    fig.legend(handles=[Line2D([], [], ls="none", marker="o", ms=3.5, color=DOT_COLOR, label="One decision model"),
                        Line2D([], [], color=style.MEDIAN_COLOR, lw=1.6, label="Median over models")],
               loc="outside lower center", ncol=2, frameon=False, fontsize=7.5)
    yield fig, f"output_shape_{mode}", (
        f"Output shape of every decision model on {MODE_LABELS[mode]}, one row per dataset (grouped by task). "
        "Each grey dot is one model's mean over the dataset's distinct inputs"
        + (", averaged with equal weight over text fields" if mode == "per_column" else "")
        + "; the dark bar is the median over models. (a) mean probability of the most probable answer, "
        "(b) share of inputs whose top probability exceeds 0.95, (c) Shannon entropy divided by the log of the "
        "number of options (0: certain, 1: uniform). Regression uses the nine ordered evidence levels. "
        "Concentration is not calibration or accuracy."
    )


def regression_levels_figure(shape, mode="joint"):
    """How much of the nine-level scale does each model use on regression requests?"""
    frame = per_dataset(shape, mode, ["mean_level_std", "expected_level_median"]) if not shape.empty else shape
    if frame.empty or "mean_level_std" not in frame:
        return
    frame = frame[frame.task.eq("regression")]
    if frame.empty:
        return
    names = _models(frame.provider)
    order = sorted(frame.dataset.unique())
    fig, axes = plt.subplots(2, 1, figsize=(style.WIDTH_FULL, 4.6), sharex=True)
    specs = (("mean_level_std", "a  Mean spread over the nine levels (SD, level units)", "Blues", 0, None),
             ("expected_level_median", "b  Median expected level", "BrBG", -4, 4))
    for ax, (metric, title, cmap, vmin, vmax) in zip(axes, specs):
        matrix = frame.pivot(index="dataset", columns="provider", values=metric).reindex(index=order, columns=names)
        values = matrix.to_numpy(float)
        image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax if vmax is not None else max(np.nanmax(values), 1e-9),
                          aspect="auto")
        for i, j in zip(*np.isfinite(values).nonzero()):
            ax.text(j, i, f"{round(values[i, j], 1) + 0.0:.1f}", ha="center", va="center", fontsize=6,
                    color=_ink(image, values[i, j]))
        ax.set_yticks(range(len(order)), order, fontsize=7)
        for tick in ax.get_yticklabels():
            tick.set_color(style.TASK_COLORS["regression"])
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.colorbar(image, ax=ax, shrink=0.85, pad=0.01)
    axes[1].set_xticks(range(len(names)), [short(n) for n in names], rotation=45, ha="right", rotation_mode="anchor",
                       fontsize=7)
    for tick, name in zip(axes[1].get_xticklabels(), names):
        tick.set_color(style.DECISION_COLORS.get(name, style.TEXT_COLOR))
    yield fig, f"regression_levels_{mode}", (
        f"Use of the nine ordered evidence levels (−4 to +4) on regression {MODE_LABELS[mode]}; rows are datasets, "
        "columns decision models, means over distinct inputs. (a) average within-input standard deviation of the level "
        "under the model's probabilities (0: all mass on one level). (b) median over inputs of the expected level. "
        "Levels describe evidence direction, not bins of the target; the full expected-level distributions are in the "
        "level-histogram table."
    )


def pairwise_figures(pairs, mode="joint"):
    """Are two models interchangeable? One full-width matrix per task (equal dataset weight)."""
    if pairs.empty or "top_agreement" not in pairs:
        return
    frame = pairs[pairs.representation.eq(mode) & pairs.weighting.eq("unique_inputs")]
    for task in [t for t in layout.TASK_ORDER if t in set(frame.task)]:
        rows = frame[frame.task.eq(task)]
        distance = "earth_movers" if task == "regression" else "total_variation"
        means = dataset_balanced(rows, ["model_a", "model_b"], ["top_agreement", distance])
        names = clustered_order(means, _models([*rows.model_a, *rows.model_b]))
        lower = pd.DataFrame(np.nan, index=names, columns=names)
        upper = lower.copy()
        for row in means.itertuples():
            i, j = sorted((names.index(row.model_a), names.index(row.model_b)))
            lower.iloc[j, i] = row.top_agreement
            upper.iloc[i, j] = getattr(row, distance)
        fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, 4.75))
        agreement = ax.imshow(np.ma.masked_invalid(lower.to_numpy(float)), cmap=style.AGREEMENT_CMAP, vmin=0, vmax=1)
        upper_values = upper.to_numpy(float)
        top = 2.0 if task == "regression" else 0.5
        dist = ax.imshow(np.ma.masked_invalid(upper_values), cmap="Oranges", vmin=0,
                         vmax=max(top, np.nanmax(upper_values) if np.isfinite(upper_values).any() else top))
        for matrix, image, fmt in ((lower, agreement, "{:.0f}"), (upper, dist, "{:.2f}" if task != "regression" else "{:.1f}")):
            values = matrix.to_numpy(float)
            for i, j in zip(*np.isfinite(values).nonzero()):
                value = values[i, j] * (100 if matrix is lower else 1)
                shown = fmt.format(value)
                shown = shown[1:] if shown.startswith("0.") else shown
                ax.text(j, i, shown, ha="center", va="center", fontsize=6.5,
                        color=_ink(image, values[i, j]))
        for k in range(len(names)):
            ax.add_patch(plt.Rectangle((k - .5, k - .5), 1, 1, color=style.FAINT_COLOR, linewidth=0))
        labels = [short(n) for n in names]
        ax.set_xticks(range(len(names)), labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=7)
        ax.set_yticks(range(len(names)), labels, fontsize=7)
        for ticks in (ax.get_xticklabels(), ax.get_yticklabels()):
            for tick, name in zip(ticks, names):
                tick.set_color(style.DECISION_COLORS.get(name, style.TEXT_COLOR))
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        label = ("Earth mover's distance over the nine levels (upper)" if task == "regression"
                 else "Total variation distance (upper)")
        fig.colorbar(agreement, ax=ax, shrink=0.6, pad=0.01, format=PercentFormatter(1)).set_label(
            "Same top answer (lower)", fontsize=7.5)
        fig.colorbar(dist, ax=ax, shrink=0.6, pad=0.04).set_label(label, fontsize=7.5)
        ax.set_title(f"{TASKS[task]} · {rows.dataset_id.nunique()} datasets · {MODE_LABELS[mode]}", loc="left",
                     color=style.TASK_COLORS[task], fontsize=9)
        yield fig, f"pairwise_agreement_{task}_{mode}", (
            f"Pairwise agreement between decision models on {TASKS[task].lower()} {MODE_LABELS[mode]}, averaged with "
            "equal weight over datasets; each dataset counts every distinct input that both models received once. "
            "Lower triangle: percentage of inputs on which the two models' most probable answers are identical "
            "(regression: the modal level); an exact tie for the top answer in either model abstains. Upper triangle: "
            + ("mean earth mover's distance between the two nine-level distributions, in level units (0–8)."
               if task == "regression" else "mean total variation distance between the probability vectors (0–1).")
            + " Models are ordered by average-linkage hierarchical clustering of 1 − top-answer agreement "
            "(optimal leaf order), so similar models sit next to each other. Similar answers do not imply accuracy."
        )


def clustered_order(means: pd.DataFrame, names: list[str]) -> list[str]:
    """Average-linkage clustering of 1 - top-answer agreement (optimal leaf order)."""
    from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
    from scipy.spatial.distance import squareform

    if len(names) < 3:
        return names
    distance = pd.DataFrame(np.nan, index=names, columns=names)
    for row in means.itertuples():
        if row.model_a in distance.index and row.model_b in distance.index:
            distance.loc[row.model_a, row.model_b] = distance.loc[row.model_b, row.model_a] = 1 - row.top_agreement
    values = distance.to_numpy(float)
    np.fill_diagonal(values, 0)
    if not np.isfinite(values).all():
        fill = np.nanmax(values) if np.isfinite(values).any() else 1.0
        values = np.where(np.isfinite(values), values, fill)
    condensed = squareform(values, checks=False)
    tree = optimal_leaf_ordering(linkage(condensed, method="average"), condensed)
    return [names[i] for i in leaves_list(tree)]


#: A1 profile strip panels: (metric, letter, x label, fixed x-limits, dotted reference line).
PROFILE_PANELS = (
    ("mean_top_probability", "a", "Top probability", (0, 1), None),
    ("mean_normalized_entropy", "b", "Normalized entropy", (0, 1), None),
    ("consensus_agreement", "c", "Same top answer\nas consensus", (0, 1), None),
    ("classification_auc", "d", "ROC-AUC\n(classification)", None, 0.5),
    ("regression_spearman", "e", "Spearman ρ\n(regression)", None, 0.0),
)
FAMILY_LABELS = {"decoder_pointer_head": "Decoder, pointer head", "decoder_lm_head": "Decoder, LM-head labels",
                 "bidirectional_encoder": "Bidirectional backbone", "other": "Other backbone",
                 "undisclosed": "Hosted, undisclosed", "not stated": "Not stated"}


NOT_STATED_MARKER = "$?$"


def family_marker(family):
    """Architecture marker; a family no local source states is marked with a question mark."""
    return style.ARCHITECTURE_MARKERS.get(family, NOT_STATED_MARKER)


def _hollow(marker):
    return marker not in ("x", NOT_STATED_MARKER)


def family_handles(families, **kwargs):
    return [Line2D([], [], ls="none", marker=family_marker(f), ms=4, color=style.PREDICTOR_INK,
                   mfc="none" if _hollow(family_marker(f)) else style.PREDICTOR_INK,
                   label=FAMILY_LABELS.get(f, f), **kwargs) for f in dict.fromkeys(families)]


def model_profile_strip(values: pd.DataFrame, properties: pd.DataFrame, *, order=None):
    """A1 summary: one row per decision model, five panels of per-dataset values (joint requests).

    ``values`` is `group_a.a1_dataset_values`; ``properties`` `model_properties.profile_table`.
    """
    from src.evaluation import group_a

    if values.empty:
        return
    providers = order or group_a.model_order(values.provider.unique(), properties)
    families = [group_a.family_of(p, properties) for p in providers]
    sizes = (properties.set_index("provider").parameters_billion if not properties.empty
             else pd.Series(dtype=float))
    counts = {m: values[values.metric.eq(m)].dataset_id.nunique() for m, *_ in PROFILE_PANELS}
    height = min(style.MAX_HEIGHT, 1.55 + 0.18 * len(providers))
    fig, axes = plt.subplots(1, 6, sharey=True, figsize=(style.WIDTH_FULL, height),
                             gridspec_kw={"width_ratios": [0.36, 1, 1, 1, 1, 1]})
    key = axes[0]
    for i, (provider, family) in enumerate(zip(providers, families)):
        marker = family_marker(family)
        key.plot(0.2, i, ls="none", marker=marker, ms=4, color=style.PREDICTOR_INK,
                 mfc="none" if _hollow(marker) else style.PREDICTOR_INK, mew=0.9)
        size = pd.to_numeric(sizes.get(provider), errors="coerce")
        key.text(0.5, i, f"{size:.0f}B" if pd.notna(size) else "–", ha="left", va="center", fontsize=7,
                 color=style.QUANTITY_COLOR)
    key.set_xlim(0, 1.3)
    key.set_xticks([])
    key.set_yticks(range(len(providers)), [short(p) for p in providers], fontsize=7.5)
    for tick, provider in zip(key.get_yticklabels(), providers):
        tick.set_color(style.DECISION_COLORS.get(provider, style.TEXT_COLOR))
    key.set_ylim(len(providers) - 0.5, -0.5)
    for spine in key.spines.values():
        spine.set_visible(False)
    key.tick_params(axis="y", length=0)
    rules = [i for i in range(1, len(families)) if families[i] != families[i - 1]]
    for position in rules:
        key.axhline(position - 0.5, color=style.GRID_COLOR, lw=0.8, zorder=0)
    for ax, (metric, letter, xlabel, limits, reference) in zip(axes[1:], PROFILE_PANELS):
        frame = values[values.metric.eq(metric)]
        for i, provider in enumerate(providers):
            v = frame.loc[frame.provider.eq(provider), "value"].dropna().to_numpy(float)
            if not len(v):
                continue
            jitter = np.linspace(-0.22, 0.22, len(v)) if len(v) > 1 else np.zeros(1)
            ax.scatter(v, i + jitter, s=5, color=DOT_COLOR, alpha=0.6, linewidths=0, zorder=2)
            ax.plot([np.median(v)] * 2, [i - 0.36, i + 0.36], color=style.MEDIAN_COLOR, lw=1.6, zorder=3)
        if limits:
            ax.set_xlim(*limits)
        elif not frame.empty:
            low = np.nanmin(frame.value.to_numpy(float))
            ax.set_xlim(min(reference - 0.1, np.floor(low * 10) / 10), 1)
        if reference is not None:
            ax.axvline(reference, color=style.WINDOW_COLOR, lw=0.7, ls=":", zorder=1)
        ax.set_title(f"{letter}  N = {counts[metric]}", loc="left", fontsize=7.5)
        ax.set_xlabel(xlabel, fontsize=7.5)
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", length=0)
        layout.light_grid(ax)
        for position in rules:
            ax.axhline(position - 0.5, color=style.GRID_COLOR, lw=0.8, zorder=0)
    handles = family_handles(families)
    handles += [Line2D([], [], ls="none", marker="o", ms=3, color=DOT_COLOR, label="One dataset"),
                Line2D([], [], color=style.MEDIAN_COLOR, lw=1.6, label="Median over datasets")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=7,
               columnspacing=1.0, handletextpad=0.3)
    yield fig, "model_profile_strip", (
        f"Profile of {len(providers)} decision models on joint requests (rows, grouped by architecture family as "
        "marked in the first column, with the verified parameter count; – = not stated). Each grey dot is one "
        "dataset and the dark bar the median over datasets; N is the number of datasets per panel. (a) mean "
        "probability of the most probable answer and (b) Shannon entropy divided by the log of the number of "
        "options, over each dataset's distinct inputs (regression: nine ordered evidence levels). (c) share of "
        "distinct inputs whose most probable answer equals that of the leave-one-out consensus (equal-weight mean "
        "distribution of all other models; ties abstain). (d) stand-alone ROC-AUC (binary) or macro one-vs-rest "
        "ROC-AUC (multiclass) of the stored joint-request probabilities on the classification datasets (dotted "
        "line 0.5). (e) Spearman correlation between the expected evidence level and the target on the regression "
        "datasets (dotted line 0; rank only, log loss is undefined there). Nothing is fitted or recalibrated, and "
        "classification and regression quality are not pooled."
    )
