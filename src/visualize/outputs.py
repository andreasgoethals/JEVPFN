"""A1 (exp01_outputs) figures at final A4 width: output shape, pairwise agreement, consensus.

Every view uses distinct inputs (``unique_inputs`` weighting). Per-column text fields are
averaged within a dataset first, so each dataset counts once per request mode.
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
        names = _models([*rows.model_a, *rows.model_b])
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
                ax.text(j, i, fmt.format(value), ha="center", va="center", fontsize=5.5,
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
            + " Similar answers do not imply accuracy."
        )


def consensus_figure(loo, mode="joint"):
    """How often does each model give the answer of the other models' mean distribution?"""
    if loo.empty or "top_agreement" not in loo:
        return
    frame = per_dataset(loo, mode, ["top_agreement"])
    names = _models(frame.provider)
    order, tasks = _dataset_order(frame)
    matrix = frame.pivot(index="dataset", columns="provider", values="top_agreement").reindex(index=order, columns=names)
    fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 1.3 + .2 * len(order))))
    values = matrix.to_numpy(float)
    image = ax.imshow(values, cmap=style.AGREEMENT_CMAP, vmin=0, vmax=1, aspect="auto")
    for i, j in zip(*np.isfinite(values).nonzero()):
        ax.text(j, i, f"{100 * values[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                color=_ink(image, values[i, j]))
    layout.dataset_axis(ax, order, tasks, separators=False, fontsize=7.5)
    for position in range(1, len(tasks)):
        if tasks[position] != tasks[position - 1]:
            ax.axhline(position - .5, color="white", linewidth=2.5)
    ax.set_xticks(range(len(names)), [short(n) for n in names], rotation=40, ha="left", rotation_mode="anchor",
                  fontsize=7)
    ax.xaxis.tick_top()
    for tick, name in zip(ax.get_xticklabels(), names):
        tick.set_color(style.DECISION_COLORS.get(name, style.TEXT_COLOR))
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(image, ax=ax, shrink=.6, ticks=[0, .25, .5, .75, 1], format=PercentFormatter(1)).set_label(
        "Same top answer as the consensus", fontsize=7.5)
    yield fig, f"consensus_agreement_{mode}", (
        f"Agreement of each decision model with the leave-one-out consensus on {MODE_LABELS[mode]}: for every distinct "
        "input, the consensus is the equal-weight mean probability distribution of all other models that received it "
        "(at least two). Cells give the percentage of inputs on which the model's most probable answer equals the "
        "consensus's (regression: modal level); a tie on either side abstains. Consensus is not truth, and related "
        "model families (for example three Qwen3.8-27B derivatives) can dominate it."
    )
