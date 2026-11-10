"""How alike are the decision models? Five label-free views at final A4 print size.

1. Pairwise similarity: same answer (lower triangle) and rank correlation (upper triangle).
2. Model profiles: distance from the other models and output concentration.
3. Agreement with the other models' majority, per dataset.
4. All-model consensus per dataset, and how it changes with the text shown.
5. The most disputed input of each task, with every model's probabilities.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from src.evaluation.decision_comparison import dataset_balanced
from src.visualize import layout, style

TASKS = layout.TASK_LABELS
#: Write cell values only while they stay at least 7 pt on the printed page.
ANNOTATE_MAX_MODELS = 8


def short(name):
    return style.DECISION_SHORT.get(name, name)


def _joint(result, name, weighting="unique_inputs"):
    frame = result.table(name)
    return frame[frame.representation.eq("joint") & frame.weighting.eq(weighting)].copy() if not frame.empty else frame


def _models(names):
    """Canonical roster order, so a model keeps its row and colour in every figure."""
    names = set(names)
    return [n for n in style.DECISION_SHORT if n in names] + sorted(names - set(style.DECISION_SHORT))


def _datasets(frame):
    """Task groups in the shared order; catalog (alphabetical) order within each task."""
    rows = frame.drop_duplicates("dataset")[["dataset", "task"]]
    rows = layout.ordered(rows.sort_values("dataset"), task="task")
    return rows.dataset.tolist(), rows.task.tolist()


def _tasks(frame):
    return [t for t in layout.TASK_ORDER if t in set(frame.task)]


def _ink(image, value):
    rgba = image.cmap(image.norm(value))
    return "white" if np.dot(rgba[:3], [0.2126, 0.7152, 0.0722]) < 0.5 else style.TEXT_COLOR


def _number(value):
    return f"{value:.2f}".replace("0.", ".", 1) if value >= 0 else f"{value:.2f}".replace("-0.", "−.", 1)


def similarity_figure(result):
    """Is one decision model interchangeable with another? Mean over datasets per task."""
    pairs = _joint(result, "pairs")
    if pairs.empty:
        return
    tasks = _tasks(pairs)
    fig = plt.figure(figsize=(style.WIDTH_FULL, 3.15))
    grid = fig.add_gridspec(2, 2 * len(tasks), height_ratios=[1, 0.045])
    images = {}
    for column, task in enumerate(tasks):
        frame = pairs[pairs.task.eq(task)]
        names = _models([*frame.model_a, *frame.model_b])
        means = dataset_balanced(frame, ["model_a", "model_b"], ["agreement", "representation_spearman"])
        lower = pd.DataFrame(np.nan, index=names, columns=names)
        upper = lower.copy()
        for row in means.itertuples():
            i, j = sorted((names.index(row.model_a), names.index(row.model_b)))
            lower.iloc[j, i] = row.agreement
            upper.iloc[i, j] = row.representation_spearman
        ax = fig.add_subplot(grid[0, 2 * column:2 * column + 2])
        images["agreement"] = ax.imshow(np.ma.masked_invalid(lower.to_numpy(float)), cmap=style.AGREEMENT_CMAP, vmin=0, vmax=1)
        images["spearman"] = ax.imshow(np.ma.masked_invalid(upper.to_numpy(float)), cmap=style.AGREEMENT_DIVERGING, vmin=-1, vmax=1)
        if len(names) <= ANNOTATE_MAX_MODELS:
            for matrix, image in ((lower, images["agreement"]), (upper, images["spearman"])):
                for i, j in zip(*np.isfinite(matrix.to_numpy(float)).nonzero()):
                    value = matrix.iloc[i, j]
                    ax.text(j, i, _number(value), ha="center", va="center", fontsize=7, color=_ink(image, value))
        for k in range(len(names)):
            ax.add_patch(plt.Rectangle((k - .5, k - .5), 1, 1, color=style.FAINT_COLOR, linewidth=0))
        labels = [short(n) for n in names]
        ax.set_xticks(range(len(names)), labels, rotation=50, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(names)), labels if column == 0 else [])
        for ticks in (ax.get_xticklabels(), ax.get_yticklabels() if column == 0 else []):
            for tick, name in zip(ticks, names):
                tick.set_color(style.DECISION_COLORS.get(name, style.TEXT_COLOR))
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{'abc'[column]}  {TASKS[task]} · {frame.dataset_id.nunique()} datasets", loc="left",
                     color=style.TASK_COLORS[task], fontsize=9)
    for k, (key, label, ticks) in enumerate((
        ("agreement", "Same answer (lower triangle)", [0, .5, 1]),
        ("spearman", "Rank correlation, Spearman ρ (upper triangle)", [-1, 0, 1]),
    )):
        cax = fig.add_subplot(grid[1, k * len(tasks):(k + 1) * len(tasks)])
        bar = fig.colorbar(images[key], cax=cax, orientation="horizontal", ticks=ticks)
        bar.set_label(label, fontsize=8)
    yield fig, "decision_model_similarity", (
        "Pairwise similarity of the decision models' joint-text features, averaged with equal weight over the datasets of each task. "
        "Each dataset counts every distinct nonmissing input once. Lower triangle: share of inputs on which two models choose the same answer "
        "(binary: positive-class probability ≥ 0.5; multiclass: most probable class; regression: sign of the expected −4…+4 evidence score). "
        "Upper triangle: Spearman rank correlation of the features across inputs (binary: positive-class probability; multiclass: mean over "
        "classes of the correlation of matching class probabilities; regression: expected score). Same answers and similar rankings are "
        "different properties; neither measures accuracy."
    )


def peer_majority_figure(table):
    """Which models deviate from the others, and on which datasets?"""
    if table.empty:
        return
    names = _models(table.provider)
    order, tasks = _datasets(table)
    matrix = table.pivot(index="dataset", columns="provider", values="agreement").reindex(index=order, columns=names)
    fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 1.2 + .2 * len(order))))
    image = ax.imshow(matrix.to_numpy(float), cmap=style.AGREEMENT_CMAP, vmin=0, vmax=1, aspect="auto")
    for i, j in zip(*np.isfinite(matrix.to_numpy(float)).nonzero()):
        value = matrix.iloc[i, j]
        ax.text(j, i, f"{100 * value:.0f}", ha="center", va="center", fontsize=7, color=_ink(image, value))
    layout.dataset_axis(ax, order, tasks, separators=False)
    for position in range(1, len(tasks)):
        if tasks[position] != tasks[position - 1]:
            ax.axhline(position - .5, color="white", linewidth=2.5)
    ax.set_xticks(range(len(names)), [short(n) for n in names], rotation=30, ha="left", rotation_mode="anchor")
    ax.xaxis.tick_top()
    for tick, name in zip(ax.get_xticklabels(), names):
        tick.set_color(style.DECISION_COLORS.get(name, style.TEXT_COLOR))
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, shrink=.6, ticks=[0, .25, .5, .75, 1], format=PercentFormatter(1))
    bar.set_label("Agrees with majority of the other models")
    yield fig, "agreement_with_peer_majority", (
        "Share of distinct nonmissing joint-text inputs on which a model's answer equals the plurality answer of all other compared models "
        "(binary: positive-class probability ≥ 0.5; multiclass: most probable class; regression: sign of the expected evidence score). "
        "Inputs on which the other models are tied are excluded; their counts are in the accompanying table. "
        "Cell values are percentages. Low values mark systematic deviation from the other models, not errors."
    )


def model_profile_figure(result):
    """Does a model differ from its peers, and how concentrated are its probabilities?"""
    consensus = _joint(result, "consensus")
    distributions = _joint(result, "distributions")
    if consensus.empty:
        return
    entropy = distributions[distributions.variable.eq("entropy")].rename(columns={"mean": "value"})
    distance = consensus.rename(columns={"tv_to_others": "value"})
    tasks = _tasks(consensus)
    names = _models(consensus.provider)
    fig, axes = plt.subplots(2, len(tasks), figsize=(style.WIDTH_FULL, 1.25 + .42 * len(names)),
                             sharey=True, sharex="row", squeeze=False)
    rows = ((distance, "Distance from the other models\n(total variation)"),
            (entropy, "Output entropy\n(0 = one answer, 1 = uniform)"))
    for r, (frame, xlabel) in enumerate(rows):
        for c, task in enumerate(tasks):
            ax = axes[r, c]
            part = frame[frame.task.eq(task)]
            for i, name in enumerate(names):
                values = part[part.provider.eq(name)].sort_values("dataset_id").value.to_numpy()
                if not len(values):
                    continue
                color = style.DECISION_COLORS.get(name, style.QUANTITY_COLOR)
                ax.plot([values.min(), values.max()], [i, i], color=color, linewidth=1, alpha=.45)
                ax.scatter(values, i + np.linspace(-.18, .18, len(values)), s=9, color=color, alpha=.6, linewidth=0)
                ax.scatter(values.mean(), i, marker="D", s=28, color=color, edgecolor="black", linewidth=.5, zorder=3)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, .25, .5, .75, 1], ["0", ".25", ".5", ".75", "1"])
            layout.light_grid(ax)
            if r == 0:
                ax.set_title(f"{TASKS[task]} · {part.dataset_id.nunique()} datasets", color=style.TASK_COLORS[task])
            ax.set_xlabel(xlabel if c == len(tasks) // 2 else "")
            if c == 0:
                ax.text(-.02, 1.02, "ab"[r], transform=ax.transAxes, fontweight="bold", ha="right", va="bottom")
    axes[0, 0].set_yticks(range(len(names)), [short(n) for n in names])
    for tick, name in zip(axes[0, 0].get_yticklabels(), names):
        tick.set_color(style.DECISION_COLORS.get(name, style.TEXT_COLOR))
    axes[0, 0].set_ylim(len(names) - .5, -.5)
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=3.5, color=style.QUANTITY_COLOR, alpha=.6, label="One dataset"),
               Line2D([], [], linestyle="none", marker="D", markersize=4.5, color=style.QUANTITY_COLOR,
                      markeredgecolor="black", label="Mean over datasets")]
    fig.legend(handles=handles, loc="outside lower center", ncols=2, frameon=False)
    yield fig, "decision_model_profiles", (
        "Per-model profile of the joint-text features. (a) Mean total variation between a model's probability vector and the average vector of "
        "the other compared models (0 identical, 1 disjoint); the model itself is excluded from its reference. (b) Mean Shannon entropy of a "
        "model's probability vector divided by the logarithm of the number of answer options (two, the dataset's class count, or nine evidence "
        "levels). Dots are datasets (each distinct nonmissing input counted once), lines their range and diamonds the equal-weight mean. "
        "Low entropy means concentrated, not correct or calibrated, probabilities."
    )


def consensus_figure(result):
    """How often do all models agree, and does it depend on the text they are shown?"""
    votes = _joint(result, "vote_patterns")
    summary = result.table("row_summary")
    if votes.empty:
        return
    models = int(votes.models.iloc[0])
    votes["largest"] = votes.pattern.map(lambda s: 0 if s == "tied top probability" else int(s.split(":")[0]))
    order, tasks = _datasets(votes)
    shares = votes.pivot_table(index="dataset", columns="largest", values="fraction", aggfunc="sum", fill_value=0).reindex(order).fillna(0)
    fig, (left, right) = plt.subplots(1, 2, figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 1.35 + .19 * len(order))),
                                      sharey=True, width_ratios=[1.25, 1])
    start = np.zeros(len(order))
    cmap = plt.get_cmap(style.AGREEMENT_CMAP)
    handles = []
    for n in sorted(shares.columns, reverse=True):
        color = cmap(.2 + .8 * n / models) if n else style.MISSING_COLOR
        left.barh(range(len(order)), shares[n], left=start, color=color, height=.74, linewidth=0)
        handles.append(Patch(color=color, label=f"{n}/{models}" if n else "tie"))
        start += shares[n].to_numpy()
    for i, value in enumerate(shares.get(models, pd.Series(0, index=order))):
        if value >= .1:
            left.text(value / 2, i, f"{value:.0%}", ha="center", va="center", fontsize=7, color="white")
    left.set_xlim(0, 1)
    left.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    left.set_xlabel("Share of distinct joint-text inputs")
    left.set_title("a  Largest agreeing group", loc="left")
    left.legend(handles=handles, ncols=len(handles), loc="lower left", bbox_to_anchor=(0, 1.06), frameon=False,
                handlelength=1, handletextpad=.3, columnspacing=.7, fontsize=7)
    layout.dataset_axis(left, order, tasks)
    fields = summary[summary.weighting.eq("unique_inputs")]
    for y, dataset in enumerate(order):
        separate = fields[fields.dataset.eq(dataset) & fields.representation.eq("per_column")].unanimous
        joint = fields[fields.dataset.eq(dataset) & fields.representation.eq("joint")].unanimous
        if len(separate) > 1:
            right.plot([separate.min(), separate.max()], [y, y], color=style.MODE_COLORS["per_column"], alpha=.45, linewidth=1)
        right.scatter(separate, np.full(len(separate), y), s=12, color=style.MODE_COLORS["per_column"], alpha=.8, linewidth=0, zorder=2)
        right.scatter(joint, np.full(len(joint), y), s=30, marker="D", color=style.MODE_COLORS["joint"],
                      edgecolor="white", linewidth=.5, zorder=3)
    layout.task_rules(right, tasks)
    layout.light_grid(right)
    right.set_xlim(-.02, 1.02)
    right.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    right.set_xlabel(f"Inputs on which all {models} models agree")
    right.set_title("b  One field or all fields", loc="left")
    right.tick_params(axis="y", length=0)
    right.legend(handles=[Line2D([], [], linestyle="none", marker="o", markersize=4, color=style.MODE_COLORS["per_column"], label="One field (per-column)"),
                          Line2D([], [], linestyle="none", marker="D", markersize=5, color=style.MODE_COLORS["joint"], label="All fields (joint)")],
                 ncols=2, loc="lower left", bbox_to_anchor=(0, 1.06), frameon=False, handletextpad=.2, columnspacing=.8, fontsize=7)
    yield fig, "all_model_consensus", (
        f"(a) For every distinct nonmissing joint-text input, the number of the {models} models that give the most common answer "
        "(binary: positive-class probability ≥ 0.5; multiclass: most probable class; regression: sign of the expected evidence score); "
        "white labels give the unanimous share. 'Tie' marks inputs where a model's highest class probability is tied. "
        "(b) Unanimous share with one text field per request (circles, one per field; line spans the fields) and with all fields "
        "jointly (diamond, equal to the unanimous share in a). Field and joint requests contain different information and can have "
        "different nonmissing denominators; consensus is not accuracy."
    )


def _case(cases, task):
    frame = cases[cases.category.eq("disputed") & cases.task.eq(task)]
    if frame.empty:
        return None
    row = frame.sort_values(["mean_pair_tv", "dataset_id", "row_id"], ascending=[False, True, True]).iloc[0]
    values = pd.DataFrame(json.loads(row.probabilities), index=json.loads(row.labels))
    return row, values[_models(values.columns)]


def _scalar_case(ax, row, values, task):
    names = list(values.columns)
    if task == "regression":
        scores = values.apply(lambda p: float(np.dot(p.to_numpy(), np.arange(-4, 5))))
        ax.axvline(0, color=style.GRID_COLOR, linewidth=1)
        ax.set_xlim(-4.2, 4.2)
        ax.set_xticks(range(-4, 5))
        ax.set_xlabel("Expected evidence score (−4 lower … +4 higher)")
    else:
        scores = values.iloc[1]
        ax.axvline(.5, color=style.GRID_COLOR, linewidth=1)
        ax.set_xlim(-.03, 1.03)
        ax.set_xlabel(f"P({textwrap_label(values.index[1])})")
    for i, name in enumerate(names):
        color = style.DECISION_COLORS.get(name, style.QUANTITY_COLOR)
        ax.plot([0 if task == "regression" else .5, scores[name]], [i, i], color=color, linewidth=1, alpha=.5)
        ax.scatter(scores[name], i, s=26, color=color, zorder=3)
    ax.set_yticks(range(len(names)), [short(n) for n in names])
    for tick, name in zip(ax.get_yticklabels(), names):
        tick.set_color(style.DECISION_COLORS.get(name, style.TEXT_COLOR))
    ax.set_ylim(len(names) - .5, -.5)
    ax.tick_params(axis="y", length=0)
    ax.set_title(f"{'a' if task == 'binary_classification' else 'c'}  {row.dataset} · {row.row_id}",
                 loc="left", color=style.TASK_COLORS[task])


def textwrap_label(label, width=26):
    label = str(label)
    return label if len(label) <= width else label[:width - 1] + "…"


def disputed_cases_figure(cases):
    """Concrete disagreement: one maximally disputed input per task, every probability."""
    if cases.empty:
        return
    found = {task: _case(cases, task) for task in layout.TASK_ORDER}
    if not any(found.values()):
        return
    fig = plt.figure(figsize=(style.WIDTH_FULL, 3.9))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.15])
    notes = []
    for position, task in ((grid[0, 0], "binary_classification"), (grid[1, 0], "regression")):
        if found[task]:
            row, values = found[task]
            _scalar_case(fig.add_subplot(position), row, values, task)
            notes.append(f"{TASKS[task].lower()}: {row.dataset} {row.row_id} (mean pairwise total variation {row.mean_pair_tv:.2f})")
    if found["multiclass_classification"]:
        row, values = found["multiclass_classification"]
        shown = values.max(axis=1).ge(.1)
        grouped = int((~shown).sum())
        if grouped:
            rest = values.loc[~shown].sum()
            values = values.loc[shown].copy()
            values.loc[f"{grouped} other classes (sum)"] = rest
        ax = fig.add_subplot(grid[:, 1])
        image = ax.imshow(values.to_numpy(float), cmap=style.AGREEMENT_CMAP, vmin=0, vmax=1, aspect="auto")
        for i, j in zip(*np.nonzero(values.to_numpy(float) >= .05)):
            value = values.iloc[i, j]
            ax.text(j, i, f"{value:.2f}".lstrip("0"), ha="center", va="center", fontsize=7, color=_ink(image, value))
        ax.set_yticks(range(len(values)), [textwrap_label(v, 24) for v in values.index])
        ax.set_xticks(range(len(values.columns)), [short(n) for n in values.columns], rotation=50, ha="right", rotation_mode="anchor")
        for tick, name in zip(ax.get_xticklabels(), values.columns):
            tick.set_color(style.DECISION_COLORS.get(name, style.TEXT_COLOR))
        ax.tick_params(length=0)
        ax.set_title(f"b  {row.dataset} · {row.row_id}", loc="left", color=style.TASK_COLORS["multiclass_classification"])
        fig.colorbar(image, ax=ax, shrink=.7, ticks=[0, .5, 1]).set_label("Class probability")
        notes.append(f"multiclass: {row.dataset} {row.row_id} (mean pairwise total variation {row.mean_pair_tv:.2f}"
                     + (f"; classes below 0.10 in every model are summed, {grouped} classes)" if grouped else ")"))
    yield fig, "most_disputed_inputs", (
        "For each task, the distinct joint-text input with the largest mean pairwise total variation between the models' probability vectors, "
        "selected without target values: " + "; ".join(notes) + ". "
        "(a) positive-class probability, (b) class probabilities (values below 0.05 not printed), (c) expected evidence score; lines start at the decision boundary. "
        "The verified question and source text of every case are in the notebook's case browser. These are deliberate extremes, not typical inputs."
    )
