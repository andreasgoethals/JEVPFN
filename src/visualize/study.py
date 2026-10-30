"""Print-width coverage, native references and cost views for separate experiments."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src.data.loaders import load_catalog
from src.evaluation.endpoints import LABELS
from src.utils.config import load_yaml
from src.visualize import layout, style


def _names():
    return {r["id"]: r["name"] for r in load_catalog()["datasets"]}


def _ordered_datasets(dataset_ids):
    """Catalog datasets in the project-wide task order, restricted to those present."""
    semantics = load_yaml("dataset_semantics")["datasets"]
    names = _names()
    present = set(dataset_ids)
    rows = [(d, semantics[d]["task_type"]) for d in names if d in present]
    rank = {t: i for i, t in enumerate(layout.TASK_ORDER)}
    rows.sort(key=lambda r: rank[r[1]])
    return [d for d, _ in rows], [t for _, t in rows], names


def coverage_figures(table, title):
    if table.empty or not table.completed.gt(0).any():
        # A page of identical 0/N cells adds no information to the coverage table.
        return
    datasets, tasks, names = _ordered_datasets(table.dataset_id)
    models = list(dict.fromkeys(table.model_id))
    fraction = np.full((len(datasets), len(models)), np.nan)
    labels = {}
    for row in table.itertuples():
        if row.dataset_id in datasets:
            i, j = datasets.index(row.dataset_id), models.index(row.model_id)
            fraction[i, j] = row.completed / row.planned if row.planned else np.nan
            labels[i, j] = f"{row.completed:g}/{row.planned:g}"
    fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 1.1 + 0.2 * len(datasets))))
    image = ax.imshow(fraction, vmin=0, vmax=1, cmap=style.OVERVIEW_CMAP, aspect="auto")
    for (i, j), label in labels.items():
        ax.text(j, i, label, ha="center", va="center", fontsize=6.5,
                color="white" if fraction[i, j] > 0.6 else style.TEXT_COLOR)
    layout.dataset_axis(ax, [names.get(d, d) for d in datasets], tasks, fontsize=7)
    ax.set_xticks(range(len(models)), [style.PREDICTOR_LABELS.get(m, m) for m in models], fontsize=7)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(image, ax=ax, label="share of planned configurations complete", shrink=0.7)
    yield fig, "coverage", (
        f"{title}: completed over planned configurations per dataset and predictor. A configuration "
        "counts as complete only with five defined outer folds of its headline metric; a stored "
        "feature inventory uses saved representation manifests instead. Zero means no qualifying "
        "saved evidence, not zero performance. Counts precede the width and resource review."
    )


def native_figures(summary):
    """Headline score of each native system per dataset, one panel per task."""
    data = summary[summary.phase.eq("experiment_3") & summary.defined_folds.eq(5)] if not summary.empty else summary
    if data.empty:
        return
    if data.duplicated(["dataset_id", "model_id"]).any():
        raise ValueError("Select one native checkpoint/campaign before plotting")
    datasets, tasks, names = _ordered_datasets(data.dataset_id)
    present = [t for t in layout.TASK_ORDER if t in tasks]
    counts = [tasks.count(t) for t in present]
    fig, axes = plt.subplots(len(present), 1, figsize=(style.WIDTH_FULL, min(
        style.MAX_HEIGHT, 1.0 + 0.24 * len(datasets) + 0.35 * len(present))),
        gridspec_kw={"height_ratios": counts})
    axes = np.atleast_1d(axes)
    for ax, task in zip(axes, present):
        rows = [d for d, t in zip(datasets, tasks) if t == task]
        group = data[data.task_type.eq(task)]
        for pos, model in enumerate(style.NATIVE_COLORS):
            part = group[group.model_id.eq(model)]
            ax.scatter(part["mean"], [rows.index(d) + (pos - 1) * 0.18 for d in part.dataset_id],
                       color=style.NATIVE_COLORS[model], marker=style.NATIVE_MARKERS[model], s=22,
                       zorder=3, label=style.PREDICTOR_LABELS[model])
        layout.dataset_axis(ax, [names.get(d, d) for d in rows], [task] * len(rows), fontsize=7,
                            separators=False)
        layout.light_grid(ax)
        ax.set_xlabel(LABELS[group.metric.iloc[0]], fontsize=7)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncols=3,
               fontsize=7, frameon=False)
    yield fig, "native_systems", (
        "Headline score of each original-text system on every dataset with five defined outer "
        "folds (mean over folds; ROC-AUC, macro one-vs-rest ROC-AUC or R²), on the same splits as "
        "Experiments 1 and 2. These systems change predictor and text treatment together, so their "
        "differences do not isolate a representation. Missing systems are absent, not zero; "
        "negative R² values are shown unmodified."
    )


def burden_figures(table):
    """Representation width against fit time, one small panel per predictor."""
    if table.empty or "seconds.fit" not in table or "columns.representation" not in table:
        return
    table = table.dropna(subset=["seconds.fit", "columns.representation"])
    models = [m for m in style.PREDICTOR_LABELS if m in set(table.model_id)]
    if not models:
        return
    columns = 3
    rows = int(np.ceil(len(models) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 1.9 * rows + 0.5)),
                             squeeze=False, sharex=True, sharey=True)
    for ax, model in zip(axes.flat, models):
        group = table[table.model_id.eq(model)]
        for phase, part in group.groupby("phase"):
            ax.scatter(part["columns.representation"], part["seconds.fit"], s=8, alpha=0.5,
                       linewidths=0, color=style.FAMILY_COLORS[phase])
        ax.set_xscale("log")
        if group["seconds.fit"].gt(0).all():
            ax.set_yscale("log")
        ax.set_title(style.PREDICTOR_LABELS[model], fontsize=8)
        layout.light_grid(ax, axis="both")
    for ax in axes.flat[len(models):]:
        ax.set_axis_off()
    for ax in axes[-1]:
        ax.set_xlabel("input columns (log)", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("fit time (s, log)", fontsize=7)
    names = {"experiment_1": "Decision features", "experiment_2": "Frozen embedding",
             "experiment_3": "Native text"}
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=4, color=c, label=names[p])
               for p, c in style.FAMILY_COLORS.items() if p in set(table.phase)]
    fig.legend(handles=handles, loc="outside upper center", ncols=len(handles), fontsize=7,
               frameon=False)
    yield fig, "fit_burden", (
        "Mean outer-fold fit time against the number of input columns after representation and "
        "PCA (before predictor-specific category encoding), for every saved five-fold "
        "configuration; one point per dataset and recipe. Extraction cost is recorded separately "
        "and is not charged to each fit. Datasets and hardware remain identified in the tables."
    )


def cross_system_figures(contrasts):
    if contrasts.empty:
        return
    semantics = load_yaml("dataset_semantics")["datasets"]
    for task, task_rows in contrasts.groupby("task_type"):
        expected = {d for d, s in semantics.items() if s["task_type"] == task}
        if task_rows.groupby("model_id").checkpoint_id.nunique().gt(1).any() or task_rows.groupby("reference_model").reference_checkpoint.nunique().gt(1).any():
            raise ValueError("Select one predictor/native checkpoint campaign before comparison")
        keys = ["model_id", "checkpoint_id", "reference_model", "reference_checkpoint", "family", "arm_id", "dataset_id", "split_id"]
        means = task_rows.groupby(keys, dropna=False).agg(effect=("difference", "mean"), n=("difference", "count")).reset_index()
        means = means[means.n.eq(5)]
        # arm_id is dataset-specific. Use the explicit representation comparison identity across datasets.
        ids = task_rows[["arm_id", "comparison_id"]].drop_duplicates()
        means = means.merge(ids, on="arm_id", validate="many_to_one")
        summaries = []
        for key, group in means.groupby(["model_id", "reference_model", "family", "comparison_id"]):
            if group.dataset_id.duplicated().any():
                raise ValueError("Multiple splits for a cross-system comparison")
            if set(group.dataset_id) == expected:
                summaries.append((*key, group.effect.mean()))
        if not summaries:
            continue
        import pandas as pd
        shown = pd.DataFrame(summaries, columns=["model", "reference", "family", "recipe", "effect"])
        models = list(style.MAIN_PREDICTORS)
        fig, axes = plt.subplots(1, 3, figsize=style.DASHBOARD_SIZE, sharey=True, sharex=True)
        colors = {"Non-text": style.MEDIAN_COLOR, "Decision": style.FAMILY_COLORS["experiment_1"],
                  "Embedding": style.FAMILY_COLORS["experiment_2"]}
        for ax, reference in zip(axes, style.NATIVE_COLORS):
            for model_pos, model in enumerate(models):
                for family_pos, (family, color) in enumerate(colors.items()):
                    values = shown.loc[shown.model.eq(model) & shown.reference.eq(reference) & shown.family.eq(family), "effect"]
                    ax.scatter(values, model_pos + (family_pos - 1)*0.16 + np.linspace(-0.035, 0.035, len(values)),
                               color=color, s=8, alpha=0.5)
            ax.axvline(0, color=style.GRID_COLOR)
            ax.set_title(style.PREDICTOR_LABELS[reference])
            ax.set_yticks(range(len(models)), [style.PREDICTOR_LABELS[m] for m in models])
            ax.set_xlabel("Fixed recipe − native")
        axes[0].invert_yaxis()
        fig.legend([Line2D([], [], color=c, marker="o", linestyle="none") for c in colors.values()],
                   list(colors), loc="outside lower center", ncol=3)
        fig.suptitle(f"{layout.TASK_LABELS[task]} · {len(expected)} matched datasets · {LABELS[task_rows.metric.iloc[0]]} differences", fontsize=9)
        yield fig, f"cross_system_{task}", (
            "Descriptive comparison with each native-text reference, pairing the same datasets/splits/folds "
            "but changing the predictor. Each point is a prespecified fixed-predictor representation recipe, "
            "averaged equally across all selected datasets of the task after five paired folds per dataset. "
            "No winning decision model, encoder or PCA width is selected. Recipe counts are unequal between "
            "families; points are not independent samples, error bars or a statistical test. Positive favors "
            "the fixed recipe. Incomplete task-wide comparisons are retained in tables but not plotted."
        )
