"""Print-width coverage, native references and cost views for separate experiments."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src.data.loaders import load_catalog
from src.evaluation.endpoints import LABELS
from src.utils.config import load_yaml
from src.visualize import style


def _names():
    return {r["id"]: r["name"] for r in load_catalog()["datasets"]}


def coverage_figures(table, title):
    if table.empty or not table.completed.gt(0).any():
        # A page of identical 0/N cells adds no information to the coverage table.
        return
    names = _names()
    models = list(dict.fromkeys(table.model_id))
    datasets = list(dict.fromkeys(table.dataset_id))
    for page, start in enumerate(range(0, len(datasets), 9), 1):
        selected = datasets[start:start + 9]
        fraction = np.zeros((len(selected), len(models)))
        labels = {}
        for row in table[table.dataset_id.isin(selected)].itertuples():
            i, j = selected.index(row.dataset_id), models.index(row.model_id)
            fraction[i, j] = row.completed / row.planned if row.planned else np.nan
            labels[i, j] = f"{row.completed:g}/{row.planned:g}"
        fig, ax = plt.subplots(figsize=style.DASHBOARD_SIZE)
        image = ax.imshow(fraction, vmin=0, vmax=1, cmap=style.OVERVIEW_CMAP, aspect="auto")
        ax.set_yticks(range(len(selected)), [names.get(d, d) for d in selected])
        ax.set_xticks(range(len(models)), [style.PREDICTOR_LABELS.get(m, m) for m in models], rotation=35, ha="right")
        for (i, j), label in labels.items():
            ax.text(j, i, label, ha="center", va="center", fontsize=7,
                    color="white" if fraction[i, j] > 0.6 else style.TEXT_COLOR)
        fig.colorbar(image, ax=ax, label="Fraction complete", shrink=0.7)
        ax.set_title(title + f" · page {page}")
        yield fig, f"coverage_{page}", (
            "Completed/planned configurations by dataset and predictor. A predictive configuration "
            "requires five defined headline-metric folds; feature inventory uses saved representation "
            "manifests and does not replace row/blob validation. Zero denotes no qualifying saved evidence, "
            "not zero predictive performance. Counts precede unsupported-resource/width review."
        )


def native_figures(summary):
    data = summary[summary.phase.eq("experiment_3") & summary.defined_folds.eq(5)] if not summary.empty else summary
    if data.empty:
        return
    names = _names()
    for task, group in data.groupby("task_type"):
        if group.duplicated(["dataset_id", "model_id"]).any():
            raise ValueError("Select one native checkpoint/campaign before plotting")
        datasets = sorted(group.dataset_id.unique(), key=lambda d: names.get(d, d))
        fig, ax = plt.subplots(figsize=style.native_reference_size(len(datasets)))
        for pos, model in enumerate(style.NATIVE_COLORS):
            part = group[group.model_id.eq(model)]
            ax.scatter(part["mean"], [datasets.index(d) + (pos - 1)*0.16 for d in part.dataset_id],
                       color=style.NATIVE_COLORS[model], marker=style.NATIVE_MARKERS[model], s=25,
                       label=style.PREDICTOR_LABELS[model])
        ax.set_yticks(range(len(datasets)), [names.get(d, d) for d in datasets])
        ax.set_ylim(len(datasets) - 0.5, -0.5)
        ax.grid(axis="x", color=style.GRID_COLOR)
        ax.set_xlabel(LABELS[group.metric.iloc[0]])
        ax.set_title("Original-text systems · " + task.replace("_", " "))
        fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=3)
        yield fig, f"native_{task}", (
            "Original-text reference systems on the same dataset and shared outer splits. Each marker "
            "is the mean of five defined held-out-fold scores. These systems change the predictor and "
            "text treatment together; their differences do not isolate a text representation. Missing "
            "or unsupported systems are absent, not zero. R² values, including negatives, are unmodified."
        )


def burden_figures(table):
    if table.empty or "seconds.fit" not in table or "columns.representation" not in table:
        return
    for model, group in table.groupby("model_id"):
        group = group.dropna(subset=["seconds.fit", "columns.representation"])
        if group.empty:
            continue
        fig, ax = plt.subplots(figsize=style.DASHBOARD_SIZE)
        for phase, part in group.groupby("phase"):
            ax.scatter(part["columns.representation"], part["seconds.fit"],
                       color=style.FAMILY_COLORS[phase], alpha=0.35, s=10, label=phase.replace("_", " "))
        if group["seconds.fit"].gt(0).all():
            ax.set_yscale("log")
        ax.set(xlabel="Columns after representation/PCA, before predictor category encoding",
               ylabel="Mean outer-fold fit time (seconds)", title=style.PREDICTOR_LABELS.get(model, model))
        ax.grid(color=style.GRID_COLOR, alpha=0.4)
        ax.legend()
        yield fig, f"burden_{model}", (
            "Measured representation width and outer-fit time for saved five-fold configurations. "
            "Each point is one dataset/recipe; fit time includes the library fit/preprocessing scope. "
            "Different datasets and hardware remain identified in the tables. Extraction cost is "
            "recorded separately and is not charged to every CV fit. No causal width/runtime relationship is assumed."
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
        fig.suptitle(f"{task.replace('_', ' ')} · {len(expected)} matched datasets · {LABELS[task_rows.metric.iloc[0]]} differences")
        yield fig, f"cross_system_{task}", (
            "Descriptive comparison with each native-text reference, pairing the same datasets/splits/folds "
            "but changing the predictor. Each point is a prespecified fixed-predictor representation recipe, "
            "averaged equally across all selected datasets of the task after five paired folds per dataset. "
            "No winning decision model, encoder or PCA width is selected. Recipe counts are unequal between "
            "families; points are not independent samples, error bars or a statistical test. Positive favors "
            "the fixed recipe. Incomplete task-wide comparisons are retained in tables but not plotted."
        )


def aligned_control_figures(contrasts):
    if contrasts.empty:
        return
    from src.visualize.decision_comparison import LABELS as PROVIDERS
    from src.visualize.paired import METRIC_VIEWS, _dataset_effects

    labels = {"non_text_plus_both__permuted__main": "Shuffled",
              "non_text_plus_both__retained_view__probabilities": "Full probabilities",
              "non_text_plus_both__retained_view__probabilities_plus_confidence": "Probabilities + uncertainty"}
    for task, (metric, direction, _) in METRIC_VIEWS.items():
        for model, part in contrasts[contrasts.task_type.eq(task) & contrasts.metric.eq(metric)].groupby("model_id"):
            means = _dataset_effects(part, ["feature_set", "provider", "dataset_id", "split_id", "arm_id"], task, direction)
            if means.empty:
                continue
            views = [v for v in labels if v in set(means.feature_set)]
            providers = [p for p in PROVIDERS if p in set(means.provider)]
            fig, axes = plt.subplots(1, len(views), figsize=style.DASHBOARD_SIZE, sharey=True, sharex=True, squeeze=False)
            for ax, view in zip(axes[0], views):
                for pos, provider in enumerate(providers):
                    values = means.loc[means.feature_set.eq(view) & means.provider.eq(provider), "effect"]
                    ax.scatter(values, pos + np.linspace(-0.12, 0.12, len(values)), s=9, alpha=0.4, color=style.MEDIAN_COLOR)
                    if len(values):
                        ax.scatter(values.mean(), pos, s=20, marker="D", color=style.DECISION_COLORS[provider])
                ax.set_title(labels[view], fontsize=8)
                ax.axvline(0, color=style.GRID_COLOR)
                ax.set_yticks(range(len(providers)), [PROVIDERS[p] for p in providers])
                ax.set_xlabel("Control − aligned main", fontsize=7)
            axes[0, 0].invert_yaxis()
            fig.suptitle(f"{style.PREDICTOR_LABELS.get(model, model)} · {task.replace('_', ' ')}")
            yield fig, f"aligned_controls_{model}_{task}", (
                "Each control is paired with the same predictor, source feature artifacts and split/fold "
                "using aligned separate-plus-joint main features. Points are per-dataset means of five "
                "defined fold differences; diamonds average the available dataset means. Positive favors "
                "the control in the prespecified AUC/R² metric. Shuffling preserves each row's missingness; "
                "retained views reuse full responses without new inference. Coverage can differ and is "
                "reported in tables. No test-set recipe selection or confidence interval is implied."
            )
