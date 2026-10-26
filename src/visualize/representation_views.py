"""Paper-width, saved-evidence-only views answering the representation questions."""

import matplotlib.pyplot as plt
import numpy as np

from src.visualize import style
from src.visualize.decision_comparison import LABELS
from src.visualize.paired import METRIC_VIEWS, MODES, _dataset_effects
from src.visualize.research import ENCODER_LABELS

RECIPES = [("joint", n) for n in (8, 16, 32, 64, 128, 256)] + [("per_column", 32)]


def direct_comparison_figures(contrasts):
    if contrasts.empty:
        return
    for task, (metric, direction, unit) in METRIC_VIEWS.items():
        data = contrasts[contrasts.task_type.eq(task) & contrasts.metric.eq(metric)]
        for (model, encoder), part in data.groupby(["model_id", "encoder"]):
            keys = ["provider", "feature_set", "dataset_id", "split_id", "arm_id",
                    "embedding_arm_id", "pca_scope", "pca_components"]
            means = _dataset_effects(part, keys, task, direction)
            if means.empty:
                continue
            providers = [p for p in LABELS if p in set(means.provider)]
            limit = max(float(means.effect.abs().max()), 0.001)
            fig, axes = plt.subplots(1, 3, figsize=style.DASHBOARD_SIZE, sharey=True)
            for ax, (mode, label) in zip(axes, MODES.items()):
                matrix = np.full((len(providers), len(RECIPES)), np.nan)
                for i, provider in enumerate(providers):
                    for j, (scope, width) in enumerate(RECIPES):
                        values = means.loc[means.provider.eq(provider) & means.feature_set.eq(mode)
                                           & means.pca_scope.eq(scope) & means.pca_components.eq(width), "effect"]
                        if len(values):
                            matrix[i, j] = values.mean()
                cmap = plt.get_cmap(style.EFFECT_CMAP).with_extremes(bad=style.FAINT_COLOR)
                artist = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
                ax.set_title(label)
                ax.set_xticks(range(7), ["J8", "J16", "J32", "J64", "J128", "J256", "P32"], rotation=90)
                ax.set_yticks(range(len(providers)), [LABELS[p] for p in providers])
                ax.tick_params(axis="both", labelsize=7, length=0)
            fig.colorbar(artist, ax=list(axes), shrink=0.75, label=unit.replace("augmented", "decision").replace("reference", "embedding"))
            fig.suptitle(f"{model} · {task.split('_')[0]}\nDecision features versus {ENCODER_LABELS[encoder]}", fontsize=9)
            yield fig, f"direct_{model}_{task}_{encoder}", (
                "Each cell compares decision features with the named frozen embedding recipe in the same predictor, "
                "dataset, checkpoint and CV split. Positive values favor decision features. J denotes joint PCA "
                "with that total width; P32 denotes 32 dimensions per text column. Dataset means require five defined "
                "paired folds; cells average their available dataset means without selecting winners. Gray cells lack "
                "complete paired evidence. Coverage can differ across cells: the accompanying table retains every "
                "dataset, recipe and fold. Regression reports percent reduction in mean RMSE relative to embeddings. "
                "These are descriptive comparisons, not confidence intervals or a best-model selection."
            )


def control_figures(paired):
    if paired.empty:
        return
    for task, (metric, direction, unit) in METRIC_VIEWS.items():
        data = paired[paired.phase.eq("experiment_5") & paired.reference.eq("non_text")
                      & paired.task_type.eq(task) & paired.metric.eq(metric)
                      & paired.inference_status.eq("prespecified")]
        for model, part in data.groupby("model_id"):
            means = _dataset_effects(part, ["feature_set", "dataset_id", "split_id", "arm_id"], task, direction)
            names = sorted(means.feature_set.unique())
            for page, start in enumerate(range(0, len(names), 7), 1):
                selected = names[start:start + 7]
                fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 1.5 + len(selected)*0.48)))
                for i, name in enumerate(selected):
                    values = means.loc[means.feature_set.eq(name), "effect"]
                    ax.scatter(values, i + np.linspace(-0.13, 0.13, len(values)), s=12, color=style.QUANTITY_COLOR, alpha=0.5)
                    ax.scatter(values.mean(), i, marker="D", s=24, color=style.TEXT_COLOR)
                import textwrap
                ax.set_yticks(range(len(selected)), [textwrap.fill(n.replace("non_text_plus_", "").replace("__", " / "), 30) for n in selected])
                ax.set_ylim(len(selected)-0.5, -0.5)
                ax.axvline(0, color=style.GRID_COLOR)
                ax.set_xlabel(unit)
                ax.set_title(f"{model} · {task.split('_')[0]} · controls versus non-text")
                yield fig, f"controls_{model}_{task}_{page}", (
                    "Prespecified controls compared with the same predictor using non-text columns. Points are "
                    "dataset means of five defined paired fold differences; diamonds average the available dataset "
                    "means. Positive values indicate improvement. Outcome-selected exploratory combinations are "
                    "excluded. Missing controls are not replaced by zeros; coverage and the complete control names "
                    "remain in the tables. Fold variability is not used as inferential uncertainty."
                )


def overlap_figures(paired):
    if paired.empty:
        return
    for task, (metric, direction, unit) in METRIC_VIEWS.items():
        for model, part in paired[paired.phase.eq("experiment_1") & paired.reference.eq("non_text")
                                  & paired.task_type.eq(task) & paired.feature_set.isin(MODES)].groupby("model_id"):
            cohort_values = {}
            for cohort in style.COHORT_COLORS:
                chosen = part[part.metric.eq(f"text.{cohort}.{metric}")]
                means = _dataset_effects(chosen, ["provider", "feature_set", "dataset_id", "split_id", "arm_id"], task, direction)
                if not means.empty:
                    cohort_values[cohort] = means
            if not cohort_values:
                continue
            providers = [p for p in LABELS if any(p in set(m.provider) for m in cohort_values.values())]
            fig, axes = plt.subplots(1, 3, figsize=style.DASHBOARD_SIZE, sharey=True, sharex=True)
            for ax, (mode, label) in zip(axes, MODES.items()):
                for j, (cohort, means) in enumerate(cohort_values.items()):
                    for i, provider in enumerate(providers):
                        values = means.loc[means.feature_set.eq(mode) & means.provider.eq(provider), "effect"]
                        if len(values):
                            ax.scatter(values.mean(), i + (j - 1.5) * 0.12, s=20, color=style.COHORT_COLORS[cohort],
                                       marker=style.COHORT_MARKERS[cohort], label=cohort.replace("_", " ") if i == 0 else None)
                ax.axvline(0, color=style.GRID_COLOR)
                ax.set_title(label)
                ax.set_xlabel(unit, fontsize=7)
                ax.set_yticks(range(len(providers)), [LABELS[p] for p in providers], fontsize=7)
            axes[0].invert_yaxis()
            from matplotlib.lines import Line2D
            fig.legend([Line2D([], [], linestyle="none", marker=style.COHORT_MARKERS[c], color=style.COHORT_COLORS[c]) for c in cohort_values],
                       [c.replace("_", " ") for c in cohort_values], loc="outside lower center", ncol=2, fontsize=7)
            fig.suptitle(f"{model} · {task.split('_')[0]} · text overlap", fontsize=9)
            yield fig, f"overlap_{model}_{task}", (
                "Conditional effects on unchanged outer-test predictions, relative to the same predictor's non-text "
                "arm. Cohort membership uses text and the shared training split only; thresholds remain those "
                "selected on training data. Each marker averages dataset effects with five defined paired folds "
                "in that cohort. All-columns-unseen is a subset of unseen-joint; empty text is separate. Cohorts "
                "can have different dataset coverage and difficulty, so their differences are diagnostic, not causal "
                "effects or proof against pretraining contamination. Counts and missing metrics stay in the table."
            )
