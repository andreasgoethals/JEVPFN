"""Paper-width, saved-evidence-only views answering the representation questions."""

import matplotlib.pyplot as plt
import numpy as np

from src.visualize import style
from src.visualize.decision_comparison import LABELS
from src.visualize.layout import TASK_LABELS
from src.visualize.paired import METRIC_VIEWS, MODES, _dataset_effects

RECIPES = [("joint", n) for n in (8, 16, 32, 64, 128, 256)] + [("per_column", 32)]


def direct_comparison_figures(contrasts):
    """Decision features against all 21 embedding recipes, one page per task.

    Each cell pools every dataset × encoder × PCA recipe with equal weight, so no embedding
    recipe is selected; per-encoder and per-recipe detail stays in the contrast tables.
    """
    if contrasts.empty:
        return
    from src.visualize.paired import MODE_KEYS

    models = list(style.MAIN_PREDICTORS)
    short = {m: style.PREDICTOR_LABELS[m].split(" ")[0].split("/")[0] for m in models}
    cmap = plt.get_cmap(style.EFFECT_CMAP).with_extremes(bad=style.FAINT_COLOR)
    for task, (metric, direction, unit) in METRIC_VIEWS.items():
        data = contrasts[contrasts.task_type.eq(task) & contrasts.metric.eq(metric)]
        keys = ["provider", "feature_set", "model_id", "encoder", "pca_scope", "pca_components",
                "dataset_id", "split_id", "arm_id", "embedding_arm_id"]
        means = _dataset_effects(data, keys, task, direction) if not data.empty else data
        if means.empty:
            continue
        providers = [p for p in LABELS if p in set(means.provider)]
        cells = means.groupby(["provider", "feature_set", "model_id"]).agg(
            effect=("effect", "mean"), wins=("effect", lambda v: float((v > 0).mean())),
            pairs=("effect", "size"), datasets=("dataset_id", "nunique")).reset_index()
        limit = max(float(np.nanpercentile(np.abs(cells.effect), 95)), 1e-4)
        fig, axes = plt.subplots(1, len(MODES), figsize=(style.WIDTH_FULL, min(
            style.MAX_HEIGHT, 1.1 + 0.2 * len(providers))), sharey=True)
        for ax, (mode, label) in zip(axes, MODES.items()):
            values = np.full((len(providers), len(models)), np.nan)
            shares = np.full_like(values, np.nan)
            for record in cells[cells.feature_set.eq(mode)].itertuples():
                if record.model_id in models:
                    i, j = providers.index(record.provider), models.index(record.model_id)
                    values[i, j], shares[i, j] = record.effect, record.wins
            ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
            for (i, j), share in np.ndenumerate(shares):
                if np.isfinite(share):
                    ax.text(j, i, f"{100 * share:.0f}%", ha="center", va="center", fontsize=5.8,
                            color=style.BACKGROUND_COLOR if abs(values[i, j]) > 0.6 * limit
                            else style.TEXT_COLOR)
            ax.set_title(label, fontsize=8, color=style.MODE_COLORS[MODE_KEYS[mode]])
            ax.set_xticks(range(len(models)), [short[m] for m in models], rotation=90, fontsize=6.5)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[0].set_yticks(range(len(providers)), [LABELS[p] for p in providers], fontsize=6.5)
        for tick, provider in zip(axes[0].get_yticklabels(), providers):
            tick.set_color(style.DECISION_COLORS[provider])
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-limit, limit))
        fig.colorbar(mappable, ax=list(axes), shrink=0.8, pad=0.02,
                     label=unit.replace("augmented", "decision").replace("reference", "embedding"))
        yield fig, f"decision_versus_embedding_{task}", (
            f"{TASK_LABELS[task]} datasets: decision features against frozen embeddings in the same "
            "predictor, dataset and outer folds. Colour is the mean paired difference in the "
            "prespecified endpoint over every dataset and all 21 embedding recipes (3 encoders × 7 "
            "PCA settings), weighted equally; the percentage is the share of these dataset × recipe "
            "pairs in which the decision features score higher. Positive favours decision features. "
            "Pairs need five defined folds; grey cells lack complete evidence. Descriptive, without "
            "selecting a recipe; the scale is clipped at the 95th percentile of |difference|."
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
            fig.suptitle(f"{style.PREDICTOR_LABELS[model]} · {TASK_LABELS[task]} · seen and unseen text",
                         fontsize=9)
            yield fig, f"overlap_{model}_{task}", (
                "Conditional effects on unchanged outer-test predictions, relative to the same predictor's non-text "
                "arm. Cohort membership uses text and the shared training split only; thresholds remain those "
                "selected on training data. Each marker averages dataset effects with five defined paired folds "
                "in that cohort. All-columns-unseen is a subset of unseen-joint; empty text is separate. Cohorts "
                "can have different dataset coverage and difficulty, so their differences are diagnostic, not causal "
                "effects or proof against pretraining contamination. Counts and missing metrics stay in the table."
            )
