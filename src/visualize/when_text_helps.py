"""Exp 8 (RQ2-E) figures: decision-model axis, ceiling check, cohorts, complementarity, data map.

Every function yields ``(figure, name, caption)`` and yields nothing when its input is empty,
so the notebook shows "pending" instead of an empty canvas. Descriptive views only; no
representation or decision model is selected.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

from src.evaluation.aggregate import MEDIAN
from src.evaluation.row_benefit import POST_TREATMENT_AXES, POST_TREATMENT_LABEL
from src.visualize import layout, style

_MODE = {"non_text_plus_per_column": "per_column", "non_text_plus_joint": "joint",
         "non_text_plus_both": "both"}
_MODE_LABELS = {"per_column": "per-column", "joint": "joint", "both": "both"}


def _predictors(values):
    known = [m for m in style.MAIN_PREDICTORS if m in set(values)]
    return known + sorted(set(values) - set(known))


def _symmetric_norm(values):
    finite = np.abs(np.asarray(values, float))
    finite = finite[np.isfinite(finite)]
    limit = float(np.percentile(finite, 95)) if len(finite) else 1.0
    limit = limit if limit > 0 else 1.0
    return TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit)


def _property_label(row):
    parts = []
    if pd.notna(row.get("parameters_billion")):
        parts.append(f"{row['parameters_billion']:.3g}B")
    for key in ("architecture_family", "readout_type"):
        value = row.get(key)
        parts.append(str(value).replace("_", " ") if pd.notna(value) and value else "n/v")
    return " · ".join(parts)


# ---------------------------------------------------------------- decision-model axis


def model_gain_figures(model_gains, covariates=None):
    """Decision models ranked by mean normalised gain, per predictor, properties annotated."""
    if model_gains is None or model_gains.empty:
        return
    covariates = covariates if covariates is not None else pd.DataFrame(columns=["provider"])
    lookup = covariates.set_index("provider").to_dict("index") if not covariates.empty else {}
    for feature_set in [f for f in _MODE if f in set(model_gains.feature_set)]:
        cell = model_gains[model_gains.feature_set.eq(feature_set)]
        wide = cell.pivot_table(index="provider", columns="model_id", values="mean_gain")
        wide = wide[_predictors(wide.columns)]
        order = wide.mean(axis=1).sort_values(ascending=False).index
        wide = wide.loc[order]
        height = min(style.MAX_HEIGHT, 1.0 + 0.19 * len(wide))
        fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, height))
        cmap = plt.get_cmap(style.EFFECT_CMAP).with_extremes(bad=style.FAINT_COLOR)
        image = ax.imshow(np.ma.masked_invalid(wide.to_numpy(float)), cmap=cmap, aspect="auto",
                          norm=_symmetric_norm(wide.to_numpy()))
        for (i, j), value in np.ndenumerate(wide.to_numpy(float)):
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6)
        ax.set_xticks(range(wide.shape[1]),
                      [style.PREDICTOR_LABELS.get(m, m) for m in wide.columns], fontsize=7,
                      rotation=30, ha="right")
        ax.set_yticks(range(len(wide)), [style.DECISION_SHORT.get(p, p) for p in wide.index],
                      fontsize=7)
        for tick, provider in zip(ax.get_yticklabels(), wide.index):
            tick.set_color(style.DECISION_COLORS.get(provider, style.TEXT_COLOR))
        for i, provider in enumerate(wide.index):
            ax.text(1.02, i, _property_label(lookup.get(provider, {})), fontsize=6, va="center",
                    ha="left", transform=ax.get_yaxis_transform(), clip_on=False)
        ax.tick_params(length=0)
        fig.colorbar(image, ax=ax, orientation="horizontal", shrink=0.5, aspect=40,
                     label="Mean normalised gain")
        n = f"{int(cell.n_datasets.min())}–{int(cell.n_datasets.max())}"
        mode = _MODE[feature_set]
        yield fig, f"decision_models_ranked_{mode}", (
            f"Mean normalised gain of each decision model's {_MODE_LABELS[mode]} features over the "
            f"same predictor's no-text baseline, averaged with equal weight over the {n} datasets "
            "observed for every decision model of that predictor. Rows are ordered by the mean over "
            "predictors; the right margin gives verified parameter count, architecture family and "
            "readout type (n/v: not verified in a local source). Size, training data and readout "
            "are confounded across releases; descriptive, no model is selected."
        )


def raw_gain_headroom_figures(gains):
    """Raw (unnormalised) gain against the baseline's headroom, per predictor."""
    if gains is None or gains.empty:
        return
    rows = gains[gains.aggregation.eq(MEDIAN)].copy()
    if rows.empty:
        return
    if "raw_gain" not in rows:
        rows["raw_gain"] = rows.augmented - rows.baseline
    rows["headroom"] = 1.0 - rows.baseline
    models = _predictors(rows.model_id)
    columns = min(3, len(models))
    lines = int(np.ceil(len(models) / columns))
    fig, axes = plt.subplots(lines, columns, figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 0.9 + 1.7 * lines)),
                             sharex=True, sharey=True, squeeze=False)
    limit = max(float(rows.headroom.max()), 1e-3)
    for ax, model in zip(axes.flat, models):
        part = rows[rows.model_id.eq(model)]
        ax.axhline(0, color=style.MEDIAN_COLOR, linewidth=0.7)
        ax.plot([0, limit], [0, limit], color=style.GRID_COLOR, linewidth=0.8, linestyle="--")
        for feature_set, group in part.groupby("feature_set"):
            mode = _MODE.get(feature_set, feature_set)
            ax.scatter(group.headroom, group.raw_gain, s=10, alpha=0.8,
                       color=style.MODE_COLORS.get(mode, style.QUANTITY_COLOR))
        ax.set_title(style.PREDICTOR_LABELS.get(model, model), fontsize=8)
        layout.light_grid(ax, axis="both")
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(models):]:
        ax.set_axis_off()
    fig.supxlabel("Baseline headroom (1 − no-text headline metric)", fontsize=7)
    fig.supylabel("Raw gain (augmented − baseline)", fontsize=7)
    modes = [m for m in _MODE.values() if m in {_MODE.get(f) for f in rows.feature_set}]
    fig.legend(handles=[Line2D([], [], color=style.MODE_COLORS[m], marker="o", linestyle="",
                               markersize=4, label=f"Decision · {_MODE_LABELS[m]}") for m in modes],
               loc="outside upper center", ncols=len(modes), fontsize=7, frameon=False)
    yield fig, "raw_gain_vs_headroom", (
        "Unnormalised five-fold mean gain in the headline metric against the baseline's headroom "
        "(one point per dataset; median over decision models per mode). The dashed diagonal is "
        "the ceiling, where the augmented model would reach a perfect score. Normalised gain "
        "divides by headroom, so its association with headroom is partly mechanical; this raw "
        "view shows whether larger headroom also brings larger absolute gains. Descriptive."
    )


# ------------------------------------------------------------------------------ cohorts


def _cohort_order(cohorts):
    preferred = ["all", "present", "absent", "seen", "unseen", "0", "1", "2", "3",
                 "lt_0.5", "0.5_to_0.7", "0.7_to_0.9", "ge_0.9", "0.5_to_0.75", "0.75_to_1",
                 "unanimous", "shift_0", "shift_1", "shift_2", "shift_3"]
    known = [c for c in preferred if c in cohorts]
    rest = sorted(set(cohorts) - set(known) - {"missing"})
    return known + rest + (["missing"] if "missing" in cohorts else [])


def cohort_figures(balanced):
    """Equal-dataset mean row benefit per cohort; post-treatment axes are marked descriptive."""
    if balanced is None or balanced.empty:
        return
    table = balanced[balanced.axis.ne("all")].copy()
    table["loss"] = np.where(table.task.eq("regression"), "regression", "classification")
    for loss, part in table.groupby("loss", sort=True):
        axes_names = sorted(set(part.axis), key=lambda a: (a in POST_TREATMENT_AXES,
                                                           a.startswith("novelty_by"), a))
        columns = min(4, len(axes_names))
        lines = int(np.ceil(len(axes_names) / columns))
        fig, axes = plt.subplots(lines, columns, figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 0.8 + 1.35 * lines)),
                                 sharey=True, squeeze=False)
        for ax, axis in zip(axes.flat, axes_names):
            data = part[part.axis.eq(axis)]
            cohorts = _cohort_order(set(data.cohort))
            for phase, group in data.groupby("phase", sort=True):
                means = group.groupby("cohort").mean_benefit.mean().reindex(cohorts)
                ax.plot(range(len(cohorts)), means.to_numpy(float), marker="o", markersize=3,
                        linewidth=0.8, color=style.FAMILY_COLORS.get(phase, style.QUANTITY_COLOR))
            ax.axhline(0, color=style.MEDIAN_COLOR, linewidth=0.6)
            ax.set_xticks(range(len(cohorts)), cohorts, rotation=40, ha="right", fontsize=6)
            title = axis.replace("novelty_by_field_type:", "novelty · ").replace("_", " ")
            if axis in POST_TREATMENT_AXES:
                ax.set_facecolor(style.FAINT_COLOR)
                title = f"{title}\n{POST_TREATMENT_LABEL}"
                ax.title.set_color(style.POST_TREATMENT_COLOR)
            ax.set_title(title, fontsize=7)
            ax.tick_params(axis="y", labelsize=7)
            layout.light_grid(ax, axis="y")
        for ax in axes.flat[len(axes_names):]:
            ax.set_axis_off()
        phases = sorted(set(part.phase))
        names = {"experiment_1": "Decision features", "experiment_2": "Frozen embeddings",
                 "experiment_3": "Supervised TAR"}
        fig.legend(handles=[Line2D([], [], color=style.FAMILY_COLORS.get(p, style.QUANTITY_COLOR),
                                   marker="o", markersize=3, label=names.get(p, p)) for p in phases],
                   loc="outside upper center", ncols=len(phases), fontsize=7, frameon=False)
        unit = "log loss" if loss == "classification" else "squared error / training variance"
        yield fig, f"row_cohorts_{loss}", (
            f"Mean held-out benefit ({unit} of the structured baseline minus that of the augmented "
            "model; positive means the representation helped) per fixed cohort, averaged over "
            "predictors and representations of each family, with datasets weighted equally. "
            "Shaded panels are post-treatment: prediction-shift cohorts are defined by the augmented "
            "model's own predictions and are descriptive only. Novelty-by-field-type panels "
            "split seen/unseen text within each label-free field type. Observational, not causal."
        )


# ---------------------------------------------------------------------- complementarity


def complementarity_figures(summary):
    """Equal-dataset mean row-benefit rank correlation and top-10% overlap above chance."""
    if summary is None or summary.empty:
        return
    rows = summary[summary.aggregation.eq("mode_by_family")]
    if rows.empty:
        return
    rows = rows.assign(column=rows.decision.str.split("/").str[1].map(_MODE_LABELS).fillna(rows.decision)
                       + " vs " + rows.other.map({"embedding": "frozen", "tar": "TAR"}).fillna(rows.other),
                       excess=rows.mean_jaccard_top - rows.mean_jaccard_chance)
    models = _predictors(rows.model_id)
    columns = sorted(set(rows.column))
    fig, axes = plt.subplots(1, 2, figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 1.3 + 0.3 * len(models))),
                             sharey=True)
    cmap = plt.get_cmap(style.AGREEMENT_DIVERGING).with_extremes(bad=style.FAINT_COLOR)
    for ax, value, label, norm in (
        (axes[0], "mean_spearman_rho", "Spearman rho of row benefits", TwoSlopeNorm(0, -1, 1)),
        (axes[1], "excess", "Top-10% Jaccard minus chance", None),
    ):
        wide = rows.pivot_table(index="model_id", columns="column", values=value).reindex(
            index=models, columns=columns)
        matrix = wide.to_numpy(float)
        image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, aspect="auto",
                          norm=norm or _symmetric_norm(matrix))
        for (i, j), cell in np.ndenumerate(matrix):
            if np.isfinite(cell):
                ax.text(j, i, f"{cell:.2f}", ha="center", va="center", fontsize=6)
        ax.set_xticks(range(len(columns)), columns, rotation=35, ha="right", fontsize=6)
        ax.set_yticks(range(len(models)), [style.PREDICTOR_LABELS.get(m, m) for m in models],
                      fontsize=7)
        ax.tick_params(length=0)
        ax.set_title(label, fontsize=7)
        fig.colorbar(image, ax=ax, shrink=0.8)
    n = f"{int(rows.n_datasets.min())}–{int(rows.n_datasets.max())}"
    yield fig, "row_benefit_complementarity", (
        "Do decision features help the same held-out rows as frozen embeddings or TAR? Left: "
        "Spearman correlation between the two arms' per-row benefits on identical rows and folds "
        "of the same predictor; right: overlap (Jaccard) of the two arms' top-10% benefit rows "
        "minus the overlap expected for independent sets. Values are averaged within each dataset "
        f"over decision models and recipes, then over {n} datasets with equal weight. High values "
        "suggest substitutes, low values potential complements. Descriptive."
    )


# ------------------------------------------------------------------------------ data map


def data_map_figures(grids, *, feature_set="non_text_plus_joint",
                     x_axis="baseline_max_probability", model_id=None,
                     min_rows=style.DATA_MAP_MIN_ROWS):
    """Per dataset: baseline confidence × decision confidence, cells coloured by mean benefit.

    Classification only. By default cells pool all predictors and decision models of one
    decision mode (row × model pairs); pass ``model_id`` for a single predictor.
    """
    if grids is None or grids.empty:
        return
    part = grids[grids.feature_set.eq(feature_set) & grids.x_axis.eq(x_axis)]
    if model_id is not None:
        part = part[part.model_id.eq(model_id)]
    if part.empty:
        return
    pooled = {}
    for row in part.itertuples():
        counts, sums = np.asarray(row.counts, float), np.asarray(row.benefit_sum, float)
        if row.dataset_id in pooled:
            pooled[row.dataset_id][0] += counts
            pooled[row.dataset_id][1] += sums
        else:
            pooled[row.dataset_id] = [counts, sums, row.task_type, row.bins]
    order = layout.ordered(pd.DataFrame([{"dataset_id": d, "task_type": v[2]} for d, v in pooled.items()]))
    datasets = list(order.dataset_id)
    means = {d: np.where(v[0] >= min_rows, v[1] / np.maximum(v[0], 1), np.nan) for d, v in pooled.items()}
    norm = _symmetric_norm(np.concatenate([m.ravel() for m in means.values()]))
    cmap = plt.get_cmap(style.EFFECT_CMAP).with_extremes(bad=style.FAINT_COLOR)
    lines, columns = style.DATA_MAP_PANELS
    per_page = lines * columns
    scope = "all predictors" if model_id is None else style.PREDICTOR_LABELS.get(model_id, model_id)
    mode = _MODE.get(feature_set, feature_set)
    xlabel = {"baseline_max_probability": "Baseline maximum class probability",
              "baseline_certainty": "Baseline certainty (1 − normalised entropy)"}[x_axis]
    pages = [datasets[i:i + per_page] for i in range(0, len(datasets), per_page)]
    for page, names in enumerate(pages, start=1):
        rows_needed = int(np.ceil(len(names) / columns))
        fig, axes = plt.subplots(rows_needed, columns, squeeze=False, sharex=True, sharey=True,
                                 figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 0.9 + 1.45 * rows_needed)))
        image = None
        for ax, name in zip(axes.flat, names):
            image = ax.imshow(np.ma.masked_invalid(means[name].T), origin="lower", cmap=cmap,
                              norm=norm, extent=(0, 1, 0, 1), aspect="auto", interpolation="nearest")
            ax.set_title(name, fontsize=7, color=style.TASK_COLORS.get(pooled[name][2], style.TEXT_COLOR))
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
            ax.tick_params(labelsize=6)
            ax.text(0.02, 0.98, f"{int(pooled[name][0].sum()):,}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=5, color=style.MEDIAN_COLOR)
        for ax in axes.flat[len(names):]:
            ax.set_axis_off()
        fig.supxlabel(xlabel, fontsize=7)
        fig.supylabel("Decision confidence (mean max probability)", fontsize=7)
        if image is not None:
            fig.colorbar(image, ax=axes, shrink=0.8, label="Mean row benefit")
        suffix = f"_{page}" if len(pages) > 1 else ""
        yield fig, f"data_map_{mode}_{x_axis}{suffix}", (
            f"Outer-test rows of each classification dataset placed by the structured baseline's "
            f"confidence and by the {_MODE_LABELS.get(mode, mode)} decision features' own confidence, "
            f"pooling {scope} and every decision model (row × model pairs; corner count). Colour is "
            "the mean row benefit, baseline log loss minus augmented log loss, centred at zero "
            f"(positive helped); cells with fewer than {min_rows} pairs are blank. Both axes are "
            f"label-free. Page {page} of {len(pages)}. Descriptive."
        )
