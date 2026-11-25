"""B3 figures (B/exp08_when_text_helps): when and for whom do decision features help?

Reads the B3 export schema (`benchmark_export` B3 tables) through `group_b_views`. Every
function yields ``(figure, name, caption)`` and yields nothing for an empty table. Descriptive
and observational: associations, not causal effects; no decision model is selected.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src.visualize import layout, style
from src.visualize.group_b import (
    MARKERS,
    PREDICTORS,
    TASK_UNITS,
    _dataset_rows,
    _family_layout,
    _family_title,
    _zero,
    predictor_handles,
)

PROPERTY_LABELS = {
    "n_rows": "rows", "log10_n_rows": "log10 rows", "n_text_fields": "text columns",
    "n_non_text_fields": "non-text columns", "n_classes": "classes",
    "median_text_chars": "median text length", "log10_median_text_chars": "log10 median text length",
    "missing_text_share": "empty-text share", "repeated_text_share": "repeated-text share",
    "unseen_text_share": "unique row-text share", "free_text_share": "free-text column share",
    "decision_redundancy_r2": "decision redundancy (all rows)",
    "pilot_redundancy_r2": "decision redundancy (200 rows/model)",
}
DIAGNOSTIC_LABELS = {"standalone_classification_auc": "Classification\nAUC",
                     "standalone_regression_spearman": "Regression\nSpearman ρ",
                     "calibration_ece": "Calibration error\n(ECE)",
                     "consensus_distance": "Distance to\nconsensus",
                     "stability_excess_tv": "Excess change\nover replay"}
INPUT_SETS = {"pre_extraction": "Pre-extraction\n(text length)",
              "pilot": "200 rows per model\n(+ pilot redundancy)",
              "post_extraction": "Post-extraction\n(+ redundancy)"}


def headroom_and_associations(headroom, associations, *, name, what,
                              representation="decision/both/median"):
    """(a) Median-model gain against the baseline headroom per dataset, per task family;
    (b) Spearman rho of each label-free property with that gain, per predictor (BH asterisks)."""
    if headroom is None or headroom.empty:
        return
    assoc = (associations[associations.representation.eq(representation) & associations.task_family.eq("all")
                          & associations.outcome.eq("headroom_share")]
             if associations is not None and not associations.empty and {"outcome", "task_family"} <= set(associations)
             else None)
    has_b = assoc is not None and not assoc.empty and assoc.spearman_rho.notna().any()
    fig = plt.figure(figsize=(style.WIDTH_FULL, style.MAX_HEIGHT if has_b else 2.5), layout="constrained")
    grid = fig.add_gridspec(2 if has_b else 1, 2, height_ratios=[1, 1.25] if has_b else [1])
    for j, family in enumerate(style.TASK_FAMILIES):
        ax = fig.add_subplot(grid[0, j])
        part = headroom[headroom.task_family.eq(family)]
        if part.empty:
            ax.axis("off")
            continue
        colors = [style.TASK_COLORS[t] for t in part.task_type]
        ax.vlines(part.headroom, part.low, part.high, colors=colors, lw=0.5, alpha=0.6)
        ax.scatter(part.headroom, part.value, s=14, c=colors, linewidths=0, zorder=3)
        top = float(np.nanmax([part.headroom.max(), part.high.max(), 0.01])) * 1.08
        ax.plot([0, top], [0, top], color=style.MEDIAN_COLOR, lw=0.6, ls="--")
        ax.set_xlim(0, top)
        _zero(ax, "y")
        layout.light_grid(ax, axis="both")
        ax.tick_params(labelsize=6.5)
        ax.set_xlabel("Baseline headroom, 1 − no-text score", fontsize=7)
        ax.set_ylabel(f"Gain ({TASK_UNITS[family]})", fontsize=7)
        _family_title(ax, family, len(part))
    if has_b:
        ax = fig.add_subplot(grid[1, :])
        table = assoc.pivot_table(index="property", columns="model_id", values="spearman_rho")
        qs = assoc.pivot_table(index="property", columns="model_id", values="q_value_bh")
        properties = [p for p in PROPERTY_LABELS if p in table.index] + sorted(set(table.index) - set(PROPERTY_LABELS))
        models = [m for m in PREDICTORS if m in table.columns]
        table, qs = table.reindex(index=properties, columns=models), qs.reindex(index=properties, columns=models)
        image = ax.imshow(table.to_numpy(float).T, cmap=style.EFFECT_CMAP, vmin=-1, vmax=1, aspect="auto")
        for (i, j), value in np.ndenumerate(qs.to_numpy(float).T):
            if np.isfinite(value) and value <= 0.10:
                ax.text(j, i, "*", ha="center", va="center", fontsize=9)
        ax.set_yticks(range(len(models)), [style.PREDICTOR_LABELS.get(m, m) for m in models], fontsize=6.5)
        ax.set_xticks(range(len(properties)), [PROPERTY_LABELS.get(p, p.replace("_", " ")) for p in properties],
                      rotation=35, ha="right", fontsize=6.5)
        ax.tick_params(length=0)
        bar = fig.colorbar(image, ax=ax, shrink=0.8, pad=0.01)
        bar.set_label("Spearman ρ over datasets", fontsize=7)
        bar.ax.tick_params(labelsize=6.5)
    fig.legend(handles=layout.task_handles(marker=True), loc="outside upper right", ncol=3,
               frameon=False, fontsize=6.5)
    yield fig, name, (
        f"{what}. Top: per dataset, the gain of the median decision model's both-view features over "
        "the no-text baseline against the baseline headroom (one minus the no-text score), each the "
        "median over the seven predictors (line: predictors' range of the gain); the dashed diagonal "
        "is the ceiling where the gain closes all headroom. Classification and regression on separate "
        "axes."
        + (" The lower panel shows each property's Spearman correlation with the share of headroom gained "
           "across datasets for each predictor. Asterisks mark Benjamini–Hochberg q ≤ 0.10 across the "
           "dataset-property grid, including the raw-gain correlations within each task family "
           "(5,000 permutations)." if has_b else ""))


def diagnostics_vs_usefulness(table, relations, *, name, what):
    """Per decision model: four Group A diagnostics against equal-dataset usefulness."""
    if table is None or table.empty:
        return
    columns = [c for c in DIAGNOSTIC_LABELS if c in table and table[c].notna().any()]
    if not columns:
        return
    fig, axes = plt.subplots(1, len(columns), figsize=(style.WIDTH_FULL, 2.3), layout="constrained",
                             sharey=True, squeeze=False)
    stats = relations.set_index("property") if relations is not None and not relations.empty else None
    for ax, column in zip(axes[0], columns):
        ax.vlines(table[column], table.low, table.high, color=style.DECISION_POOL_COLOR, lw=0.6)
        ax.scatter(table[column], table.usefulness, s=12, color=style.MEDIAN_COLOR, zorder=3, linewidths=0)
        _zero(ax, "y")
        layout.light_grid(ax, axis="both")
        ax.tick_params(labelsize=6.5)
        ax.set_xlabel(DIAGNOSTIC_LABELS[column], fontsize=7)
        if stats is not None and column in stats.index:
            row = stats.loc[column]
            ax.set_title(f"ρ = {row.spearman_rho:.2f}, q = {row.q_value_bh:.2f}", loc="left", fontsize=7)
    axes[0, 0].set_ylabel("Mean headroom share (all datasets)", fontsize=7)
    yield fig, name, (
        f"{what}: each point is one decision model ({len(table)} models): its Group A diagnostic "
        "(joint requests, distinct inputs, equal dataset weight) against the equal-dataset mean share "
        "of the baseline headroom gained by its both-view features over all datasets, averaged over "
        "the seven predictors (line: predictors' range). Titles give Spearman ρ across models with a "
        "Benjamini-Hochberg q over the diagnostics shown (permutation p-values). Exploratory: size, "
        "training data and readout are confounded across models.")


def column_refits(table, *, name, what):
    """Per dataset: one dot per text column, the median drop when that column's block is removed."""
    if table is None or table.empty:
        return
    fig, axes, families, rows, names = _family_layout(table, row_height=0.26, extra=1.3)
    types = [t for t in style.FIELD_TYPE_COLORS if "field_type" in table and t in set(table.field_type)]
    for ax, family in zip(axes, families):
        order = [d for d, _ in rows[family]]
        y = {d: i for i, d in enumerate(order)}
        part = table[table.dataset_id.isin(order)]
        colors = ([style.FIELD_TYPE_COLORS.get(t, style.MEDIAN_COLOR) for t in part.field_type]
                  if "field_type" in part else style.MEDIAN_COLOR)
        jitter = np.random.default_rng(2).uniform(-0.15, 0.15, len(part))
        ax.scatter(part.value, part.dataset_id.map(y) + jitter, s=14, c=colors, linewidths=0, zorder=3)
        _zero(ax)
        _dataset_rows(ax, rows[family], names)
        layout.light_grid(ax)
        _family_title(ax, family, len(order))
        ax.set_xlabel(f"Loss when the column's block is removed ({TASK_UNITS[family]})", fontsize=7)
        ax.tick_params(axis="x", labelsize=6.5)
    if types:
        fig.legend(handles=[Line2D([], [], ls="none", marker="o", ms=4, color=style.FIELD_TYPE_COLORS[t],
                                   label=t.replace("_", " ")) for t in types],
                   loc="outside lower center", ncol=len(types), frameon=False, fontsize=7)
    yield fig, name, (
        f"{what}: per dataset with at least two text columns ({table.dataset_id.nunique()} datasets, "
        f"{len(table)} columns), one dot per text column: the per-column decision arm's five-fold mean "
        "headline score minus that of the same arm refitted without the column's decision block "
        "(fold-paired; positive = the column's block helps), median over the selected decision models "
        "and then over the seven predictors (small vertical jitter). Colour gives the label-free field "
        "type. No permutation importance, shuffling or SHAP.")


COHORT_ORDER = {"availability": ("present", "absent"),
                "decision_confidence": ("lt_0.5", "0.5_to_0.7", "0.7_to_0.9", "ge_0.9"),
                "decision_agreement": ("lt_0.5", "0.5_to_0.75", "0.75_to_1", "unanimous")}
COHORT_TITLES = {"availability": "Text present", "length": "Text length bin",
                 "baseline_entropy": "Baseline uncertainty", "decision_confidence": "Decision confidence",
                 "decision_agreement": "Model agreement"}


def cohort_benefits(table, *, name, what):
    """Equal-dataset mean row benefit per fixed label-free cohort, one row of panels per family."""
    if table is None or table.empty:
        return
    from src.evaluation.group_b_views import COHORT_AXES

    families = [f for f in style.TASK_FAMILIES if f in set(table.task_family)]
    axes_names = [a for a in COHORT_AXES if a in set(table.axis)]
    fig, axes = plt.subplots(len(families), len(axes_names), squeeze=False, layout="constrained",
                             figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 0.9 + 1.75 * len(families))))
    for i, family in enumerate(families):
        for j, axis in enumerate(axes_names):
            ax = axes[i, j]
            part = table[table.task_family.eq(family) & table.axis.eq(axis)]
            cohorts = list(dict.fromkeys(part.sort_values("cohort").cohort))
            fixed = COHORT_ORDER.get(axis)
            if fixed:
                cohorts = [c for c in fixed if c in cohorts] + [c for c in cohorts if c not in fixed]
            cohorts = [c for c in cohorts if c != "missing"] + (["missing"] if "missing" in cohorts else [])
            for mode in ("per_column", "joint", "both"):
                rows = part[part["mode"].eq(mode)].set_index("cohort").reindex(cohorts)
                x = np.arange(len(cohorts))
                ax.fill_between(x, rows.low, rows.high, color=style.MODE_COLORS[mode], alpha=0.12, lw=0)
                ax.plot(x, rows.mean_benefit, color=style.MODE_COLORS[mode], lw=1, marker="o", ms=2.5)
            _zero(ax, "y")
            ax.set_xticks(range(len(cohorts)), [c.replace("_to_", "–").replace("_", " ") for c in cohorts],
                          rotation=45, ha="right", fontsize=6)
            ax.tick_params(axis="y", labelsize=6)
            layout.light_grid(ax, axis="y")
            if i == 0:
                ax.set_title(COHORT_TITLES.get(axis, axis), fontsize=7)
            if j == 0:
                ax.set_ylabel("Log-loss benefit" if family == "classification" else "Scaled SE benefit",
                              fontsize=7)
    fig.legend(handles=[Line2D([], [], color=style.MODE_COLORS[m], lw=1, marker="o", ms=2.5,
                               label=f"Decision, {m.replace('_', ' ')}") for m in ("per_column", "joint", "both")],
               loc="outside lower center", ncol=3, frameon=False, fontsize=7)
    yield fig, name, (
        f"{what}: equal-dataset mean held-out row benefit (no-text loss minus the decision arm's loss; "
        "log loss for classification, squared error over the training-target variance for regression; "
        "positive = the arm helped) within fixed label-free cohorts, averaged over the seven predictors "
        "(band: their range) per decision view. Rows: classification and regression; columns: text "
        "availability, training-defined text-length bins, baseline uncertainty quartile bins, decision "
        "confidence and cross-model agreement. Observational, not causal.")


def lodo(predictions, summary, *, name, what):
    """(a) LODO predicted against actual gain per input set; (b) regret against always/never."""
    if predictions is None or predictions.empty:
        return
    for table in (predictions, summary):
        if table is not None and not table.empty and (
                "outcome" not in table or not table.outcome.eq("headroom_share").fillna(False).all()):
            raise ValueError("B3 LODO figures require headroom-share predictions and regret")
    sets = [s for s in INPUT_SETS if s in set(predictions.input_set)]
    fig = plt.figure(figsize=(style.WIDTH_FULL, 4.2), layout="constrained")
    grid = fig.add_gridspec(2, len(sets), height_ratios=[1.25, 1])
    shared = None
    for j, input_set in enumerate(sets):
        ax = fig.add_subplot(grid[0, j], sharex=shared, sharey=shared)
        shared = shared or ax
        part = predictions[predictions.input_set.eq(input_set)]
        for model in [m for m in PREDICTORS if m in set(part.model_id)]:
            rows = part[part.model_id.eq(model)]
            ax.scatter(rows.predicted, rows.actual, s=9, marker=MARKERS[model], facecolors="none",
                       edgecolors=style.PREDICTOR_INK, linewidths=0.5, alpha=0.7)
        _zero(ax, "x")
        _zero(ax, "y")
        ax.tick_params(labelsize=6.5)
        ax.set_title(INPUT_SETS[input_set].split("\n")[0], loc="left", fontsize=7)
        ax.set_xlabel("Predicted headroom share", fontsize=7)
        if j == 0:
            ax.set_ylabel("Observed headroom share", fontsize=7)
        layout.light_grid(ax, axis="both")
    if summary is not None and not summary.empty:
        ax = fig.add_subplot(grid[1, :])
        labels = [INPUT_SETS[s].replace("\n", " ") for s in sets] + ["Always use", "Never use"]
        for model in [m for m in PREDICTORS if m in set(summary.model_id)]:
            rows = summary[summary.model_id.eq(model)].set_index("input_set")
            values = [rows.regret.get(s, np.nan) for s in sets]
            values += [rows.regret_always.iloc[0], rows.regret_never.iloc[0]]
            ax.scatter(values, np.arange(len(labels)) + 0.0, marker=MARKERS[model], s=14, facecolors="none",
                       edgecolors=style.PREDICTOR_INK, linewidths=0.6)
        for i, s in enumerate(sets):
            p = summary[summary.input_set.eq(s)].get("shuffle_p_rho")
            if p is not None and p.notna().any():
                ax.annotate(f"shuffle p(ρ) {p.min():.2f}–{p.max():.2f}", (1.0, i),
                            xycoords=("axes fraction", "data"), fontsize=6, ha="right", va="center",
                            color=style.MEDIAN_COLOR)
        ax.set_yticks(range(len(labels)), labels, fontsize=6.5)
        ax.set_ylim(len(labels) - 0.5, -0.5)
        ax.tick_params(axis="x", labelsize=6.5)
        ax.set_xlabel("Regret: oracle gain minus realised gain (mean over datasets; lower is better)", fontsize=7)
        layout.light_grid(ax)
    fig.legend(handles=predictor_handles([m for m in PREDICTORS if m in set(predictions.model_id)]),
               loc="outside lower center", ncol=7, frameon=False, fontsize=6.5)
    yield fig, name, (
        f"{what}. Top: leave-one-dataset-out prediction of each dataset's gain from the median decision "
        "model's both-view features (ridge, alpha 1, on standardised label-free properties fitted on the "
        "other datasets) against the actual gain, for three nested input sets; one marker shape per "
        "predictor; upper-right and lower-left quadrants are correct use/don't-use decisions. Gains and "
        "regrets are fractions of the headroom above the same predictor's no-text score. Bottom: regret of each rule "
        "(use when the prediction exceeds 0) against always and never using the features; the range of "
        "the shuffle-null p-value of the rank correlation over predictors is printed (2,000 shuffles "
        "that refit the whole rule).")
