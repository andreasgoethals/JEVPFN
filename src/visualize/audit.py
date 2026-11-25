"""Study-audit figures: completeness, compute, spending, failures and run time (no outcomes).

Inputs are the tables of `src.evaluation.study_audit`. Every function draws only what is
measured; callers print a waiting line instead of calling a function without data.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.visualize import layout, style
from src.visualize.study import _ordered_datasets

STATUS_COLORS = {
    "complete": style.EVIDENCE_MARKS["observed"], "observed": style.EVIDENCE_MARKS["observed"],
    "partial": style.EVIDENCE_MARKS["review"], "roster_gap": style.EVIDENCE_MARKS["review"],
    "missing": style.EVIDENCE_MARKS["failed"], "unknown": style.EVIDENCE_MARKS["unknown"],
    "waiting": style.EVIDENCE_MARKS["unknown"], "excluded": style.EVIDENCE_MARKS["not_applicable"],
}
STATUS_LABELS = {
    "observed": "complete artifact found locally", "partial": "requests recorded, no complete artifact",
    "unknown": "no local evidence", "complete": "complete (five defined folds)",
    "roster_gap": "decision models missing", "missing": "incomplete", "excluded": "not planned (by design)",
}
FAMILY_ORDER = ("no text", "decision", "decision+frozen", "frozen", "classifier", "TAR", "column refit",
                "native", "hidden", "hidden+decision", "adapted")


def _status_image(codes, palette):
    return np.array([[to_rgb(palette[c]) for c in row] for row in codes])


def _legend(fig, statuses, *, ncols=3):
    handles = [Patch(facecolor=STATUS_COLORS[s], edgecolor=style.GRID_COLOR, label=STATUS_LABELS[s])
               for s in statuses]
    fig.legend(handles=handles, loc="outside lower center", ncols=ncols, fontsize=7, frameon=False)


def feature_store_figure(status):
    """Decision-feature store completeness: datasets x decision models, split per view."""
    datasets, tasks, names = _ordered_datasets(status.dataset_id)
    models = list(dict.fromkeys(status.provider))
    lookup = {(r.dataset_id, r.provider, r.view): r.status for r in status.itertuples()}
    codes = [[lookup.get((d, m, v), "unknown") for m in models for v in ("per_column", "joint")]
             for d in datasets]
    fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, 4.4))
    ax.imshow(_status_image(codes, STATUS_COLORS), aspect="auto", interpolation="nearest")
    for j in range(1, len(models)):
        ax.axvline(2 * j - 0.5, color=style.BACKGROUND_COLOR, linewidth=2.2)
    for j in range(len(models)):
        ax.axvline(2 * j + 0.5, color=style.GRID_COLOR, linewidth=0.4)
    for i in range(1, len(datasets)):
        ax.axhline(i - 0.5, color=style.GRID_COLOR, linewidth=0.4)
    layout.dataset_axis(ax, [names.get(d, d) for d in datasets], tasks, fontsize=7, separators=False)
    for position in range(1, len(tasks)):
        if tasks[position] != tasks[position - 1]:
            ax.axhline(position - 0.5, color=style.MEDIAN_COLOR, linewidth=0.8)
    ax.set_xticks([2 * j + 0.5 for j in range(len(models))],
                  [style.DECISION_SHORT.get(m, m) for m in models], rotation=45, ha="left", fontsize=7)
    for tick, model in zip(ax.get_xticklabels(), models):
        tick.set_color(style.DECISION_COLORS.get(model, style.TEXT_COLOR))
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    present = [s for s in ("observed", "partial", "unknown") if any(s in row for row in codes)]
    _legend(fig, present)
    counts = status.groupby("status").size()
    observed = int(counts.get("observed", 0))
    manifest_only = int(status.source.eq("compact manifest").sum())
    return fig, "feature_store_completeness", (
        f"Each cell covers one of {len(datasets)} datasets and one of {len(models)} decision models. "
        "Its left half represents per-column requests and its right half represents joint requests. "
        f"Of {len(status)} views, {observed} have a published feature block. Local stores contain "
        f"{observed - manifest_only} of these views. Publication records alone account for {manifest_only}. "
        + (f"Amber marks {int(counts['partial'])} Jev views with recorded requests and an incomplete "
           "feature block under the fixed allocation. " if "partial" in counts else "")
        + f"White marks {int(counts.get('unknown', 0))} views awaiting evidence."
    )


def completeness_figure(table, studies=None):
    """Per study and arm family: datasets x predictors, one sliver per predictor."""
    studies = [s for s in (studies or dict.fromkeys(table.study)) if s in set(table.study)]
    if not studies:
        return None
    datasets, tasks, names = _ordered_datasets(table.dataset_id)
    columns = []
    for study in studies:
        part = table[table.study.eq(study)]
        families = [f for f in FAMILY_ORDER if f in set(part.family)]
        columns += [(study, f) for f in families]
    predictors = [m for m in (*style.MAIN_PREDICTORS, *style.NATIVE_COLORS) if m in set(table.model_id)]
    lookup = {(r.study, r.family, r.model_id, r.dataset_id): r.status for r in table.itertuples()}
    codes = []
    for d in datasets:
        row = []
        for study, family in columns:
            for m in predictors:
                row.append(lookup.get((study, family, m, d), "excluded"))
            row.append(None)  # gap between cells
        codes.append(row[:-1])
    palette = {**STATUS_COLORS, None: style.BACKGROUND_COLOR}
    width = len(predictors) + 1
    fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 1.6 + 0.17 * len(datasets))))
    ax.imshow(_status_image(codes, palette), aspect="auto", interpolation="nearest")
    layout.dataset_axis(ax, [names.get(d, d) for d in datasets], tasks, fontsize=7, separators=False)
    ax.set_xticks([k * width + (len(predictors) - 1) / 2 for k in range(len(columns))],
                  [f"{s} · {f}" for s, f in columns], rotation=45, ha="left", fontsize=7)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    present = [s for s in ("complete", "roster_gap", "missing", "excluded") if any(s in r for r in codes)]
    _legend(fig, present, ncols=4)
    order = ", ".join(style.PREDICTOR_LABELS.get(m, m) for m in predictors)
    return fig, "arm_completeness", (
        "Completeness of every planned benchmark arm in the verified result exports, per study and "
        "arm family (columns) and dataset (rows). Each cell holds one sliver per predictor, left to "
        f"right: {order}. Complete means every planned arm of the family has five defined outer folds "
        "of the headline metric; amber marks missing whole decision models (roster gap); white "
        "marks combinations not planned by design. Studies without an export are not drawn."
    )


def compute_figure(by_study=None, spend=None, budget=None):
    """GPU-hours and CPU core-hours per study (successful vs all attempts) and Jev spending."""
    panels = []
    if by_study is not None and not by_study.empty:
        panels += [("gpu", "GPU-hours"), ("cpu", "CPU core-hours")]
    if spend is not None and not spend.empty:
        panels.append(("jev", "Jev"))
    if not panels:
        return None
    ratios = [1.0 if key != "jev" else 1.15 for key, _ in panels]
    fig, axes = plt.subplots(1, len(panels), figsize=(style.WIDTH_FULL, 3.6),
                             gridspec_kw={"width_ratios": ratios}, squeeze=False)
    axes = axes[0]
    for ax, (key, label) in zip(axes, panels):
        if key in ("gpu", "cpu"):
            unit = "gpu_hours" if key == "gpu" else "cpu_core_hours"
            data = by_study.sort_values(["gpu_hours", "cpu_core_hours", "study"])
            y = np.arange(len(data))
            ax.barh(y, data[unit], height=0.62, color=style.FAINT_COLOR, edgecolor=style.MEDIAN_COLOR,
                    linewidth=0.6, label="all attempts")
            ax.barh(y, data["successful_" + unit], height=0.62, color=style.MEDIAN_COLOR,
                    label="successful jobs")
            ax.set_yticks(y, data.study, fontsize=7)
            ax.set_xlabel(f"{label} (elapsed × allocated)", fontsize=7)
            layout.light_grid(ax)
            ax.set_xlim(0, max(float(data[unit].max()) * 1.1, 1e-9))
        else:
            datasets, tasks, names = _ordered_datasets(spend.dataset_id)
            data = spend.set_index("dataset_id").reindex(datasets)
            y = np.arange(len(datasets))
            ax.barh(y, data.spent_usd, height=0.62, color=style.DECISION_COLORS["jev"], label="recorded charges")
            ax.barh(y, data.reserved_usd, left=data.spent_usd, height=0.62, color=style.EVIDENCE_MARKS["review"],
                    label="open reservations")
            if data.allocation_usd.notna().any():
                ax.vlines(data.allocation_usd, y - 0.4, y + 0.4, color=style.WINDOW_COLOR, linewidth=1)
            layout.dataset_axis(ax, [names.get(d, d) for d in datasets], tasks, fontsize=7)
            layout.light_grid(ax)
            ax.set_xlabel("Jev charges per dataset (USD)", fontsize=7)
            if budget:
                ax.set_title(f"Total ${budget['committed_usd']:.2f} of the ${budget['ceiling_usd']:.2f} ceiling"
                             .replace("$", r"\$"), fontsize=7.5)
    handles, seen = [], set()
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in seen:
                seen.add(label)
                handles.append(handle)
    if panels[-1][0] == "jev":
        handles.append(Line2D([], [], color=style.WINDOW_COLOR, linewidth=1, label="dataset allocation"))
    fig.legend(handles=handles, loc="outside lower center", ncols=min(len(handles), 5), fontsize=7,
               frameon=False)
    parts = []
    if panels[0][0] == "gpu":
        parts.append("GPU-hours (elapsed time × allocated GPUs) and CPU core-hours per study from the "
                     "Slurm compute ledger. Both panels use the same study order. Filled bars show hours used by successful jobs. Outlines include all finished "
                     "attempts including failed, cancelled and timed-out jobs")
    if panels[-1][0] == "jev":
        parts.append("recorded Jev charges and open reservations per dataset from the stored job ledgers "
                     "(one record per request), with the fixed per-dataset allocation as a vertical tick")
    caption = ". ".join(parts)
    return fig, "compute_and_spending", caption[0].upper() + caption[1:] + "."


def failure_register_figure(register):
    """Terminal job attempts per stage, stacked by final state."""
    if register is None or register.empty:
        return None
    states = ["COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL"]
    states += sorted(set(register.state) - set(states))
    colors = {"COMPLETED": style.EVIDENCE_MARKS["observed"], "FAILED": style.EVIDENCE_MARKS["failed"],
              "TIMEOUT": style.EVIDENCE_MARKS["review"], "CANCELLED": style.MISSING_COLOR,
              "OUT_OF_MEMORY": "#7B2D26", "NODE_FAIL": style.P95_COLOR}
    wide = register.pivot_table(index="stage", columns="state", values="jobs", aggfunc="sum", fill_value=0)
    wide = wide.reindex(columns=[s for s in states if s in wide.columns])
    wide = wide.loc[wide.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 1.2 + 0.24 * len(wide))))
    left = np.zeros(len(wide))
    y = np.arange(len(wide))
    for state in wide.columns:
        ax.barh(y, wide[state], left=left, height=0.62, color=colors.get(state, style.P95_COLOR),
                label=state.lower().replace("_", " "))
        left += wide[state].to_numpy()
    failed = wide.drop(columns=["COMPLETED"], errors="ignore").sum(axis=1)
    for i, (total, bad) in enumerate(zip(left, failed)):
        ax.annotate(f"{int(bad):,}/{int(total):,}", (total, i), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=6.5, color=style.MEDIAN_COLOR)
    ax.set_yticks(y, [s.replace("_", " ") for s in wide.index], fontsize=7)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlim(0, left.max() * 4)
    ax.set_xlabel("Finished job attempts (linear to 10, then log)", fontsize=7)
    layout.light_grid(ax)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncols=len(wide.columns),
               fontsize=7, frameon=False)
    return fig, "failure_register", (
        "Terminal Slurm jobs per stage, stacked by final state, from the compute ledger. Each label gives the number of "
        "unsuccessful attempts divided by all finished attempts. The category "
        "“cancelled” includes deliberate stops at handovers. Linear up to 10 jobs, logarithmic above."
    )


def runtime_figure(fits):
    """Fit seconds against input columns, one panel per predictor, colour by arm family."""
    if fits is None or fits.empty:
        return None
    predictors = [m for m in style.MAIN_PREDICTORS if m in set(fits.model_id)]
    if not predictors:
        return None
    family_colors = {"no text": style.NON_TEXT_COLOR, "decision": style.MODE_COLORS["both"],
                     "decision+frozen": style.MODE_COLORS["per_column"],
                     "frozen": style.EMBEDDING_COLORS["e5_small_v2"], "classifier": style.PROBE_COLOR,
                     "TAR": style.TAR_COLOR, "column refit": style.MODE_COLORS["joint"],
                     "hidden": style.STRANDS_COLORS["hidden"], "hidden+decision": style.STRANDS_COLORS["head"],
                     "adapted": style.STRANDS_COLORS["lora_head"]}
    cols = 4
    rows = int(np.ceil(len(predictors) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 1.75 * rows + 0.8)),
                             sharex=True, sharey=True, squeeze=False)
    for ax, model in zip(axes.flat, predictors):
        part = fits[fits.model_id.eq(model)]
        for family in FAMILY_ORDER:
            sub = part[part.family.eq(family)]
            if not sub.empty:
                ax.scatter(sub["columns"], sub.fit_seconds.clip(lower=1e-2), s=5, alpha=0.6, linewidths=0,
                           color=family_colors.get(family, style.P95_COLOR))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(style.PREDICTOR_LABELS[model], fontsize=7.5, color=style.PREDICTOR_INK)
        layout.light_grid(ax, axis="both")
    for ax in axes.flat[len(predictors):]:
        ax.set_axis_off()
    for ax in axes[-1]:
        ax.set_xlabel("input columns (log)", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("fit seconds (log)", fontsize=7)
    shown = [f for f in FAMILY_ORDER if f in set(fits.family)]
    fig.legend(handles=[Line2D([], [], linestyle="none", marker="o", markersize=4,
                               color=family_colors.get(f, style.P95_COLOR), label=f) for f in shown],
               loc="outside lower center", ncols=min(len(shown), 6), fontsize=7, frameon=False)
    return fig, "fit_runtime", (
        "Mean outer-fold fit wall time against the number of predictor input columns for every "
        f"complete exported arm ({len(fits)} arms), one panel per predictor; colour is the arm family. "
        "Times below 0.01 s are drawn at 0.01 s. Extraction and PCA preparation are recorded "
        "separately and are not part of the fit time."
    )
