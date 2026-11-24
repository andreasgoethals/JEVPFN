"""Group B figures (B1, B2, B4, B5), drawn at final A4 width from the per-experiment exports.

Every generator yields ``(figure, name, caption)`` and yields nothing for an empty table, so
the notebooks show a waiting state instead of an empty figure. Gains are "arm minus the same
predictor's no-text baseline" in the fixed headline metric; raw gains stay within one task
family (classification: ROC-AUC/macro AUC, N = 9; regression: R², N = 9) in separate panels,
and only the headroom share pools all 18 datasets. Colour = representation, marker shape =
predictor (one ink), grey dots = individual decision models. Tables come from
`src.evaluation.group_b_views`; B3 lives in `when_text_helps`, the synthesis in
`group_b_synthesis`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from src.visualize import layout, style

PREDICTORS = style.MAIN_PREDICTORS
# Kept for Group C figures (group_c imports them); Group B now draws predictors in one ink.
PREDICTOR_COLORS = dict(zip(PREDICTORS, ("#4E79A7", "#F28E2B", "#59A14F", "#B07AA1", "#9C755F",
                                         "#76B7B2", "#E15759")))
MARKERS = style.PREDICTOR_MARKERS
NATIVE_MARKER = "*"
FAMILY_LABELS = {
    "decision/joint/median": "Decision, joint (median model)",
    "decision/per_column/median": "Decision, per column (median model)",
    "decision/both/median": "Decision, both (median model)",
    "decision+frozen/both/median": "Decision both + E5-small PCA32 (median model)",
    "baseline": "No text (same predictor)",
}
ENCODER_NAMES = {"e5_small_v2": "E5-small", "e5_large_v2": "E5-large",
                 "qwen3_embedding_0_6b": "Qwen3-Emb-0.6B", "e5_small_v2_tar": "TAR E5-small"}
TASK_UNITS = {"classification": "ΔAUC", "regression": "ΔR²"}
MEDIAN_WORD = "median over the seven predictors"


def _scope(text):
    return text.replace("per_column", "per column ").replace("joint", "joint ").strip()


def _label(representation: str) -> str:
    if representation in FAMILY_LABELS:
        return FAMILY_LABELS[representation]
    parts = representation.split("/")
    if parts[0] == "frozen":
        return f"Frozen {ENCODER_NAMES.get(parts[1], parts[1])}, {_scope(parts[2])}"
    if parts[0] == "tar":
        return f"TAR E5-small, {_scope(parts[2])}"
    if parts[0] == "probe":
        return f"Label-trained classifier, {ENCODER_NAMES.get(parts[1], parts[1])}, {parts[2].replace('_', ' ')}"
    if parts[0] == "native":
        return f"{style.PREDICTOR_LABELS.get(parts[1], parts[1])} (native text)"
    if parts[0] == "decision" and len(parts) == 3:
        return f"{style.DECISION_SHORT.get(parts[2], parts[2])}, {parts[1].replace('_', ' ')}"
    return representation


def color_of(representation: str) -> str:
    """Colour = representation: decision view, encoder, label-trained route, native system."""
    parts = representation.split("/")
    if parts[0] == "decision" and len(parts) > 1:
        return style.MODE_COLORS.get(parts[1], style.MEDIAN_COLOR)
    if parts[0] == "decision+frozen":
        return style.DECISION_EMBEDDING_COLOR
    if parts[0] == "frozen":
        return style.EMBEDDING_COLORS.get(parts[1], style.MEDIAN_COLOR)
    if parts[0] == "tar":
        return style.TAR_COLOR
    if parts[0] == "probe":
        return style.PROBE_COLOR
    if parts[0] == "native":
        return style.NATIVE_COLORS.get(parts[1], style.MEDIAN_COLOR)
    if parts[0] == "baseline":
        return style.NON_TEXT_COLOR
    return style.MEDIAN_COLOR


#: Label-trained arms are also told apart by line and marker (PROBE_COLOR is close to E5-small).
LINESTYLES = {"probe": "--", "tar": ":"}
ARM_MARKERS = {"probe": "s", "tar": "^"}


def line_of(representation):
    return LINESTYLES.get(representation.split("/")[0], "-")


def marker_of(representation):
    return ARM_MARKERS.get(representation.split("/")[0], "o")


def _legend(fig, models):
    """Coloured predictor legend used by Group C figures (unchanged for them)."""
    handles = [Line2D([], [], ls="none", marker=MARKERS.get(m, "o"), ms=4,
                      color=PREDICTOR_COLORS.get(m, "black"), label=style.PREDICTOR_LABELS.get(m, m))
               for m in models]
    fig.legend(handles=handles, loc="outside lower center", ncol=min(4, len(handles)), frameon=False,
               fontsize=7.5)


def predictor_handles(models):
    handles = [Line2D([], [], ls="none", marker=MARKERS.get(m, NATIVE_MARKER), ms=4, mfc="none",
                      mec=style.PREDICTOR_INK, label=style.PREDICTOR_LABELS.get(m, m))
               for m in models if m != "native"]
    if "native" in models:
        handles.append(Line2D([], [], ls="none", marker=NATIVE_MARKER, ms=5, color=style.PREDICTOR_INK,
                              label="Native system (vs median no-text baseline)"))
    return handles


def ordered_datasets(dataset_ids):
    """(ids, tasks, names) in the project-wide task order (catalog order within a task)."""
    from src.visualize.study import _ordered_datasets

    return _ordered_datasets(dataset_ids)


def _family_layout(data, *, row_height=0.19, extra=1.1, ncols=1, sharex=False):
    """One panel column per task family, rows = that family's datasets in catalog order."""
    ids, tasks, names = ordered_datasets(set(data.dataset_id))
    families = [f for f in style.TASK_FAMILIES
                if any((t == "regression") == (f == "regression") for t in tasks)]
    rows = {f: [(d, t) for d, t in zip(ids, tasks) if (t == "regression") == (f == "regression")]
            for f in families}
    height = min(style.MAX_HEIGHT, extra + row_height * max(len(r) for r in rows.values()))
    fig, axes = plt.subplots(1, len(families) * ncols, figsize=(style.WIDTH_FULL, height),
                             layout="constrained", squeeze=False, sharex=sharex)
    return fig, axes[0], families, rows, names


def _dataset_rows(ax, rows, names, *, labels=True):
    if labels:
        layout.dataset_axis(ax, [names.get(d, d) for d, _ in rows], [t for _, t in rows], fontsize=7)
    else:  # a panel sharing rows with a labelled one: same limits and rules, no labels
        ax.set_ylim(len(rows) - 0.5, -0.5)
        layout.task_rules(ax, [t for _, t in rows])
        ax.tick_params(axis="y", labelleft=False, length=0)


def _zero(ax, axis="x"):
    (ax.axvline if axis == "x" else ax.axhline)(0, color=style.MEDIAN_COLOR, lw=0.7, zorder=1)


def _family_title(ax, family, n):
    unit = TASK_UNITS[family]
    ax.set_title(f"{'Classification' if family == 'classification' else 'Regression'} ({unit}, {n} datasets)",
                 loc="left", fontsize=8)


# =============================================================================== B1


def gain_landscape(models, medians, *, name, what):
    """Per dataset: every decision model (grey, both view) and the median model per view."""
    if models is None or models.empty or medians is None or medians.empty:
        return
    pool = models[models["mode"].eq("both")]
    fig, axes, families, rows, names = _family_layout(pool, row_height=0.24, extra=1.25)
    offsets = {"per_column": -0.24, "joint": 0.0, "both": 0.24}
    for ax, family in zip(axes, families):
        order = [d for d, _ in rows[family]]
        y = {d: i for i, d in enumerate(order)}
        part = pool[pool.dataset_id.isin(order)]
        jitter = np.random.default_rng(0).uniform(-0.09, 0.09, len(part))
        ax.scatter(part.value, part.dataset_id.map(y) + jitter, s=7, color=style.DECISION_POOL_COLOR,
                   linewidths=0, zorder=2)
        for mode, offset in offsets.items():
            m = medians[medians["mode"].eq(mode) & medians.dataset_id.isin(order)]
            ys = m.dataset_id.map(y) + offset
            ax.hlines(ys, m.low, m.high, color=style.MODE_COLORS[mode], lw=0.8, zorder=3)
            ax.scatter(m.value, ys, marker="D", s=13, color=style.MODE_COLORS[mode], zorder=4,
                       edgecolors="white", linewidths=0.3)
        _zero(ax)
        _dataset_rows(ax, rows[family], names)
        layout.light_grid(ax)
        _family_title(ax, family, len(order))
        ax.set_xlabel(f"Gain over no text ({TASK_UNITS[family]})", fontsize=7.5)
        ax.tick_params(axis="x", labelsize=7)
    handles = [Line2D([], [], ls="none", marker="o", ms=3, color=style.DECISION_POOL_COLOR,
                      label="One decision model (both)")]
    handles += [Line2D([], [], marker="D", ms=3.5, lw=0.8, color=style.MODE_COLORS[m],
                       label=f"Median model, {m.replace('_', ' ')}") for m in offsets]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=7)
    n_models = int(pool.provider.nunique())
    yield fig, name, (
        f"{what}: per dataset, the gain over the same predictor's no-text baseline (five-fold mean, "
        f"{MEDIAN_WORD}) of each of the {n_models} decision models' both-view features (grey dots, "
        "small vertical jitter), and of the per-dataset median decision model for the per-column, "
        "joint and both views (diamonds; the bar spans the seven predictors' values of that median "
        "model). Left: classification datasets in ROC-AUC (binary) or macro one-vs-rest ROC-AUC "
        "(multiclass); right: regression datasets in R². Axes differ between panels.")


def model_rank_heatmap(per_dataset, summary, *, win_rates=None, profile=None, name, what, mode="both"):
    """Rank of each decision model within each dataset (rows: models by mean rank)."""
    if per_dataset is None or per_dataset.empty or summary is None or summary.empty:
        return
    order = list(summary.sort_values(["mean_rank", "provider"]).provider)
    ids, tasks, names = ordered_datasets(set(per_dataset.dataset_id))
    table = per_dataset.pivot_table(index="provider", columns="dataset_id", values="rank").reindex(
        index=order, columns=ids)
    margins = 2 + (win_rates is not None and not win_rates.empty)
    fig = plt.figure(figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 1.75 + 0.165 * len(order))),
                     layout="constrained")
    grid = fig.add_gridspec(1, 1 + margins, width_ratios=[len(ids) * 0.2, 0.9, *([0.9] * (margins - 1))])
    ax = fig.add_subplot(grid[0])
    image = ax.imshow(table.to_numpy(float), cmap=plt.get_cmap(style.AGREEMENT_CMAP + "_r"),
                      vmin=1, vmax=len(order), aspect="auto")
    ax.set_yticks(range(len(order)), [style.DECISION_SHORT.get(p, p) for p in order], fontsize=7)
    ax.set_xticks(range(len(ids)), [names.get(d, d) for d in ids], rotation=90, fontsize=6.5)
    for tick, task in zip(ax.get_xticklabels(), tasks):
        tick.set_color(style.TASK_COLORS[task])
    for position in range(1, len(tasks)):
        if tasks[position] != tasks[position - 1]:
            ax.axvline(position - 0.5, color="white", lw=1.5)
    ax.tick_params(length=0)
    bar = fig.colorbar(image, ax=ax, location="top", shrink=0.5, aspect=30, pad=0.01)
    bar.set_label("Rank within dataset (1 = largest gain)", fontsize=7)
    bar.ax.tick_params(labelsize=6.5)
    rank_ax = fig.add_subplot(grid[1], sharey=ax)
    s = summary.set_index("provider").reindex(order)
    y = np.arange(len(order))
    rank_ax.hlines(y, s.rank_ci_low, s.rank_ci_high, color=style.MEDIAN_COLOR, lw=0.8)
    rank_ax.scatter(s.mean_rank, y, s=10, color=style.MEDIAN_COLOR, zorder=3)
    rank_ax.scatter(s.median_rank, y, s=10, facecolors="none", edgecolors=style.MEDIAN_COLOR, zorder=3)
    rank_ax.set_xlabel("Mean rank (95% CI)\n○ median", fontsize=7)
    rank_ax.tick_params(axis="y", labelleft=False, length=0)
    rank_ax.tick_params(axis="x", labelsize=6.5)
    layout.light_grid(rank_ax)
    if margins == 3:
        win_ax = fig.add_subplot(grid[2], sharey=ax)
        w = win_rates[win_rates.model_id.eq("all")].assign(
            provider=lambda t: t.learner.str.split("/").str[2]).set_index("provider").reindex(order)
        win_ax.hlines(y, w.win_rate_ci_low_dataset_bootstrap, w.win_rate_ci_high_dataset_bootstrap,
                      color=style.MODE_COLORS[mode], lw=0.8)
        win_ax.scatter(w.win_rate, y, s=10, color=style.MODE_COLORS[mode], zorder=3)
        win_ax.axvline(0.5, color=style.MEDIAN_COLOR, lw=0.6, ls=":")
        win_ax.set_xlim(0, 1)
        win_ax.set_xlabel("Fold win rate\nvs no text", fontsize=7)
        win_ax.tick_params(axis="y", labelleft=False, length=0)
        win_ax.tick_params(axis="x", labelsize=6.5)
        layout.light_grid(win_ax)
        if profile is not None and not profile.empty and "parameters_billion" in profile:
            sizes = profile.set_index("provider").parameters_billion.reindex(order)
            for i, value in enumerate(sizes):
                if np.isfinite(value):
                    win_ax.annotate(f"{value:g}B", (1.02, i), xycoords=("axes fraction", "data"),
                                    fontsize=6, va="center", annotation_clip=False)
    yield fig, name, (
        f"{what}: rank of each decision model's {mode.replace('_', '-')}-view gain within each dataset "
        "(1 = largest gain over the no-text baseline; ranks taken within each predictor over the "
        f"complete roster, ties sharing the mean rank, then averaged over the seven predictors). Rows "
        f"are ordered by the mean rank over {table.shape[1]} datasets; columns are datasets grouped by "
        "task (label colour). Margin: mean rank with a 95% dataset-bootstrap percentile interval "
        "(2,000 draws) and the median rank (open circle)"
        + ("; fold-level win rate against the same predictor without text over dataset x predictor x "
           "fold cells, with a 95% interval from resampling datasets (all their folds together)"
           if margins == 3 else "")
        + ("; parameter count where verified." if profile is not None else "."))


def view_contrast(contrasts, *, name, what, single_text=()):
    """Joint - per column and both - per column per dataset (>= 2 text columns)."""
    if contrasts is None or contrasts.empty:
        return
    ids, tasks, names = ordered_datasets(set(contrasts.dataset_id))
    families = [f for f in style.TASK_FAMILIES if any((t == "regression") == (f == "regression") for t in tasks)]
    rows = {f: [(d, t) for d, t in zip(ids, tasks) if (t == "regression") == (f == "regression")]
            for f in families}
    kinds = [k for k in ("joint_minus_per_column", "both_minus_per_column") if k in set(contrasts.contrast)]
    fig, axes = plt.subplots(len(families), len(kinds), figsize=(style.WIDTH_FULL, min(
        style.MAX_HEIGHT, 1.2 + 0.2 * len(ids))), layout="constrained", squeeze=False,
        gridspec_kw={"height_ratios": [len(rows[f]) for f in families]}, sharey="row")
    for i, family in enumerate(families):
        order = [d for d, _ in rows[family]]
        y = {d: k for k, d in enumerate(order)}
        for j, kind in enumerate(kinds):
            ax = axes[i, j]
            part = contrasts[contrasts.contrast.eq(kind) & contrasts.dataset_id.isin(order)]
            jitter = np.random.default_rng(1).uniform(-0.12, 0.12, len(part))
            ax.scatter(part.value, part.dataset_id.map(y) + jitter, s=7, color=style.DECISION_POOL_COLOR,
                       linewidths=0, zorder=2)
            med = part.groupby("dataset_id").value.median()
            view = kind.split("_minus")[0]
            ax.scatter(med.to_numpy(), med.index.map(y), marker="D", s=14, color=style.MODE_COLORS[view],
                       zorder=3, edgecolors="white", linewidths=0.3)
            _zero(ax)
            layout.light_grid(ax)
            _dataset_rows(ax, rows[family], names, labels=j == 0)
            ax.tick_params(axis="x", labelsize=7)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            if i == 0:
                ax.set_title(f"{view.capitalize()} minus per column", loc="left", fontsize=8)
            ax.set_xlabel(f"Paired difference ({TASK_UNITS[family]})", fontsize=7)
    handles = [Line2D([], [], ls="none", marker="o", ms=3, color=style.DECISION_POOL_COLOR,
                      label="One decision model"),
               *[Line2D([], [], ls="none", marker="D", ms=3.5, color=style.MODE_COLORS[k.split('_minus')[0]],
                        label=f"Median over models ({k.split('_minus')[0]})") for k in kinds]]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=7)
    same = ", ".join(sorted(single_text)) if single_text else "none"
    yield fig, name, (
        f"{what}: per dataset with at least two text columns ({len(ids)} datasets), the paired "
        "difference between a decision model's joint (left) or both (right) view and its per-column "
        f"view (difference of the five-fold mean gains within each predictor, then {MEDIAN_WORD}); "
        "grey dots are individual decision models, diamonds their median. Rows are split by task "
        "family, with ROC-AUC/macro AUC (classification) and R² (regression) on separate axes. "
        f"Datasets with one text column, where the three views are the same block: {same}.")


def gain_forest(summary, representations, *, name, what, share=False, separators=(), median=True):
    """Equal-dataset mean (filled) and median (open) gain per arm and predictor, with the
    dataset-bootstrap 95% interval of the mean; raw gains in one panel per task family."""
    if summary is None or summary.empty:
        return
    data = summary[summary.representation.isin(representations)]
    if share:
        data = data[data.task_family.eq("all")] if "task_family" in data else data
    elif "task_family" in data:
        data = data[data.task_family.isin(style.TASK_FAMILIES)]
    else:
        return
    if data.empty:
        return
    rows = [r for r in representations if r in set(data.representation)]
    models = [m for m in (*PREDICTORS, "native") if m in set(data.model_id)]
    panels = ["all"] if share else [f for f in style.TASK_FAMILIES if f in set(data.task_family)]
    fig, axes = plt.subplots(1, len(panels), sharey=True, layout="constrained", squeeze=False,
                             figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 1.0 + 0.3 * len(rows))))
    offsets = dict(zip(models, np.linspace(-0.32, 0.32, max(len(models), 1))))
    mean, low, high, med = (("mean_headroom_share", "share_ci_low", "share_ci_high", "median_headroom_share")
                            if share else ("mean_raw_gain", "raw_ci_low", "raw_ci_high", "median_raw_gain"))
    for ax, panel in zip(axes[0], panels):
        part = data if share else data[data.task_family.eq(panel)]
        for model in models:
            frame = part[part.model_id.eq(model)].set_index("representation").reindex(rows)
            y = np.arange(len(rows)) + (0 if model == "native" else offsets[model])
            colors = [color_of(r) for r in rows]
            ax.hlines(y, frame[low], frame[high], colors=colors, lw=0.7, zorder=2)
            marker = MARKERS.get(model, NATIVE_MARKER)
            ax.scatter(frame[mean], y, marker=marker, s=14 if model != "native" else 30, c=colors,
                       linewidths=0, zorder=3)
            if median:
                ax.scatter(frame[med], y, marker=marker, s=12, facecolors="none", edgecolors=colors,
                           linewidths=0.6, zorder=3)
        _zero(ax)
        for position in separators:
            ax.axhline(position - 0.5, color=style.GRID_COLOR, lw=0.8)
        layout.light_grid(ax)
        ax.tick_params(axis="x", labelsize=7)
        if share:
            n = int(part.datasets.max())
            ax.set_xlabel(f"Share of the baseline headroom gained (all {n} datasets)", fontsize=7.5)
        else:
            n = int(part.datasets.max())
            _family_title(ax, panel, n)
            ax.set_xlabel(f"Gain over no text ({TASK_UNITS[panel]})", fontsize=7.5)
    axes[0, 0].set_yticks(range(len(rows)), [_label(r) for r in rows], fontsize=7)
    for tick, r in zip(axes[0, 0].get_yticklabels(), rows):
        tick.set_color(color_of(r))
    axes[0, 0].set_ylim(len(rows) - 0.5, -0.5)
    axes[0, 0].tick_params(axis="y", length=0)
    handles = predictor_handles(models)
    if median:
        handles.append(Line2D([], [], ls="none", marker=None, label="filled = mean, open = median"))
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=7)
    quantity = ("share of the baseline headroom gained, (score with text - score without) / (1 - "
                "score without), pooled over all datasets" if share else
                "gain over the same predictor's no-text baseline, per task family (ROC-AUC or macro "
                "AUC for classification, R² for regression; separate axes, never averaged together)")
    yield fig, name, (
        f"{what}: equal-dataset mean (filled marker) and median (open marker) {quantity}; lines are "
        "95% percentile intervals of the mean from resampling datasets (2,000 draws). One marker "
        "shape per predictor; colour marks the representation. Each dataset contributes its five-fold "
        "mean; median-model rows take the per-dataset median over all seventeen decision models.")


NORMALIZED_TEXT = (
    "Per dataset and outer fold, each configuration's headline score (binary ROC-AUC, multiclass "
    "macro one-vs-rest ROC-AUC, regression R² with negative values clipped at 0) is min-max "
    "normalised across all {configurations} compared configurations (predictor x arm), so 1 is the "
    "best and 0 the worst of them on that dataset; folds on which all tie are skipped, and the fold "
    "mean is each dataset's score (MulTaBench-style pooling of AUC and R²). The score is relative to "
    "this comparison set and complements, never replaces, the per-task-family gains.")


def normalized_score_forest(summary, representations, *, name, what, separators=(), labels=None,
                            colors=None, note=""):
    """Equal-dataset mean (filled) and median (open) normalised score per arm and predictor over
    all datasets, with the dataset-bootstrap 95% interval of the mean (`group_b.summarize_normalized`)."""
    if summary is None or summary.empty:
        return
    data = summary[summary.task_family.eq("all") & summary.representation.isin(representations)]
    if data.empty:
        return
    rows = [r for r in representations if r in set(data.representation)]

    def label(representation):
        return (labels or {}).get(representation) or _label(representation)

    def color(representation):
        return (colors or {}).get(representation) or color_of(representation)

    models = [m for m in PREDICTORS if m in set(data.model_id)]
    natives = sorted(set(data.model_id) - set(PREDICTORS))
    fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 1.0 + 0.3 * len(rows))),
                           layout="constrained")
    offsets = dict(zip(models, np.linspace(-0.32, 0.32, max(len(models), 1))))
    for model in [*models, *natives]:
        frame = data[data.model_id.eq(model)].set_index("representation").reindex(rows)
        y = np.arange(len(rows)) + offsets.get(model, 0.0)
        colors_ = [color(r) for r in rows]
        marker = MARKERS.get(model, NATIVE_MARKER)
        size = 14 if model in offsets else 30
        ax.hlines(y, frame.normalized_ci_low, frame.normalized_ci_high, colors=colors_, lw=0.7, zorder=2)
        ax.scatter(frame.mean_normalized_score, y, marker=marker, s=size, c=colors_, linewidths=0, zorder=3)
        ax.scatter(frame.median_normalized_score, y, marker=marker, s=size * 0.85, facecolors="none",
                   edgecolors=colors_, linewidths=0.6, zorder=3)
    for position in separators:
        ax.axhline(position - 0.5, color=style.GRID_COLOR, lw=0.8)
    ax.set_xlim(-0.02, 1.02)
    layout.light_grid(ax)
    ax.tick_params(axis="x", labelsize=7)
    n, k = int(data.datasets.max()), int(data.configurations.max())
    ax.set_xlabel(f"Normalised score (0 = worst, 1 = best of {k} configurations; all {n} datasets)",
                  fontsize=7.5)
    ax.set_yticks(range(len(rows)), [label(r) for r in rows], fontsize=7)
    for tick, r in zip(ax.get_yticklabels(), rows):
        tick.set_color(color(r))
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    handles = predictor_handles(models)
    if natives:
        handles.append(Line2D([], [], ls="none", marker=NATIVE_MARKER, ms=5, color=style.PREDICTOR_INK,
                              label="Native-text system"))
    handles.append(Line2D([], [], ls="none", marker=None, label="filled = mean, open = median"))
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=7)
    yield fig, name, (
        f"{what}: equal-dataset mean (filled marker) and median (open marker) normalised score; lines are "
        "95% percentile intervals of the mean from resampling datasets (2,000 draws). One marker shape "
        "per predictor; colour marks the representation. " + NORMALIZED_TEXT.format(configurations=k)
        + (f" {note}" if note else ""))


def fold_win_rate_forest(win_rates, learners, *, name, what):
    """Fold-level win rate against the same predictor without text, per predictor and pooled."""
    if win_rates is None or win_rates.empty:
        return
    data = win_rates[win_rates.learner.isin(learners)]
    if data.empty:
        return
    rows = [r for r in learners if r in set(data.learner)]
    models = [m for m in PREDICTORS if m in set(data.model_id)]
    fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 1.2 + 0.42 * len(rows))),
                           layout="constrained")
    offsets = dict(zip([*models, "all"], np.linspace(-0.33, 0.33, len(models) + 1)))
    for model in [*models, "all"]:
        frame = data[data.model_id.eq(model)].set_index("learner").reindex(rows)
        y = np.arange(len(rows)) + offsets[model]
        colors = [color_of(r) for r in rows]
        ax.hlines(y, frame.win_rate_ci_low_dataset_bootstrap, frame.win_rate_ci_high_dataset_bootstrap,
                  colors=colors, lw=0.7 if model != "all" else 1.4)
        ax.scatter(frame.win_rate, y, marker=MARKERS.get(model, "D"), s=14 if model != "all" else 24,
                   c=colors if model != "all" else style.PREDICTOR_INK, zorder=3, linewidths=0)
    ax.axvline(0.5, color=style.MEDIAN_COLOR, lw=0.7, ls=":")
    ax.set_xlim(0, 1)
    ax.set_yticks(range(len(rows)), [_label(r) for r in rows], fontsize=7)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Share of (dataset, fold) cells won against the same predictor without text", fontsize=7.5)
    layout.light_grid(ax)
    handles = predictor_handles(models) + [Line2D([], [], ls="none", marker="D", ms=4,
                                                  color=style.PREDICTOR_INK, label="All predictors pooled")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=7)
    cells = int(data.fold_cells.max())
    excluded = int(data.excluded_incomplete_cells.max())
    yield fig, name, (
        f"{what}: share of paired (dataset, outer fold) cells on which the arm's headline score exceeds "
        "the same predictor's no-text score (ties count as not won), per predictor and pooled over "
        f"predictors (diamond; up to {cells} cells). Lines are 95% percentile intervals from resampling "
        "datasets with all their folds (cluster bootstrap, 2,000 draws); folds are never resampled "
        f"alone. Descriptive, no p-value; at most {excluded} incomplete cells were excluded.")


def model_by_predictor(gains, *, order=None, mode="both", name, what):
    """Equal-dataset mean raw gain of every decision model (rows) in every predictor, per family."""
    if gains is None or gains.empty:
        return
    data = gains[gains.representation.str.startswith(f"decision/{mode}/")
                 & ~gains.representation.str.endswith("/median")]
    if data.empty:
        return
    from src.evaluation.group_b import task_family

    data = data.assign(provider=data.representation.str.split("/").str[2],
                       task_family=data.task_type.map(task_family))
    order = order or sorted(set(data.provider))
    families = [f for f in style.TASK_FAMILIES if f in set(data.task_family)]
    fig, axes = plt.subplots(1, len(families), figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 1.5 + 0.17 * len(order))),
                             layout="constrained", squeeze=False, sharey=True)
    for ax, family in zip(axes[0], families):
        part = data[data.task_family.eq(family)]
        table = part.pivot_table(index="provider", columns="model_id", values="raw_gain", aggfunc="mean")
        table = table.reindex(index=order, columns=[m for m in PREDICTORS if m in table.columns])
        limit = float(np.nanmax(np.abs(table.to_numpy()))) or 1.0
        image = ax.imshow(table.to_numpy(), cmap=style.EFFECT_CMAP, vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(table.shape[1]), [style.PREDICTOR_LABELS.get(m, m) for m in table.columns],
                      fontsize=6.5, rotation=40, ha="right")
        ax.set_yticks(range(len(order)), [style.DECISION_SHORT.get(p, p) for p in order], fontsize=7)
        ax.tick_params(length=0)
        _family_title(ax, family, int(part.dataset_id.nunique()))
        bar = fig.colorbar(image, ax=ax, shrink=0.7, pad=0.02)
        bar.set_label(f"Mean {TASK_UNITS[family]}", fontsize=7)
        bar.ax.tick_params(labelsize=6.5)
    yield fig, name, (
        f"{what}: equal-dataset mean gain of each decision model's {mode.replace('_', '-')}-view features "
        "over the same predictor's no-text baseline, per predictor (columns), separately for "
        "classification (ROC-AUC/macro AUC) and regression (R²) with separate colour scales centred "
        "at zero. Rows follow the decision models' mean within-dataset rank.")


def predictor_rank_agreement(agreement, *, name, what):
    """Kendall tau_b of the decision-model ranking between every pair of predictors."""
    if agreement is None or agreement.empty:
        return
    table = agreement.pivot_table(index="predictor_a", columns="predictor_b", values="tau_b")
    models = [m for m in PREDICTORS if m in table.index]
    table = table.reindex(index=models, columns=models)
    fig, ax = plt.subplots(figsize=(style.WIDTH_HALF, 2.9), layout="constrained")
    image = ax.imshow(table.to_numpy(float), cmap=style.AGREEMENT_DIVERGING, vmin=-1, vmax=1)
    for (i, j), value in np.ndenumerate(table.to_numpy(float)):
        if i != j and np.isfinite(value):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(value) > 0.6 else style.TEXT_COLOR)
    labels = [style.PREDICTOR_LABELS.get(m, m) for m in models]
    ax.set_xticks(range(len(models)), labels, rotation=45, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(models)), labels, fontsize=6.5)
    ax.tick_params(length=0)
    bar = fig.colorbar(image, ax=ax, shrink=0.8)
    bar.set_label("Kendall τ_b", fontsize=7)
    bar.ax.tick_params(labelsize=6.5)
    n_models, n_data = int(agreement.models.max()), int(agreement.datasets.max())
    yield fig, name, (
        f"{what}: Kendall τ_b between two predictors' rankings of the {n_models} decision models, each "
        f"ranking ordering models by their mean within-dataset rank of the gain over {n_data} datasets "
        "(both view; 1 = identical order, 0 = unrelated).")


# =============================================================================== B2


def paired_scatter(panels, *, name, what):
    """Per dataset: decision gain (y) against an alternative arm's gain (x), identity line.

    `panels` maps a row title to a `group_b_views.paired_gains` table; columns are task families.
    """
    panels = {k: v for k, v in panels.items() if v is not None and not v.empty}
    if not panels:
        return
    families = [f for f in style.TASK_FAMILIES
                if any(f in set(t.task_family) for t in panels.values())]
    fig, axes = plt.subplots(len(panels), len(families), squeeze=False, layout="constrained",
                             figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 0.6 + 2.15 * len(panels))))
    for i, (title, table) in enumerate(panels.items()):
        for j, family in enumerate(families):
            ax = axes[i, j]
            part = table[table.task_family.eq(family)]
            colors = [style.TASK_COLORS[t] for t in part.task_type]
            ax.hlines(part.y, part.x_low, part.x_high, colors=colors, lw=0.5, alpha=0.6)
            ax.vlines(part.x, part.y_low, part.y_high, colors=colors, lw=0.5, alpha=0.6)
            ax.scatter(part.x, part.y, s=14, c=colors, zorder=3, linewidths=0)
            values = np.concatenate([part[c].to_numpy(float) for c in ("x_low", "x_high", "y_low", "y_high")])
            values = values[np.isfinite(values)]
            lo, hi = (min(values.min(), 0), max(values.max(), 0)) if len(values) else (-1, 1)
            pad = 0.05 * (hi - lo or 1)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=style.MEDIAN_COLOR, lw=0.6, ls="--")
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            _zero(ax, "x")
            _zero(ax, "y")
            ax.tick_params(labelsize=6.5)
            unit = TASK_UNITS[family]
            ax.set_xlabel(f"{_label(part.x_representation.iloc[0]) if len(part) else title} ({unit})",
                          fontsize=7)
            ax.set_ylabel(f"{_label(part.y_representation.iloc[0]).replace(' (median model)', '')} ({unit})"
                          if len(part) else unit, fontsize=7)
            above = int((part.y > part.x).sum())
            ax.set_title(f"{'Classification' if family == 'classification' else 'Regression'}: "
                         f"{above}/{len(part)} above the line", loc="left", fontsize=7.5)
    fig.legend(handles=layout.task_handles(marker=True), loc="outside lower center", ncol=3,
               frameon=False, fontsize=7)
    yield fig, name, (
        f"{what}: per dataset, the gain over the no-text baseline of the median decision model "
        "(y) against the alternative arm (x), each the median over the seven predictors of the "
        "five-fold mean gain; thin lines span the predictors' values. The dashed line marks equal "
        "gains; points above it favour the decision features. Classification (ROC-AUC/macro AUC) "
        "and regression (R²) use separate axes; marker colour gives the task.")


COMPACT_MARKERS = {"per_column": "o", "joint": "s", "both": "D", "per_column32": "o", "joint64": "s"}


def compactness(table, *, name, what):
    """Median gain over predictors against the number of added columns (log x), per arm."""
    if table is None or table.empty:
        return
    families = [f for f in style.TASK_FAMILIES if f in set(table.task_family)]
    fig, axes = plt.subplots(1, len(families), figsize=(style.WIDTH_FULL, 2.9), layout="constrained",
                             squeeze=False)
    arms = sorted(set(table.representation), key=lambda r: (r.split("/")[0] != "decision",
                                                            r.split("/")[0], r))
    for ax, family in zip(axes[0], families):
        part = table[table.task_family.eq(family)]
        for arm in arms:
            rows = part[part.representation.eq(arm)]
            ax.scatter(rows["columns"].clip(lower=1), rows.value, s=12, color=color_of(arm),
                       marker=COMPACT_MARKERS.get(arm.split("/")[-1] if not arm.endswith("/median")
                                                  else arm.split("/")[1], "*"),
                       linewidths=0, alpha=0.85)
        ax.set_xscale("log", base=2)
        _zero(ax, "y")
        layout.light_grid(ax, axis="both")
        ax.tick_params(labelsize=6.5)
        ax.set_xlabel("Added representation columns (log scale)", fontsize=7)
        ax.set_ylabel(f"Gain over no text ({TASK_UNITS[family]})", fontsize=7)
        _family_title(ax, family, int(part.dataset_id.nunique()))
    handles = [Line2D([], [], ls="none", ms=4, color=color_of(a),
                      marker=COMPACT_MARKERS.get(a.split("/")[-1] if not a.endswith("/median") else a.split("/")[1], "*"),
                      label=_label(a).replace(" (median model)", "")) for a in arms]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=6.5)
    yield fig, name, (
        f"{what}: per dataset and arm, the gain over the no-text baseline ({MEDIAN_WORD} of the "
        "five-fold mean gain) against the number of representation columns the arm adds (median over "
        "predictors and folds; decision rows use the median model's gain and the roster's median "
        "width), on a log2 axis. Classification (ROC-AUC/macro AUC) and regression (R²) are separate "
        "panels with separate axes.")


# =============================================================================== B4


SIZE_ARMS = ("decision/both/median", "frozen/e5_small_v2/per_column32", "probe/e5_small_v2/per_column")


def _size_ticks(ax, sizes):
    ax.set_xticks(np.log2(sizes), [f"{s // 1024}k" if s >= 1024 else str(s) for s in sizes], fontsize=6.5)


def size_curves(curves, *, name, what, curve="main", representations=SIZE_ARMS, outcome="raw_gain",
                model="all"):
    """Pooled gain against labelled training rows (log2), one panel per task family."""
    if curves is None or curves.empty:
        return
    data = curves[curves.curve.eq(curve) & curves.representation.isin(representations)]
    if data.empty:
        return
    families = [f for f in style.TASK_FAMILIES if f in set(data.task_family)]
    fig, axes = plt.subplots(1, len(families), figsize=(style.WIDTH_FULL, 2.8), layout="constrained",
                             squeeze=False)
    arms = [r for r in representations if r in set(data.representation)]
    for ax, family in zip(axes[0], families):
        part = data[data.task_family.eq(family)]
        for arm in arms:
            rows = part[part.representation.eq(arm)]
            main = rows[rows.model_id.eq(model)].sort_values("training_size")
            x = np.log2(main.training_size)
            if model == "all":
                spread = rows[rows.model_id.ne("all")].groupby("training_size").pooled.agg(["min", "max"])
                ax.fill_between(np.log2(spread.index), spread["min"], spread["max"], color=color_of(arm),
                                alpha=0.08, lw=0)
            ax.fill_between(x, main.ci_low, main.ci_high, color=color_of(arm), alpha=0.22, lw=0)
            ax.plot(x, main.pooled, color=color_of(arm), lw=1.2, marker=marker_of(arm), ms=2.5,
                    ls=line_of(arm))
        _zero(ax, "y")
        _size_ticks(ax, sorted(set(part.training_size)))
        layout.light_grid(ax, axis="both")
        ax.tick_params(axis="y", labelsize=6.5)
        ax.set_xlabel("Labelled training rows (log2 scale)", fontsize=7)
        ax.set_ylabel(("Gain over same-size no text" if outcome == "raw_gain" else "Headroom share")
                      + f" ({TASK_UNITS[family]})" * (outcome == "raw_gain"), fontsize=7)
        _family_title(ax, family, int(part.datasets.max()))
    handles = [Line2D([], [], color=color_of(a), lw=1.2, ls=line_of(a), marker=marker_of(a), ms=2.5,
                      label=_label(a)) for a in arms]
    if model == "all":
        handles.append(Patch(color=style.MEDIAN_COLOR, alpha=0.1, label="Range of the seven predictors"))
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, fontsize=7)
    scope = ("main curve: every dataset at 128 to 2,048 rows" if curve == "main" else
             "extended curve: the seven datasets with at least 16,384 training rows in every fold, at every size")
    who = ("the mean over the seven predictors (per dataset and draw)" if model == "all"
           else style.PREDICTOR_LABELS.get(model, model))
    yield fig, name, (
        f"{what} ({scope}): gain over the same-size, same-draw no-text baseline for {who}, against the "
        "number of sampled labelled training rows (nested stratified subsamples; test rows unchanged). "
        "Lines are DerSimonian-Laird random-effects means over datasets per size (five-fold means, "
        "averaged over draws; draw variance as within-dataset variance) with their 95% interval "
        + ("(dark band); the faint band spans the seven predictors' own pooled curves. " if model == "all"
           else "(band). ")
        + "The decision line uses the per-dataset median over all seventeen decision models. "
        "Classification (ROC-AUC/macro AUC) and regression (R²) are separate panels.")


def size_curves_by_predictor(curves, *, name, what, family, representations=SIZE_ARMS):
    """Supporting: one small panel per predictor (pooled curves, main curve) for one family."""
    if curves is None or curves.empty:
        return
    data = curves[curves.curve.eq("main") & curves.task_family.eq(family)
                  & curves.representation.isin(representations) & curves.model_id.ne("all")]
    if data.empty:
        return
    models = [m for m in PREDICTORS if m in set(data.model_id)]
    fig, axes = plt.subplots(2, 4, figsize=(style.WIDTH_FULL, 3.6), layout="constrained", sharex=True,
                             sharey=True)
    arms = [r for r in representations if r in set(data.representation)]
    for ax, model in zip(axes.flat, models):
        for arm in arms:
            rows = data[data.model_id.eq(model) & data.representation.eq(arm)].sort_values("training_size")
            ax.fill_between(np.log2(rows.training_size), rows.ci_low, rows.ci_high, color=color_of(arm),
                            alpha=0.2, lw=0)
            ax.plot(np.log2(rows.training_size), rows.pooled, color=color_of(arm), lw=1, ls=line_of(arm))
        _zero(ax, "y")
        _size_ticks(ax, sorted(set(data.training_size)))
        ax.tick_params(axis="y", labelsize=6.5)
        ax.set_title(style.PREDICTOR_LABELS.get(model, model), loc="left", fontsize=7.5)
        layout.light_grid(ax, axis="y")
    for ax in axes.flat[len(models):]:
        ax.axis("off")
    fig.legend(handles=[Line2D([], [], color=color_of(a), lw=1.2, ls=line_of(a), label=_label(a))
                        for a in arms], loc="outside lower center", ncol=3, frameon=False, fontsize=6.5)
    fig.supxlabel("Labelled training rows (log2 scale)", fontsize=7.5)
    fig.supylabel(f"Gain over same-size no text ({TASK_UNITS[family]})", fontsize=7.5)
    yield fig, name, (
        f"{what}, {family} datasets ({TASK_UNITS[family]}), main curve: per predictor, the DerSimonian-"
        "Laird random-effects mean gain over the same-size no-text baseline with its 95% interval, "
        "against the number of sampled labelled training rows. The decision line is the per-dataset "
        "median over all seventeen decision models.")


def size_slopes(slopes, *, name, what, representations=SIZE_ARMS):
    """Per dataset: slope of the predictor-averaged gain per doubling of labelled rows."""
    if slopes is None or slopes.empty:
        return
    data = slopes[slopes.representation.isin(representations)]
    if data.empty:
        return
    fig, axes, families, rows, names = _family_layout(data, row_height=0.22, extra=1.2)
    arms = [r for r in representations if r in set(data.representation)]
    offsets = dict(zip(arms, np.linspace(-0.22, 0.22, len(arms))))
    for ax, family in zip(axes, families):
        order = [d for d, _ in rows[family]]
        y = {d: i for i, d in enumerate(order)}
        for arm in arms:
            part = data[data.representation.eq(arm) & data.dataset_id.isin(order)]
            ys = part.dataset_id.map(y) + offsets[arm]
            ax.hlines(ys, part.low, part.high, color=color_of(arm), lw=0.6)
            ax.scatter(part.value, ys, s=12, color=color_of(arm), marker=marker_of(arm), zorder=3, linewidths=0)
        _zero(ax)
        _dataset_rows(ax, rows[family], names)
        layout.light_grid(ax)
        _family_title(ax, family, len(order))
        ax.set_xlabel(f"Change in gain per doubling ({TASK_UNITS[family]})", fontsize=7)
        ax.tick_params(axis="x", labelsize=6.5)
    fig.legend(handles=[Line2D([], [], ls="none", marker=marker_of(a), ms=4, color=color_of(a), label=_label(a))
                        for a in arms], loc="outside lower center", ncol=3, frameon=False, fontsize=7)
    yield fig, name, (
        f"{what}: per dataset, the least-squares slope of the gain over the same-size no-text baseline on "
        "log2 labelled rows (128 to 2,048 rows; draws averaged), as the median over the seven "
        "predictors' slopes (dot) and their range (line). Negative slopes mean the gain shrinks as "
        "labelled rows double. Separate panels and axes for classification and regression.")


# =============================================================================== B5


def native_references(scores, *, name, what):
    """Native-text systems' absolute headline score beside the predictors with and without text."""
    if scores is None or scores.empty or not scores.representation.str.startswith("native/").any():
        return
    fig, axes, families, rows, names = _family_layout(scores, row_height=0.25, extra=1.3)
    natives = [m for m in style.NATIVE_COLORS if f"native/{m}" in set(scores.representation)]
    for ax, family in zip(axes, families):
        order = [d for d, _ in rows[family]]
        y = {d: i for i, d in enumerate(order)}
        part = scores[scores.dataset_id.isin(order)]
        base = part[part.representation.eq("baseline")]
        ax.scatter(base.value, base.dataset_id.map(y) - 0.18, s=7, color=style.NON_TEXT_COLOR, linewidths=0)
        dec = part[part.representation.eq("decision/both/median")].groupby("dataset_id").value.agg(
            ["min", "median", "max"])
        ys = dec.index.map(y)
        ax.hlines(ys, dec["min"], dec["max"], color=style.MODE_COLORS["both"], lw=1.6, alpha=0.6)
        ax.scatter(dec["median"], ys, s=10, color=style.MODE_COLORS["both"], zorder=3, linewidths=0)
        for model in natives:
            rows_n = part[part.representation.eq(f"native/{model}")]
            ax.scatter(rows_n.value, rows_n.dataset_id.map(y) + 0.18, s=16, marker=style.NATIVE_MARKERS[model],
                       color=style.NATIVE_COLORS[model], zorder=4, linewidths=0)
        _dataset_rows(ax, rows[family], names)
        layout.light_grid(ax)
        metric = "ROC-AUC / macro AUC" if family == "classification" else "R²"
        ax.set_title(f"{'Classification' if family == 'classification' else 'Regression'} ({metric}, "
                     f"{len(order)} datasets)", loc="left", fontsize=8)
        ax.set_xlabel(f"Five-fold mean {metric}", fontsize=7)
        ax.tick_params(axis="x", labelsize=6.5)
    handles = [Line2D([], [], ls="none", marker="o", ms=3, color=style.NON_TEXT_COLOR,
                      label="Seven predictors, no text"),
               Line2D([], [], marker="o", ms=3, lw=1.6, alpha=0.8, color=style.MODE_COLORS["both"],
                      label="Median decision model (both), range over predictors")]
    handles += [Line2D([], [], ls="none", marker=style.NATIVE_MARKERS[m], ms=4, color=style.NATIVE_COLORS[m],
                       label=style.PREDICTOR_LABELS[m]) for m in natives]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=7)
    yield fig, name, (
        f"{what}: five-fold mean headline score per dataset of each native-text system (coloured "
        "markers), of each of the seven predictors without text (grey) and of the per-dataset median "
        "decision model's both-view arm in each predictor (green dot: median over predictors; bar: "
        "their range). Absolute scores, not gains: native systems share no predictor with the "
        "decision-feature arms and are descriptive whole-system references.")


def render(save, generator):
    """Save (PDF + caption), display and close every figure of a generator."""
    import matplotlib.pyplot as plt

    from src.utils.notebook_display import display_local

    for fig, name, caption in generator:
        save(fig, name, caption=caption)
        display_local(fig)
        plt.close(fig)
