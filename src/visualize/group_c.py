"""Group C figures (C1 inside Strands, C2 training Strands, C synthesis), at final A4 width.

Every generator yields ``(figure, name, caption)`` and yields nothing for an empty table, so
the notebooks show a waiting state instead of an empty figure. Inputs are the tables of
`src.evaluation.group_c_views`: per-dataset gains ("arm minus the same predictor's no-text
baseline") or fold-paired contrasts ("A minus B") in the fixed headline metric, summarised
within a task family (classification ΔAUC, regression ΔR², never averaged together). Colour
is the representation (Strands ramp, TAR, label-trained classifier); predictors are marker
shapes in one ink; datasets carry equal weight and intervals resample datasets.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src.evaluation.group_c_views import CELLS, FACTORIAL, PREDICTOR_MEAN, PROBES, TAR, ZERO_SHOT
from src.visualize import layout, style

PREDICTORS = style.MAIN_PREDICTORS
FAMILIES = style.TASK_FAMILIES
FAMILY_UNITS = {"classification": ("Classification", "ΔAUC"), "regression": ("Regression", "ΔR²")}
POSITIONS = ("state_end", "question_end", "answer")
POSITION_LABELS = {"state_end": "end of text", "question_end": "end of question", "answer": "<answer>"}
VIEW_LABELS = {"per_column": "per column", "joint": "joint"}
CELL_LABELS = {"zero_shot": "Zero-shot", "head_only": "Head only", "lora_only": "LoRA only",
               "lora_head": "LoRA + head"}
CELL_COLORS = {"zero_shot": style.STRANDS_COLORS["zero_shot"], "head_only": style.STRANDS_COLORS["head"],
               "lora_only": style.STRANDS_COLORS["lora"], "lora_head": style.STRANDS_COLORS["lora_head"]}
ENCODERS = {"e5_small_v2_tar": "E5-small", "e5_small_v2": "E5-small", "e5_large_v2": "E5-large",
            "qwen3_embedding_0_6b": "Qwen3-Emb-0.6B"}
SCOPES = {"per_column32": "per column PCA32", "joint64": "joint PCA64", "joint": "joint"}
COMPARISON_LABELS = {
    "answer_last_vs_strands_probabilities": "<answer> L24 vs probabilities",
    "answer_last_plus_probabilities_vs_probabilities": "<answer> L24 + probabilities vs probabilities",
    "question_end_vs_state_end": "end of question vs end of text",
    "answer_vs_question_end": "<answer> vs end of question",
    "last_vs_middle_layer": "layer 24 vs layer 12",
    "lora_vs_zero_shot": "LoRA only vs zero-shot",
    "head_only_vs_zero_shot": "Head only vs zero-shot",
    "lora_vs_head_only": "LoRA only vs head only",
    "lora_head_vs_lora_only": "LoRA + head vs LoRA only",
}
#: C1 contrast groups of the combined forest, in reading order.
C1_GROUPS = (
    ("Vector against and with Strands' probabilities",
     ("answer_last_vs_strands_probabilities", "answer_last_plus_probabilities_vs_probabilities")),
    ("Position path (later minus earlier)", ("question_end_vs_state_end", "answer_vs_question_end")),
    ("Layer 24 minus layer 12", ("last_vs_middle_layer",)),
)
C2_CELL_CONTRASTS = ("lora_vs_zero_shot", "head_only_vs_zero_shot", "lora_vs_head_only",
                     "lora_head_vs_lora_only")
MEAN_MARKER = "D"
CI_TEXT = "95% dataset-bootstrap percentile interval (2,000 draws)"
RAW_TEXT = ("raw gains stay within a task family (classification: binary ROC-AUC and multiclass macro "
            "one-vs-rest ROC-AUC; regression: R²)")


# ----------------------------------------------------------------------------- labels


def label(representation: str) -> str:
    """Readable name of a Group C arm, reference arm or contrast label."""
    parts = representation.split("/")
    if parts[0] == "hidden":
        return f"{POSITION_LABELS[parts[2]]}, layer {parts[3][1:]}, {VIEW_LABELS[parts[1]]}"
    if parts[0] == "hidden+decision":
        return f"<answer> L24 + probabilities, {VIEW_LABELS[parts[1]]}"
    if parts[0] == "decision":
        return f"Strands probabilities, {VIEW_LABELS[parts[1]]}"
    if parts[0] == "adapted":
        return CELL_LABELS[parts[2]]
    if parts[0] == "tar":
        return f"TAR {ENCODERS.get(parts[1], parts[1])}, {SCOPES.get(parts[2], parts[2])}"
    if parts[0] == "probe":
        return f"Label-trained classifier, {ENCODERS.get(parts[1], parts[1])}"
    if parts[0] in COMPARISON_LABELS:
        extra = [VIEW_LABELS.get(p) or POSITION_LABELS.get(p) or f"L{p}" for p in parts[1:]]
        return COMPARISON_LABELS[parts[0]] + (f" ({', '.join(extra)})" if extra else "")
    cell = parts[0].split("_vs_")[0]
    if cell in CELL_LABELS and len(parts) >= 4:  # <cell>_vs_<family>/<family>/<encoder>/<scope>
        family = {"tar": "TAR", "probe": "classifier"}[parts[1]]
        scope = "" if parts[1] == "probe" else f", {SCOPES.get(parts[3], parts[3])}"
        return f"{CELL_LABELS[cell]} vs {family} {ENCODERS.get(parts[2], parts[2])}{scope}"
    return representation


def dataset_label(dataset_id: str, names=None) -> str:
    if names and dataset_id in names:
        return names[dataset_id]
    return " ".join(dataset_id.split("_")[2:]).title()


def _present(summary, rows):
    found = set(summary.representation) if summary is not None and not summary.empty else set()
    return [r for r in rows if r in found]


def _family_n(summary, family):
    data = summary[summary.task_family.eq(family) & summary.model_id.eq(PREDICTOR_MEAN)]
    return int(data.datasets.max()) if not data.empty else 0


def _family_title(ax, summary, family):
    name, unit = FAMILY_UNITS[family]
    n = _family_n(summary, family)
    ax.set_title(f"{name} ({unit}, {n} datasets)", fontsize=8)


def _legend(fig, extra=(), models=PREDICTORS, ncol=5):
    handles = [Line2D([], [], ls="none", marker=style.PREDICTOR_MARKERS[m], ms=3.6, mfc="none",
                      mec=style.PREDICTOR_INK, mew=0.7, label=style.PREDICTOR_LABELS.get(m, m))
               for m in models]
    fig.legend(handles=[*extra, *handles], loc="outside lower center", ncol=ncol, frameon=False,
               fontsize=7, handletextpad=0.3, columnspacing=0.9)


def _mean_handles(color=style.PREDICTOR_INK):
    return [Line2D([], [], ls="-", lw=0.9, marker=MEAN_MARKER, ms=3.8, color=color,
                   label="Mean over predictors (95% CI)"),
            Line2D([], [], ls="none", marker="|", ms=6, mew=1.2, color=color,
                   label="Median over datasets")]


# ----------------------------------------------------------------------------- forests


def family_forest(summary, rows, *, name, caption, xlabel="{unit}", groups=None, colors=None,
                  annotations=None, annotation_title=None, labels=None):
    """Two panels (classification ΔAUC | regression ΔR²): per arm the predictor mean with its
    dataset-bootstrap interval and dataset median, and each predictor's equal-dataset mean as a
    hollow marker. ``groups`` = ((title, rows), ...) adds bold header rows."""
    if summary is None or summary.empty:
        return
    if groups is None:
        groups = ((None, rows),)
    layout_rows = []  # (label, representation or None)
    for title, members in groups:
        members = _present(summary, members)
        if not members:
            continue
        if title:
            layout_rows.append((title, None))
        layout_rows += [((labels or {}).get(r) or label(r), r) for r in members]
    if not any(r for _, r in layout_rows):
        return
    colors = colors or {}
    height = min(style.MAX_HEIGHT, 1.25 + 0.2 * len(layout_rows))
    fig, axes = plt.subplots(1, 2, figsize=(style.WIDTH_FULL, height), sharey=True, layout="constrained")
    y = np.arange(len(layout_rows))
    for ax, family in zip(axes, FAMILIES):
        data = summary[summary.task_family.eq(family)]
        for position, (_, rep) in zip(y, layout_rows):
            if rep is None:
                continue
            rows_rep = data[data.representation.eq(rep)]
            for model, frame in rows_rep[rows_rep.model_id.ne(PREDICTOR_MEAN)].groupby("model_id"):
                ax.plot(frame.mean_raw_gain, [position] * len(frame), ls="none",
                        marker=style.PREDICTOR_MARKERS.get(model, "o"), ms=3.2, mfc="none",
                        mec=style.PREDICTOR_INK, mew=0.5, alpha=0.55, zorder=2)
            mean = rows_rep[rows_rep.model_id.eq(PREDICTOR_MEAN)]
            if mean.empty:
                continue
            m = mean.iloc[0]
            color = colors.get(rep, style.PREDICTOR_INK)
            ax.errorbar(m.mean_raw_gain, position, xerr=[[m.mean_raw_gain - m.raw_ci_low],
                                                         [m.raw_ci_high - m.mean_raw_gain]],
                        fmt=MEAN_MARKER, ms=4.2, color=color, mec="black", mew=0.4, lw=1.0, capsize=0,
                        ecolor=style.PREDICTOR_INK, zorder=4)
            ax.plot(m.median_raw_gain, position, ls="none", marker="|", ms=6, mew=1.2,
                    color=style.PREDICTOR_INK, zorder=5)
        ax.axvline(0, color=style.MEDIAN_COLOR, lw=0.7, zorder=1)
        layout.light_grid(ax)
        _family_title(ax, summary, family)
        ax.set_xlabel(xlabel.format(unit=FAMILY_UNITS[family][1]), fontsize=7.5)
        ax.tick_params(axis="x", labelsize=7)
    axes[0].set_yticks(y, [text for text, _ in layout_rows], fontsize=7)
    for tick, (_, rep) in zip(axes[0].get_yticklabels(), layout_rows):
        if rep is None:
            tick.set_fontweight("bold")
    axes[0].set_ylim(len(layout_rows) - 0.5, -0.5)
    axes[0].tick_params(axis="y", length=0)
    if annotations:
        for position, (_, rep) in zip(y, layout_rows):
            if rep in annotations:
                axes[1].annotate(annotations[rep], (1.02, position), xycoords=("axes fraction", "data"),
                                 va="center", ha="left", fontsize=7, color=style.MEDIAN_COLOR,
                                 annotation_clip=False)
        if annotation_title:
            axes[1].annotate(annotation_title, (1.02, 1.0), xycoords="axes fraction", va="bottom",
                             ha="left", fontsize=7, fontweight="bold", color=style.MEDIAN_COLOR)
    models = [m for m in PREDICTORS if m in set(summary.model_id)]
    _legend(fig, _mean_handles(), models=models)
    yield fig, name, caption


def _forest_caption(what, difference=False):
    quantity = ("the fold-paired difference A minus B (positive favours A)" if difference
                else "the gain over the same predictor's no-text baseline")
    return (f"{what}. Per dataset, {quantity} is the five-fold mean in the headline metric; {RAW_TEXT}, "
            "shown in separate panels. Filled diamond: equal-dataset mean of the per-dataset mean "
            f"over the seven predictors, with its {CI_TEXT}; vertical tick: median over datasets; "
            "hollow markers: each predictor's equal-dataset mean (shape = predictor). Only datasets "
            "with five finite folds for every predictor enter the predictor mean; n per panel in its title.")


def c1_gains(summary, *, view, name):
    """Supporting C1 forest: every hidden-state arm of one view (and C1b) against no text."""
    rows = [f"decision/{view}/strands"]
    rows += [f"hidden/{view}/{p}/L{layer}" for p in POSITIONS for layer in (12, 24)]
    rows += [f"hidden+decision/{view}/answer/L24/strands"]
    scope = "one PCA32 block per text column" if view == "per_column" else "the joint request, PCA64"
    yield from family_forest(
        summary, rows, name=name, xlabel="Gain over no text ({unit})",
        colors={f"decision/{view}/strands": style.STRANDS_COLORS["zero_shot"]},
        caption=_forest_caption(
            f"Strands hidden states ({scope}; PCA fitted on outer-training rows) added to the "
            "non-text columns, with Strands' own probabilities of the same view (B1 arm) as reference"))


def c1_contrasts(summary, *, name):
    """C1 fold-paired comparisons in one grouped forest (both views)."""
    if summary is None or summary.empty:
        return
    names = sorted(set(summary.representation))
    groups = []
    for title, prefixes in C1_GROUPS:
        members = [n for n in names if n.split("/")[0] in prefixes]
        order = {p: i for i, p in enumerate(prefixes)}
        members.sort(key=lambda n: (order[n.split("/")[0]], n.split("/")[1] != "per_column", n))
        groups.append((title, members))
    yield from family_forest(
        summary, [], groups=groups, name=name, xlabel="Difference A − B ({unit})",
        caption=_forest_caption(
            "C1 comparisons, per column and joint: the <answer> layer-24 vector against and together "
            "with Strands' own probabilities, each later prompt position against the earlier one at "
            "layers 12 and 24, and layer 24 against layer 12 at each position", difference=True))


def c2_gains(summary, *, name):
    """Supporting C2 forest: zero-shot and the three trained cells against no text."""
    rows = list(CELLS.values())
    yield from family_forest(
        summary, rows, name=name, xlabel="Gain over no text ({unit})",
        colors={rep: CELL_COLORS[cell] for cell, rep in CELLS.items()},
        caption=_forest_caption(
            "Strands joint-question probabilities without labels (zero-shot, the B1 Strands joint arm) and "
            "after cross-fitted training on the dataset's labels (three parts per outer training fold)"))


def c2_cells(summary, *, name):
    rows = list(C2_CELL_CONTRASTS)
    yield from family_forest(
        summary, rows, name=name, xlabel="Difference A − B ({unit})",
        caption=_forest_caption("The four C2 comparisons between the 2 × 2 cells (joint question)",
                                difference=True))


def c2_context(summary, *, name):
    """Each C2 cell against TAR (two PCA recipes) and the label-trained classifier (three encoders)."""
    if summary is None or summary.empty:
        return
    names = set(summary.representation)
    context = [*TAR, *PROBES]
    groups, labels = [], {}
    for cell in CELLS:
        members = {f"{cell}_vs_{c.split('/')[0]}/{c}": c for c in context}
        labels.update({m: "minus " + label(c).replace("Label-trained classifier", "classifier")
                       for m, c in members.items()})
        groups.append((CELL_LABELS[cell], [m for m in members if m in names]))
    yield from family_forest(
        summary, [], groups=groups, labels=labels, name=name, xlabel="Difference A − B ({unit})",
        caption=_forest_caption(
            "Each C2 cell (zero-shot, head only, LoRA only, LoRA + head) minus each label-using "
            "reference: TAR E5-small with per-column PCA32 or joint PCA64, and the B2 label-trained "
            "classifier on the joint request with E5-small, E5-large or Qwen3-Emb-0.6B (context comparisons)",
            difference=True))


# ------------------------------------------------------------------- C1: position path


def position_path(summary, *, name):
    """C1 summary: gain along the prompt (end of text -> end of question -> <answer>) at layers
    12 and 24, per view and task family, with Strands' own probabilities as reference."""
    if summary is None or summary.empty:
        return
    reps = [f"hidden/{v}/{p}/L{layer}" for v in VIEW_LABELS for p in POSITIONS for layer in (12, 24)]
    if not _present(summary, reps):
        return
    fig, axes = plt.subplots(2, 2, figsize=(style.WIDTH_FULL, 4.3), sharex=True, layout="constrained")
    x = np.arange(len(POSITIONS))
    layer_style = {12: dict(ls="--", marker="o", mfc="white"), 24: dict(ls="-", marker="o", mfc=None)}
    color = style.STRANDS_COLORS["hidden"]
    reference_color = style.STRANDS_COLORS["zero_shot"]
    for row, view in enumerate(VIEW_LABELS):
        for col, family in enumerate(FAMILIES):
            ax = axes[row, col]
            data = summary[summary.task_family.eq(family)]
            drawn = False
            for layer in (12, 24):
                labels = [f"hidden/{view}/{p}/L{layer}" for p in POSITIONS]
                mean = data[data.model_id.eq(PREDICTOR_MEAN)].set_index("representation").reindex(labels)
                per = data[data.model_id.ne(PREDICTOR_MEAN) & data.representation.isin(labels)]
                band = per.groupby("representation").mean_raw_gain.agg(["min", "max"]).reindex(labels)
                if mean.mean_raw_gain.notna().any():
                    drawn = True
                    ax.fill_between(x, band["min"], band["max"], color=color, alpha=0.13 if layer == 24 else 0.07,
                                    lw=0, zorder=1)
                    kw = layer_style[layer]
                    ax.plot(x, mean.mean_raw_gain, color=color, lw=1.2, ms=3.6, mec=color,
                            mfc=kw["mfc"] or color, ls=kw["ls"], marker=kw["marker"], zorder=3)
            reference = data[data.model_id.eq(PREDICTOR_MEAN) & data.representation.eq(f"decision/{view}/strands")]
            if not reference.empty:
                ax.axhline(reference.mean_raw_gain.iloc[0], color=reference_color, lw=1.0, ls=":", zorder=2)
            plus = data[data.model_id.eq(PREDICTOR_MEAN)
                        & data.representation.eq(f"hidden+decision/{view}/answer/L24/strands")]
            if not plus.empty:
                ax.plot(x[-1] + 0.12, plus.mean_raw_gain.iloc[0], ls="none", marker="*", ms=6.5,
                        color=reference_color, mec="black", mew=0.3, zorder=4)
            if not drawn:
                ax.text(0.5, 0.5, "not in export", transform=ax.transAxes, ha="center", va="center",
                        fontsize=7, color=style.MEDIAN_COLOR)
            ax.axhline(0, color=style.MEDIAN_COLOR, lw=0.6, zorder=0)
            layout.light_grid(ax, "y")
            ax.tick_params(labelsize=7)
            ax.set_xlim(-0.25, len(POSITIONS) - 0.6)
            if row == 0:
                _family_title(ax, summary, family)
            ax.set_ylabel(f"{VIEW_LABELS[view].capitalize()} view: gain ({FAMILY_UNITS[family][1]})",
                          fontsize=7.5)
    for ax in axes[1]:
        ax.set_xticks(x, [POSITION_LABELS[p] for p in POSITIONS], fontsize=7)
    handles = [
        Line2D([], [], color=color, lw=1.2, marker="o", ms=3.6, label="Layer 24 (head input)"),
        Line2D([], [], color=color, lw=1.2, ls="--", marker="o", ms=3.6, mfc="white", label="Layer 12"),
        plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.2, lw=0, label="Range over predictors"),
        Line2D([], [], color=reference_color, lw=1.0, ls=":", label="Strands' own probabilities (B1)"),
        Line2D([], [], ls="none", marker="*", ms=6.5, color=reference_color, mec="black", mew=0.3,
               label="<answer> L24 + probabilities"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=7)
    caption = (
        "Strands hidden states added to the non-text columns, read at three prompt positions (last token "
        "of the state text, last token of the question, the <answer> token) and two layers (12 and 24). "
        "Rows: per text column (PCA32 per column) and joint request (PCA64); columns: task families "
        f"({RAW_TEXT}). Lines: equal-dataset mean over datasets of the per-dataset mean gain over the seven "
        "predictors (five-fold mean, gain over the same predictor's no-text baseline); shaded band: range "
        "of the seven predictors' equal-dataset means; dotted line: Strands' own probabilities of the same "
        "view (B1 arm, same folds); star: the <answer> layer-24 vector together with those probabilities "
        "(C1b). Intervals are in the companion comparison forest; n per panel in its title.")
    yield fig, name, caption


def c1_dataset_scatter(gains, *, name, names=None):
    """Per dataset: gain of the <answer> L24 vector against the gain of Strands' probabilities."""
    if gains is None or gains.empty:
        return
    data = gains[gains.model_id.eq(PREDICTOR_MEAN)]
    pairs = []
    for view in VIEW_LABELS:
        x = data[data.representation.eq(f"decision/{view}/strands")].set_index("dataset_id")
        y = data[data.representation.eq(f"hidden/{view}/answer/L24")].set_index("dataset_id")
        both = x.index.intersection(y.index)
        if len(both):
            pairs.append((view, x.loc[both], y.loc[both]))
    if not pairs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(style.WIDTH_FULL, 3.0), layout="constrained")
    markers = {"binary_classification": "o", "multiclass_classification": "s", "regression": "D"}
    for ax, family in zip(axes, FAMILIES):
        values = []
        for view, x, y in pairs:
            keep = x.task_type.map(lambda t: "regression" if t == "regression" else "classification").eq(family)
            for dataset in x.index[keep]:
                ax.plot(x.loc[dataset, "raw_gain"], y.loc[dataset, "raw_gain"], ls="none",
                        marker=markers[x.loc[dataset, "task_type"]], ms=4, color=style.MODE_COLORS[view],
                        mec="white", mew=0.3, alpha=0.9)
            values += [*x.loc[keep, "raw_gain"], *y.loc[keep, "raw_gain"]]
        name_, unit = FAMILY_UNITS[family]
        if values:
            low, high = min(values + [0]), max(values + [0])
            pad = 0.06 * (high - low or 1)
            lims = (low - pad, high + pad)
            ax.plot(lims, lims, color=style.MEDIAN_COLOR, lw=0.7, ls="--", zorder=0)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect("equal", adjustable="box")
        n = len({d for _, x, _ in pairs for d, t in x.task_type.items()
                 if ("regression" if t == "regression" else "classification") == family})
        ax.set_title(f"{name_} ({unit}, {n} datasets)", fontsize=8)
        ax.axhline(0, color=style.GRID_COLOR, lw=0.6, zorder=0)
        ax.axvline(0, color=style.GRID_COLOR, lw=0.6, zorder=0)
        ax.set_xlabel(f"Strands probabilities, gain over no text ({unit})", fontsize=7.5)
        ax.set_ylabel(f"<answer> L24 vector, gain ({unit})", fontsize=7.5)
        ax.tick_params(labelsize=7)
    handles = [Line2D([], [], ls="none", marker="o", ms=4, color=style.MODE_COLORS[v],
                      label=f"{VIEW_LABELS[v].capitalize()} view") for v in VIEW_LABELS]
    handles += [Line2D([], [], ls="none", marker=markers[t], ms=4, color=style.MEDIAN_COLOR,
                       label=layout.TASK_LABELS[t]) for t in markers]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False, fontsize=7)
    caption = (
        "One point per dataset and view: the gain over the no-text baseline of the <answer> layer-24 "
        "hidden vector (y; PCA32 per column or PCA64 joint) against that of Strands' own probabilities of "
        "the same view (x; B1 arm), each the mean over the seven predictors of the five-fold mean gain. "
        "Dashed line: identity (points above favour the vector). Colour: view; shape: task type; "
        f"{RAW_TEXT}.")
    yield fig, name, caption


# ------------------------------------------------------------------- C2: 2 x 2 design


def factorial_plot(summary, *, name):
    """C2 summary: 2 x 2 interaction (head trained on x, LoRA as line), raw gain per task family,
    with TAR and the label-trained classifier as reference lines."""
    if summary is None or summary.empty:
        return
    mean = summary[summary.model_id.eq(PREDICTOR_MEAN)]
    if not set(CELLS.values()) & set(mean.representation):
        return
    fig, axes = plt.subplots(1, 2, figsize=(style.WIDTH_FULL, 3.0), layout="constrained")
    references = {TAR[0]: dict(color=style.TAR_COLOR, ls="-"), TAR[1]: dict(color=style.TAR_COLOR, ls="--"),
                  PROBES[0]: dict(color=style.PROBE_COLOR, ls="-.")}
    lines = {False: ("zero_shot", "head_only"), True: ("lora_only", "lora_head")}
    line_colors = {False: style.STRANDS_COLORS["zero_shot"], True: style.STRANDS_COLORS["lora_head"]}
    for ax, family in zip(axes, FAMILIES):
        data = mean[mean.task_family.eq(family)].set_index("representation")
        for lora, cells in lines.items():
            offset = 0.03 if lora else -0.03
            xs, ys = [], []
            for cell in cells:
                rep = CELLS[cell]
                if rep not in data.index:
                    continue
                row = data.loc[rep]
                x = FACTORIAL[cell][1] + offset
                ax.errorbar(x, row.mean_raw_gain, yerr=[[row.mean_raw_gain - row.raw_ci_low],
                                                         [row.raw_ci_high - row.mean_raw_gain]],
                            fmt="o", ms=4.8, color=CELL_COLORS[cell], mec="black", mew=0.4, lw=0.9,
                            ecolor=style.PREDICTOR_INK, capsize=0, zorder=4)
                ax.plot(x + 0.07 * (1 if lora else -1), row.median_raw_gain, ls="none", marker="_", ms=6,
                        mew=1.4, color=style.PREDICTOR_INK, zorder=4)
                xs.append(x)
                ys.append(row.mean_raw_gain)
            if len(xs) == 2:
                ax.plot(xs, ys, color=line_colors[lora], lw=1.1, ls="-" if lora else "--", zorder=3)
        for rep, kw in references.items():
            if rep in data.index:
                ax.axhline(data.loc[rep].mean_raw_gain, lw=0.9, zorder=2, **kw)
        ax.axhline(0, color=style.MEDIAN_COLOR, lw=0.6, zorder=1)
        ax.set_xticks([0, 1], ["Released head", "Head trained"], fontsize=7.5)
        ax.set_xlim(-0.4, 1.4)
        layout.light_grid(ax, "y")
        ax.tick_params(axis="y", labelsize=7)
        _family_title(ax, summary, family)
        ax.set_ylabel(f"Gain over no text ({FAMILY_UNITS[family][1]})", fontsize=7.5)
    handles = [Line2D([], [], color=line_colors[False], ls="--", lw=1.1, label="Released LoRA"),
               Line2D([], [], color=line_colors[True], ls="-", lw=1.1, label="LoRA trained")]
    handles += [Line2D([], [], ls="none", marker="o", ms=4.5, color=CELL_COLORS[c], mec="black", mew=0.4,
                       label=CELL_LABELS[c]) for c in CELLS]
    handles += [Line2D([], [], ls="none", marker="_", ms=6, mew=1.2, color=style.MEDIAN_COLOR,
                       label="Median over datasets")]
    handles += [Line2D([], [], lw=0.9, label=label(rep), **kw) for rep, kw in references.items()]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=7)
    caption = (
        "C2 2 × 2 design on the joint question: x = pointer head released or trained, line = LoRA adapter "
        "released (dashed) or trained (solid); the four points are zero-shot (B1 Strands joint arm), head "
        "only, LoRA only and LoRA + head (cross-fitted, three parts per outer training fold). Point: "
        "equal-dataset mean of the per-dataset mean over the seven predictors of the five-fold gain over "
        f"the no-text baseline, with its {CI_TEXT}; short dark bar beside it: median over datasets. Horizontal lines: "
        "TAR E5-small (per-column PCA32 solid, joint PCA64 dashed) and the label-trained classifier on "
        f"E5-small (joint, dash-dot), same aggregation. {RAW_TEXT[0].upper() + RAW_TEXT[1:]}; n per panel in its title.")
    yield fig, name, caption


def c2_datasets(gains, *, name, names=None):
    """Per dataset: the four C2 cells and the two label-using references, one row per dataset."""
    if gains is None or gains.empty:
        return
    data = gains[gains.model_id.eq(PREDICTOR_MEAN)]
    arms = {**{rep: (CELL_COLORS[c], "o", CELL_LABELS[c]) for c, rep in CELLS.items()},
            TAR[0]: (style.TAR_COLOR, "x", label(TAR[0])), PROBES[0]: (style.PROBE_COLOR, "+", label(PROBES[0]))}
    data = data[data.representation.isin(arms)]
    if data.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(style.WIDTH_FULL, 3.4), layout="constrained")
    for ax, family in zip(axes, FAMILIES):
        part = data[data.task_type.map(lambda t: "regression" if t == "regression" else "classification")
                    .eq(family)]
        datasets = layout.ordered(part[["dataset_id", "task_type"]].drop_duplicates())
        name_, unit = FAMILY_UNITS[family]
        ax.set_title(f"{name_} ({unit}, {len(datasets)} datasets)", fontsize=8)
        if datasets.empty:
            ax.text(0.5, 0.5, "not in export", transform=ax.transAxes, ha="center", va="center", fontsize=7)
            ax.set_yticks([])
            continue
        index = {d: i for i, d in enumerate(datasets.dataset_id)}
        for rep, (color, marker, _) in arms.items():
            rows = part[part.representation.eq(rep)]
            filled = marker in "o"
            ax.plot(rows.raw_gain, rows.dataset_id.map(index), ls="none", marker=marker, ms=4,
                    color=color, mec="black" if filled else color, mew=0.3 if filled else 1.0, alpha=0.9)
        ax.axvline(0, color=style.MEDIAN_COLOR, lw=0.6, zorder=0)
        layout.light_grid(ax)
        layout.dataset_axis(ax, [dataset_label(d, names) for d in datasets.dataset_id], list(datasets.task_type),
                            fontsize=7)
        ax.set_xlabel(f"Gain over no text ({unit})", fontsize=7.5)
        ax.tick_params(axis="x", labelsize=7)
    handles = [Line2D([], [], ls="none", marker=m, ms=4, color=c, mec="black" if m == "o" else c,
                      mew=0.3 if m == "o" else 1.0, label=text) for c, m, text in arms.values()]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=7)
    caption = (
        "One row per dataset (ordered by task; label colour = binary, multiclass or regression): gain over "
        "the no-text baseline of zero-shot Strands (B1 joint arm), the three trained C2 cells, TAR E5-small "
        "(per-column PCA32) and the label-trained classifier on E5-small (joint), each the mean over the "
        f"seven predictors of the five-fold mean gain. {RAW_TEXT[0].upper() + RAW_TEXT[1:]}.")
    yield fig, name, caption


# ---------------------------------------------------------------------- C synthesis


def strands_ladder(summary, *, name, gpu_hours=None):
    """C synthesis: from zero-shot probabilities to label-trained Strands, beside TAR and the
    label-trained classifier; raw gain per task family; training GPU hours in the margin."""
    rows = [ZERO_SHOT, "hidden/joint/answer/L24", CELLS["head_only"], CELLS["lora_only"], CELLS["lora_head"]]
    references = [*TAR, PROBES[0]]
    if summary is None or summary.empty or not _present(summary, rows):
        return
    groups = (("Strands without labels", rows[:2]), ("Strands trained on labels", rows[2:]),
              ("Label-using references", references))
    colors = {ZERO_SHOT: style.STRANDS_COLORS["zero_shot"], rows[1]: style.STRANDS_COLORS["hidden"],
              **{CELLS[c]: CELL_COLORS[c] for c in ("head_only", "lora_only", "lora_head")},
              TAR[0]: style.TAR_COLOR, TAR[1]: style.TAR_COLOR, PROBES[0]: style.PROBE_COLOR}
    annotations = {ZERO_SHOT: "no training", rows[1]: "no training"}
    for cell in ("head_only", "lora_only", "lora_head"):
        hours = (gpu_hours or {}).get(cell)
        annotations[CELLS[cell]] = f"{hours:,.1f}" if hours is not None else "not in export"
    labels = {ZERO_SHOT: "Zero-shot probabilities (B1)", rows[1]: "<answer> L24 hidden vector (C1)",
              **{CELLS[c]: f"{CELL_LABELS[c]} (C2)" for c in ("head_only", "lora_only", "lora_head")}}
    yield from family_forest(
        summary, [], groups=groups, colors=colors, labels=labels, name=name,
        xlabel="Gain over no text ({unit})", annotations=annotations, annotation_title="Training GPU-h",
        caption=_forest_caption(
            "Strands from zero-shot to label-trained, joint question: its own probabilities (B1 arm), the "
            "label-free <answer> layer-24 hidden vector (C1, PCA64), and the cross-fitted head-only, "
            "LoRA-only and LoRA + head cells (C2), beside TAR E5-small (per-column PCA32, joint PCA64) and "
            "the label-trained classifier on E5-small (joint). Right margin: total measured B200 GPU hours "
            "of the cell's cross-fitted training fits (sum over datasets, folds and parts; feature "
            "extraction excluded)"))


def render(save, generator):
    """Save (PDF + caption), display and close every figure of a generator."""
    from src.utils.notebook_display import display_local

    for fig, name, caption in generator:
        save(fig, name, caption=caption)
        display_local(fig)
        plt.close(fig)
