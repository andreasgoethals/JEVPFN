"""Critical-difference (CD) diagrams in the style of Demsar (2006), at final A4 width.

Learners sit on a mean-rank axis (rank 1 = best on every dataset). The Nemenyi critical
difference is drawn as a bar but is DESCRIPTIVE here: mean-rank post-hoc tests depend on the
learner pool (Benavoli, Corani & Mangili 2016). The confirmatory annotation comes from the
two-sided Wilcoxon signed-rank tests with Holm adjustment of `inference.compare_learners`:

* all pairwise comparisons: thick bars join runs of adjacent learners in which NO pair has a
  Holm-significant difference (a statement about every joined pair, never about mean ranks);
* comparisons with one control: a filled marker flags a Holm-significant difference from the
  control, an open marker none.

Inputs are the ``(omnibus, table)`` pair of `compare_learners`; nothing is recomputed here.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import pandas as pd

from src.visualize import style

ENCODERS = {"e5_small_v2": "E5-small", "e5_large_v2": "E5-large", "qwen3_embedding_0_6b": "Qwen3-0.6B",
            "e5_small_v2_tar": "E5-small"}
SCOPES = {"per_column32": "per column 32", "joint64": "joint 64", "per_column": "per column", "joint": "joint"}


def _compact(name: str) -> str | None:
    """Short arm labels: a CD diagram holds up to 18 labels at full width."""
    parts = name.split("/")
    words = [w.replace("_", " ") for w in parts]
    if parts[0] == "frozen" and len(parts) == 3:
        return f"Frozen {ENCODERS.get(parts[1], parts[1])}, {SCOPES.get(parts[2], words[2])}"
    if parts[0] == "tar" and len(parts) == 3:
        return f"TAR {ENCODERS.get(parts[1], parts[1])}, {SCOPES.get(parts[2], words[2])}"
    if parts[0] == "probe" and len(parts) == 3:
        return f"Classifier {ENCODERS.get(parts[1], parts[1])}, {SCOPES.get(parts[2], words[2])}"
    if parts[0] == "decision" and len(parts) == 3:
        who = "median" if parts[2] == "median" else style.DECISION_SHORT.get(parts[2], parts[2])
        return f"Decision {words[1]}, {who}"
    if parts[0] == "decision+frozen" and len(parts) == 3:
        who = "median" if parts[2] == "median" else style.DECISION_SHORT.get(parts[2], parts[2])
        return f"Decision + E5-small, {who}"
    if parts[0] == "hidden" and len(parts) == 4:
        return f"Hidden {words[1]}, {words[2]}, {parts[3]}"
    if parts[0] == "hidden+decision" and len(parts) == 5:
        return f"Hidden + probabilities {words[1]}, {words[2]}, {parts[3]}"
    return None


def learner_label(name: str) -> str:
    """One readable label per learner, reusing the project's predictor and model names."""
    from src.visualize.group_b import _label

    fixed = {"baseline": "No-text baseline", "reference": "Canonical presentation",
             "always": "Always use", "never": "Never use"}
    if name in fixed:
        return fixed[name]
    if name in style.DECISION_SHORT:
        return style.DECISION_SHORT[name]
    if name in style.PREDICTOR_LABELS:
        return style.PREDICTOR_LABELS[name]
    if name.startswith("native/"):
        return style.PREDICTOR_LABELS.get(name.split("/", 1)[1], name)
    if name.startswith("rule/"):
        return f"LODO rule, {name.split('/', 1)[1].replace('_', '-')}"
    if name.startswith("adapted/joint/"):
        return {"head_only": "Head only", "lora_only": "LoRA only",
                "lora_head": "LoRA + head"}.get(name.rsplit("/", 1)[1], name)
    if name == "decision/joint/strands":
        return "Zero-shot Strands, joint"
    return _compact(name) or _label(name)


def control_of(table: pd.DataFrame) -> str | None:
    """The control learner of a `compare_learners` table, or None for all pairwise rows."""
    control = table.attrs.get("control")
    if control is not None:
        return str(control)
    if "comparison" in table and table.comparison.eq("versus_control").all() and table.versus.nunique() == 1:
        return str(table.versus.iloc[0])
    return None


def _significant(table: pd.DataFrame, alpha: float) -> set[frozenset]:
    rows = table[table.p_holm < alpha]
    return {frozenset((a, b)) for a, b in zip(rows.learner, rows.versus)}


def no_difference_groups(order, table: pd.DataFrame, alpha: float = 0.05) -> list[tuple[int, int]]:
    """Maximal runs (i, j) of learners, adjacent in `order`, with no Holm-significant pair.

    Every pair inside a run must be non-significant, so a bar never implies a transitive
    "equivalence" that a pairwise test did not show. Runs contained in another run are dropped.
    """
    order = list(order)
    tested = {frozenset((a, b)) for a, b in zip(table.learner, table.versus)}
    significant = _significant(table, alpha)
    runs = []
    for i in range(len(order)):
        j = i
        while j + 1 < len(order) and all(
                frozenset((order[m], order[j + 1])) in tested
                and frozenset((order[m], order[j + 1])) not in significant
                for m in range(i, j + 1)):
            j += 1
        if j > i:
            runs.append((i, j))
    return [r for r in runs if not any(o != r and o[0] <= r[0] and r[1] <= o[1] for o in runs)]


def cd_diagram(omnibus: dict, table: pd.DataFrame, *, labels=None, title: str | None = None,
               control: str | None = None) -> plt.Figure:
    """One CD diagram from `compare_learners` output; see the module docstring."""
    ranks = pd.Series(omnibus["mean_ranks"], dtype=float).sort_values(kind="stable")
    k = len(ranks)
    if k < 2:
        raise ValueError("A CD diagram needs at least two learners")
    alpha, cd = float(omnibus.get("alpha", 0.05)), float(omnibus["nemenyi_cd"])
    names = {learner: (labels or {}).get(learner) or learner_label(str(learner)) for learner in ranks.index}
    control = control or control_of(table)
    groups = [] if control else no_difference_groups(ranks.index, table, alpha)
    significant = _significant(table, alpha)
    left = math.ceil(k / 2)
    width, height = style.cd_diagram_size(left, len(groups), title=bool(title))
    fig = plt.figure(figsize=(width, height), layout="none")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(height, 0)  # y in inches from the top.
    ax.axis("off")

    # Leave room for the widest label as actually rendered (never let a label leave the page).
    renderer = fig.canvas.get_renderer()
    probe = [fig.text(0, 0, f"{n} (0.00)", fontsize=style.CD_FONT, fontweight="bold") for n in names.values()]
    widest = max(t.get_window_extent(renderer).width for t in probe) / fig.dpi
    for t in probe:
        t.remove()
    margin = max(0.12, (widest + 0.22) / width)
    if margin > 0.42:
        raise ValueError("Learner labels are too long for a full-width CD diagram; pass shorter labels")
    lo, hi = margin, 1 - margin

    def x(rank):
        return lo + (rank - 1) / (k - 1) * (hi - lo)

    ink = dict(color=style.CD_INK, lw=0.8, solid_capstyle="butt")
    top = 0.12
    if title:
        ax.text(0.01, top, title, ha="left", va="center", fontsize=plt.rcParams["axes.titlesize"],
                color=style.TEXT_COLOR)
        top += 0.22
    # Nemenyi CD bar (descriptive).
    y_cd = top + 0.14
    end = min(1 + cd, k)
    ax.plot([x(1), x(end)], [y_cd, y_cd], color=style.CD_BAR_COLOR, lw=1.4)
    for edge in (1, end):
        ax.plot([x(edge)] * 2, [y_cd - 0.04, y_cd + 0.04], color=style.CD_BAR_COLOR, lw=0.8)
    clipped = "" if 1 + cd <= k else f" > {k - 1} ranks"
    ax.text(x(1), y_cd - 0.07, f"Nemenyi CD = {cd:.2f}{clipped} (descriptive)", ha="left", va="bottom",
            fontsize=style.CD_NOTE_FONT, color=style.CD_BAR_COLOR)
    # Mean-rank axis, rank 1 (best) on the left.
    axis_y = y_cd + 0.36
    ax.plot([lo, hi], [axis_y, axis_y], **ink)
    every = 1 if (hi - lo) * width / (k - 1) >= 0.17 else 2  # Tick labels never collide.
    for rank in range(1, k + 1):
        ax.plot([x(rank)] * 2, [axis_y, axis_y - 0.05], **ink)
        if (rank - 1) % every == 0:
            ax.text(x(rank), axis_y - 0.07, str(rank), ha="center", va="bottom",
                    fontsize=style.CD_NOTE_FONT, color=style.CD_INK)
        if rank < k:
            ax.plot([x(rank + 0.5)] * 2, [axis_y, axis_y - 0.025], **ink)
    ax.text(lo - 0.012, axis_y, "Mean rank", ha="right", va="center", fontsize=style.CD_NOTE_FONT,
            color=style.CD_INK)
    # Bars joining learners with no Holm-significant Wilcoxon difference.
    for g, (i, j) in enumerate(groups):
        y = axis_y + 0.13 + g * style.CD_GROUP_STEP
        ax.plot([x(ranks.iloc[i]) - 0.006, x(ranks.iloc[j]) + 0.006], [y, y], color=style.CD_GROUP_COLOR,
                lw=2.2, solid_capstyle="round")
    start = axis_y + 0.24 + len(groups) * style.CD_GROUP_STEP
    order = list(ranks.index)
    sides = [(order[:left], lo - 0.012, "right"), (order[left:][::-1], hi + 0.012, "left")]
    for members, edge, align in sides:
        for row, learner in enumerate(members):
            rank, y = ranks[learner], start + row * style.CD_ROW_STEP
            ax.plot([x(rank), x(rank), edge], [axis_y, y, y], **ink)
            text = f"{names[learner]} ({rank:.2f})"
            ax.text(edge - 0.006 if align == "right" else edge + 0.006, y, text, ha=align, va="center",
                    fontsize=style.CD_FONT, color=style.TEXT_COLOR,
                    fontweight="bold" if learner == control else "normal")
            if control is not None:
                if learner == control:
                    ax.plot(x(rank), axis_y, marker="D", ms=4.5, color=style.CD_INK, zorder=3)
                else:
                    hit = frozenset((learner, control)) in significant
                    ax.plot(x(rank), axis_y, marker="o", ms=4.5, zorder=3,
                            markerfacecolor=style.CD_SIGNIFICANT_COLOR if hit else style.BACKGROUND_COLOR,
                            markeredgecolor=style.CD_SIGNIFICANT_COLOR if hit else style.CD_INK,
                            markeredgewidth=0.8)
    n = omnibus.get("datasets")
    if control is not None:
        note = (f"Filled circle: Holm-adjusted two-sided Wilcoxon p < {alpha:g} versus "
                f"{names[control]} (diamond); open: not significant.")
    else:
        note = (f"Bars join learners of which no pair differs (Holm-adjusted two-sided Wilcoxon "
                f"signed-rank, p < {alpha:g}).")
    note += f" N = {n} datasets." if n else ""
    ax.text(0.01, height - 0.1, note, ha="left", va="center", fontsize=style.CD_NOTE_FONT,
            color=style.CD_INK)
    return fig


def cd_figures(result: dict, *, labels=None):
    """(figure, name, caption) per compared group of a `learner_scores.run_application` record.

    Yields nothing unless the framework is approved, the export exists and the group's scores
    were complete, so notebooks show the waiting or stopped state instead of an empty figure.
    """
    if not result or result.get("status") != "compared":
        return
    for group in result["groups"]:
        if group.get("status") != "compared":
            continue
        omnibus, table = group["omnibus"], group["table"]
        fig = cd_diagram(omnibus, table, labels=labels)
        control = control_of(table)
        how = (f"filled circles mark a Holm-adjusted two-sided Wilcoxon signed-rank difference from "
               f"{learner_label(control)} (alpha {omnibus['alpha']:g})" if control else
               "bars join adjacent learners for which no pair has a Holm-adjusted two-sided "
               f"Wilcoxon signed-rank difference (alpha {omnibus['alpha']:g})")
        caption = (f"{result['application']}, {group['group']}: mean rank of {omnibus['learners']} "
                   f"learners over {omnibus['datasets']} datasets ({result['score_type'].replace('_', ' ')}"
                   f"; rank 1 is best). The bar above the axis is the descriptive Nemenyi critical "
                   f"difference; {how}.")
        yield fig, f"cd_{result['application']}_{group['group']}", caption
