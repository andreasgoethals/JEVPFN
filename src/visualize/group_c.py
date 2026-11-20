"""Group C figures (C1 hidden states, C2 learning from labels), at final A4 width.

Every generator yields ``(figure, name, caption)`` and yields nothing for an empty table, so
the notebooks show a waiting state instead of an empty figure. Gains are "arm minus the same
predictor's no-text baseline" and contrasts "arm A minus arm B" on identical rows and folds, in
the fixed headline metric; datasets carry equal weight and intervals resample datasets.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.visualize import style
from src.visualize.group_b import MARKERS, PREDICTOR_COLORS, PREDICTORS, _legend

POSITION_LABELS = {"state_end": "end of text", "question_end": "end of question", "answer": "<answer>"}
CELL_LABELS = {"head_only": "Head only", "lora_only": "LoRA only", "lora_head": "LoRA + head",
               "zero_shot": "Zero-shot"}
COMPARISON_LABELS = {
    "answer_last_vs_strands_probabilities": "<answer> vector vs Strands probabilities",
    "answer_last_plus_probabilities_vs_probabilities": "<answer> vector + probabilities vs probabilities",
    "question_end_vs_state_end": "End of question vs end of text",
    "answer_vs_question_end": "<answer> vs end of question",
    "last_vs_middle_layer": "Layer 24 vs layer 12",
    "lora_vs_zero_shot": "LoRA only vs zero-shot",
    "head_only_vs_zero_shot": "Head only vs zero-shot",
    "lora_vs_head_only": "LoRA only vs head only",
    "lora_head_vs_lora_only": "LoRA + head vs LoRA only",
}
XLABEL = "{} (ROC-AUC, macro AUC or R²)"


def label(representation: str) -> str:
    """Readable name of a Group C arm or comparison label."""
    parts = representation.split("/")
    view = {"per_column": "per column", "joint": "joint"}
    if parts[0] == "hidden":
        return f"{POSITION_LABELS[parts[2]]}, layer {parts[3][1:]}, {view[parts[1]]}"
    if parts[0] == "hidden+decision":
        return f"<answer> L24 + probabilities, {view[parts[1]]}"
    if parts[0] == "decision":
        return f"Strands probabilities, {view[parts[1]]}"
    if parts[0] == "adapted":
        return CELL_LABELS[parts[2]]
    if parts[0] in COMPARISON_LABELS:
        names = {"per_column": "per column", "joint": "joint", "12": "L12", "24": "L24"}
        extra = [POSITION_LABELS.get(p, names.get(p, p)) for p in parts[1:]]
        return COMPARISON_LABELS[parts[0]] + (f" ({', '.join(extra)})" if extra else "")
    cell = parts[0].split("_vs_")[0]
    if cell in CELL_LABELS and len(parts) >= 4:  # context: <cell>_vs_<family>/<family>/<encoder>/<scope>
        encoders = {"e5_small_v2_tar": "E5-small", "e5_small_v2": "E5-small", "e5_large_v2": "E5-large",
                    "qwen3_embedding_0_6b": "Qwen3-Emb-0.6B"}
        family = {"tar": "TAR", "probe": "label-trained classifier"}[parts[1]]
        scope = {"per_column32": "per column PCA32", "joint64": "joint PCA64", "joint": "joint"}[parts[3]]
        return f"{CELL_LABELS[cell]} vs {family} {encoders[parts[2]]}, {scope}"
    return representation


def forest(summary, rows, *, name, caption, xlabel, share=False):
    """Equal-dataset means with dataset-bootstrap 95% intervals, one marker per predictor."""
    if summary is None or summary.empty:
        return
    data = summary[summary.representation.isin(rows)]
    if data.empty:
        return
    rows = [r for r in rows if r in set(data.representation)]
    models = [m for m in PREDICTORS if m in set(data.model_id)]
    column, low, high = (("mean_headroom_share", "share_ci_low", "share_ci_high") if share
                         else ("mean_raw_gain", "raw_ci_low", "raw_ci_high"))
    fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 1.2 + 0.36 * len(rows))),
                           layout="constrained")
    offsets = np.linspace(-0.3, 0.3, max(len(models), 1))
    for offset, model in zip(offsets, models):
        frame = data[data.model_id.eq(model)].set_index("representation").reindex(rows)
        y = np.arange(len(rows)) + offset
        ax.errorbar(frame[column], y, xerr=[frame[column] - frame[low], frame[high] - frame[column]],
                    fmt=MARKERS.get(model, "o"), ms=3.2, lw=0.8, color=PREDICTOR_COLORS.get(model),
                    capsize=0)
    ax.axvline(0, color=style.MEDIAN_COLOR, lw=0.8)
    ax.set_yticks(range(len(rows)), [label(r) for r in rows], fontsize=7.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.grid(axis="x", color=style.GRID_COLOR, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, fontsize=8)
    _legend(fig, models)
    yield fig, name, caption


def c1_gains(summary, *, view, name):
    rows = [f"hidden/{view}/{p}/L{layer}" for p in POSITION_LABELS for layer in (12, 24)]
    rows += [f"hidden+decision/{view}/answer/L24/strands"]
    yield from forest(summary, rows, name=name, xlabel=XLABEL.format("Gain over the no-text baseline"),
                      caption=(f"Strands hidden states ({'one PCA32 block per text column' if view == 'per_column' else 'the joint request, PCA64'}) "
                               "added to the non-text columns: equal-dataset mean gain over the same "
                               "predictor's no-text baseline with 95% dataset-bootstrap intervals "
                               "(2,000 draws), one marker per predictor. PCA is fitted on outer-training rows."))


def contrasts(summary, names, *, name, what):
    """Fold-paired A - B contrasts (positive favours A)."""
    rows = [r for r in names if summary is not None and not summary.empty and r in set(summary.representation)]
    yield from forest(summary, rows, name=name, xlabel=XLABEL.format("Difference A − B"),
                      caption=(f"{what}: equal-dataset mean of the fold-paired difference in the "
                               "headline metric (A minus B, positive favours A) with 95% dataset-bootstrap "
                               "intervals, one marker per predictor."))


def c2_gains(summary, *, name):
    rows = ["decision/joint/strands", "adapted/joint/head_only", "adapted/joint/lora_only",
            "adapted/joint/lora_head"]
    yield from forest(summary, rows, name=name, xlabel=XLABEL.format("Gain over the no-text baseline"),
                      caption=("Strands joint-question probabilities without labels (zero-shot, B1) and "
                               "after training on the dataset's labels (cross-fitted, three parts per "
                               "outer fold): equal-dataset mean gain over the same predictor's no-text "
                               "baseline with 95% dataset-bootstrap intervals, one marker per predictor."))
