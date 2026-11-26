"""Request/feature design schematic and planned predictor-input widths; no model outputs."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src.visualize import layout, style


def predictor_width_figure(plan, summary):
    """Predictor input columns of every B1 and B2 arm, per dataset, on one log axis.

    Widths are identical across the decision models and across the three encoders, so one
    representative of each is drawn. The figure shows planned columns, not fitted rank.
    """
    order = layout.ordered(summary[["dataset_id", "dataset", "task_type"]])
    rows = plan[plan.model.eq("catboost")]
    decision = rows[rows.phase.eq("B1")]
    columns = ["dataset_id", "feature_set", "predictor_columns"]
    width = decision[columns].drop_duplicates().pivot(
        index="dataset_id", columns="feature_set", values="predictor_columns")
    width = width.reindex(columns=["non_text", "non_text_plus_joint", "non_text_plus_per_column", "non_text_plus_both"])
    embedding = rows[rows.phase.eq("B2") & rows.encoder.eq("e5_small_v2")
                     & rows.feature_set.eq("non_text_plus_frozen_e5")]
    combined = rows.loc[rows.phase.eq("B2") & rows.feature_set.eq("non_text_plus_both_plus_frozen_e5"),
                        ["dataset_id", "predictor_columns"]].drop_duplicates().set_index("dataset_id").predictor_columns
    joint = embedding[embedding.pca_scope.eq("joint")].set_index("dataset_id").predictor_columns
    separate = embedding[embedding.pca_scope.eq("per_column")].set_index("dataset_id").predictor_columns
    fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, 4.0))
    y = np.arange(len(order))
    ids = order.dataset_id
    family = style.FAMILY_COLORS["B2"]
    ax.scatter(joint.reindex(ids), y - 0.22, marker="|", s=36, color=family, zorder=4)
    ax.scatter(separate.reindex(ids), y, marker="s", s=16, color=family, zorder=3)
    ax.scatter(combined.reindex(ids), y + 0.22, marker="s", s=16, facecolors="none",
               edgecolors=style.MODE_COLORS["both"], linewidths=0.9, zorder=3)
    ax.scatter(width["non_text"].reindex(ids), y, s=16, color=style.NON_TEXT_COLOR, zorder=4)
    for mode, offset in (("joint", -0.22), ("per_column", 0), ("both", 0.22)):
        ax.scatter(width[f"non_text_plus_{mode}"].reindex(ids), y + offset, s=16, marker="D",
                   color=style.MODE_COLORS[mode], zorder=5, linewidths=0)
    ax.set_xscale("log")
    ax.set_xticks([1, 3, 10, 30, 100, 300, 1000], ["1", "3", "10", "30", "100", "300", "1,000"])
    ax.set_xlim(0.8, 1100)
    ax.set_xlabel("Predictor input columns (log scale)")
    layout.light_grid(ax)
    layout.dataset_axis(ax, list(order.dataset), order.task_type, fontsize=7)
    handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=4, color=style.NON_TEXT_COLOR,
               label="No text (non-text + flags)"),
        *[Line2D([], [], linestyle="none", marker="D", markersize=3.5,
                 color=style.MODE_COLORS[m], label=f"+ {label}")
          for m, label in (("joint", "joint"), ("per_column", "per-column"), ("both", "both"))],
        Line2D([], [], linestyle="none", marker="s", markersize=4, markerfacecolor="none",
               color=style.MODE_COLORS["both"], label="+ both + E5 per-column PCA32"),
        Line2D([], [], linestyle="none", marker="s", markersize=4, color=family,
               label="+ embedding, per-column PCA32"),
        Line2D([], [], color=family, linestyle="none", marker="|", markersize=7,
               label="+ embedding, joint PCA64"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncols=4, fontsize=7, frameon=False)
    return (
        fig,
        "predictor_input_widths",
        "Planned number of predictor input columns for each B1 (decision features) and B2 (frozen "
        "embedding) arm, per dataset. Symbols are staggered vertically to show equal widths. "
        "Joint, per-column and combined decision inputs have the same width on single-text-column "
        "datasets. Every arm includes one missing-text flag per text column. "
        f"Decision widths are the same for the {decision.provider.nunique()} selected decision models (1 feature per request for binary, "
        "9 for regression, K for multiclass) and embedding widths the same for all three encoders. "
        "Per-column PCA adds 32 coordinates per text column and joint PCA adds 64 in total. The "
        "label-trained classifier has the corresponding joint or per-column decision width and TAR "
        "the corresponding embedding width. All arms are planned for seven predictors. Kumo's "
        "default 500-column limit is raised when an arm requires more input columns.",
    )


def _box(ax, x, y, w, h, color, *, fill=None, lw=0.9):
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                facecolor=fill or style.BACKGROUND_COLOR, edgecolor=color,
                                linewidth=lw, transform=ax.transAxes))


def _arrow(ax, start, end):
    ax.annotate("", xy=end, xytext=start, xycoords="axes fraction",
                arrowprops={"arrowstyle": "-|>", "color": style.TOTAL_COLOR, "lw": 0.9,
                            "mutation_scale": 8})


def input_design_figure():
    """What one request contains, what it returns, and how blocks enter each predictor arm."""
    fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, style.MAX_HEIGHT))
    ax.set_axis_off()
    ink, faint = style.TOTAL_COLOR, style.FAINT_COLOR
    per, joint = style.MODE_COLORS["per_column"], style.MODE_COLORS["joint"]
    ax.text(0, 1, "a  One decision request", weight="bold", fontsize=8, va="top",
            transform=ax.transAxes)
    _box(ax, 0.005, 0.505, 0.40, 0.43, ink)
    lines = [
        ("Task", "dataset, target meaning, observation unit"),
        ("Question", "fixed wording per task type"),
        ("Options", "documented class meanings, or the\nnine-level direction rubric (−4 … +4)"),
        ("Text", "one named column (per-column) or all\navailable columns together (joint)"),
    ]
    y = 0.905
    for head, body in lines:
        ax.text(0.02, y, head, weight="bold", fontsize=7, va="top", transform=ax.transAxes)
        ax.text(0.10, y, body, fontsize=7, va="top", transform=ax.transAxes, linespacing=1.15)
        y -= 0.07 if "\n" not in body else 0.105
    ax.text(0.02, 0.555, "Never sent: non-text values, row targets,\nclass frequencies",
            fontsize=7, va="top", linespacing=1.15,
            color=style.DIRECTION_COLORS["higher"], transform=ax.transAxes)
    _arrow(ax, (0.41, 0.72), (0.455, 0.72))
    _box(ax, 0.46, 0.62, 0.15, 0.20, ink, fill=faint)
    ax.text(0.535, 0.72, "Frozen decision\nmodel\n(non-generative\nprobabilities)",
            ha="center", va="center", fontsize=7, transform=ax.transAxes)
    _arrow(ax, (0.615, 0.72), (0.66, 0.72))
    _box(ax, 0.665, 0.505, 0.33, 0.43, ink)
    ax.text(0.68, 0.905, "Features per request", weight="bold", fontsize=7, va="top",
            transform=ax.transAxes)
    outputs = [
        ("Binary", "1", "p(positive class)"),
        ("Multiclass", "K", "class probabilities,\nfixed class order"),
        ("Regression", "9", "probabilities over\nordered evidence levels"),
    ]
    y = 0.82
    for task, width, body in outputs:
        key = {"Binary": "binary_classification", "Multiclass": "multiclass_classification",
               "Regression": "regression"}[task]
        ax.text(0.68, y, task, fontsize=7, va="top", color=style.TASK_COLORS[key],
                weight="bold", transform=ax.transAxes)
        ax.text(0.785, y, width, fontsize=7, va="top", weight="bold", transform=ax.transAxes)
        ax.text(0.81, y, body, fontsize=7, va="top", transform=ax.transAxes)
        y -= 0.075 if "\n" not in body else 0.11
    ax.text(0, 0.465, "b  Predictor input for a row with T = 2 text columns (f features per request)",
            weight="bold", fontsize=8, va="top", transform=ax.transAxes)
    frozen, probe, tar = style.FAMILY_COLORS["B2"], style.PROBE_COLOR, style.TAR_COLOR
    base = [("non-text", 0.30, style.MISSING_COLOR), ("T flags", 0.06, faint)]
    arms = [
        ("No text (reference)", [("non-text columns", 0.30, style.MISSING_COLOR), ("T flags", 0.06, faint)], "B1"),
        ("+ joint", [*base, ("joint · f", 0.08, joint)], "B1"),
        ("+ per-column", [*base, ("A · f", 0.08, per), ("B · f", 0.08, per)], "B1"),
        ("+ both", [*base, ("A · f", 0.08, per), ("B · f", 0.08, per), ("joint · f", 0.08, joint)], "B1"),
        ("+ both + E5 PCA", [*base, ("A · f", 0.08, per), ("B · f", 0.08, per), ("joint · f", 0.08, joint),
                             ("E5 PCA 2 × 32", 0.13, frozen)], "B2"),
        ("+ frozen embedding", [*base, ("PCA of frozen E5 / Qwen vectors", 0.26, frozen)], "B2"),
        ("+ label-trained classifier", [*base, ("A · f", 0.08, probe), ("B · f", 0.08, probe)], "B3"),
        ("+ TAR embedding", [*base, ("PCA of label-tuned E5 vectors", 0.26, tar)], "B3"),
    ]
    for i, (name, blocks, study) in enumerate(arms):
        top = 0.415 - i * 0.052
        ax.text(0.20, top - 0.021, name, ha="right", va="center", fontsize=7, transform=ax.transAxes)
        x = 0.215
        for label, width, color in blocks:
            light = color in (faint, style.MISSING_COLOR)
            ax.add_patch(plt.Rectangle((x, top - 0.042), width - 0.004, 0.042, color=color,
                                       transform=ax.transAxes, linewidth=0))
            ax.text(x + width / 2, top - 0.021, label, ha="center", va="center", fontsize=6.5,
                    color=style.TEXT_COLOR if light else style.BACKGROUND_COLOR,
                    transform=ax.transAxes)
            x += width
        ax.text(0.995, top - 0.021, study, ha="right", va="center", fontsize=7, color=ink,
                transform=ax.transAxes)
    return (
        fig,
        "request_and_feature_design",
        "(a) Contents of one provider-neutral decision request and the numeric features it "
        "returns. Every decision model receives the same task context, question, options and named "
        "text; adapters differ only in native prompt framing and readout. (b) Column blocks "
        "entering the seven fixed predictors for a row with two text columns (A, B), with the study "
        "that uses each arm. The value f is 1 for binary, 9 for regression (all nine level "
        "probabilities) and K for multiclass. Frozen-embedding arms add training-only PCA (32 "
        "components per text column, or 64 jointly) of E5-small, E5-large or Qwen3 vectors; the "
        "decision + E5 arm adds E5-small per-column PCA32 to the both view. The label-trained "
        "classifier replaces each decision block by a cross-fitted logistic regression on frozen "
        "vectors with the same width (per column shown; a joint variant exists); TAR fine-tunes "
        "E5-small on each outer-training fold's labels. Native-text systems (B5) and the Strands "
        "arms of Group C are not shown. Flags mark missing text and are shared by every arm. "
        "Identical requests are answered once and reused across rows; with a single text column "
        "the joint block equals the per-column block and is included once.",
    )
