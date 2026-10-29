"""Request/feature design schematic and planned predictor-input widths; no model outputs."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src.visualize import layout, style


def predictor_width_figure(plan, summary):
    """Predictor input columns of every Experiment 1/2 arm, per dataset, on one log axis.

    Widths are identical across the twelve extractors and across the three encoders, so one
    representative of each is drawn. The figure shows planned columns, not fitted rank.
    """
    order = layout.ordered(summary[["dataset_id", "dataset", "task_type"]])
    rows = plan[plan.model.eq("catboost")]
    decision = rows[rows.phase.eq("experiment_1")]
    decision = decision[decision.provider.isna() | decision.provider.eq("jev")]
    embedding = rows[rows.phase.eq("experiment_2") & rows.encoder.eq("e5_small_v2")]
    width = {key: frame.set_index("dataset_id").predictor_columns
             for key, frame in decision.groupby("feature_set")}
    joint = embedding[embedding.pca_scope.eq("joint")].pivot(
        index="dataset_id", columns="pca_components", values="predictor_columns")
    separate = embedding[embedding.pca_scope.eq("per_column")].set_index("dataset_id") \
        .predictor_columns
    limit = 500
    fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, 4.0))
    y = np.arange(len(order))
    ids = order.dataset_id
    family = style.FAMILY_COLORS["experiment_2"]
    ax.hlines(y, joint.reindex(ids).min(axis=1), joint.reindex(ids).max(axis=1),
              color=family, linewidth=2.4, alpha=0.35, zorder=1)
    for components in joint.columns:
        ax.scatter(joint.reindex(ids)[components], y, marker="|", s=26, color=family, zorder=2)
    ax.scatter(separate.reindex(ids), y, marker="s", s=16, color=family, zorder=3)
    ax.scatter(width["non_text"].reindex(ids), y, s=16, color=style.QUANTITY_COLOR, zorder=4)
    offsets = {"non_text_plus_joint": "joint", "non_text_plus_per_column": "per_column",
               "non_text_plus_both": "both"}
    for feature_set, mode in offsets.items():
        ax.scatter(width[feature_set].reindex(ids), y, s=16, marker="D",
                   color=style.MODE_COLORS[mode], zorder=5, linewidths=0)
    ax.axvline(limit, color=style.WINDOW_COLOR, linestyle=(0, (3, 2)), linewidth=0.8)
    ax.text(limit * 1.05, 0.02, "Kumo recipe limit · 500", rotation=90, ha="left", va="bottom",
            fontsize=6, color=style.WINDOW_COLOR, transform=ax.get_xaxis_transform())
    ax.set_xscale("log")
    ax.set_xticks([1, 3, 10, 30, 100, 300, 1000], ["1", "3", "10", "30", "100", "300", "1,000"])
    ax.set_xlim(0.8, 1100)
    ax.set_xlabel("Predictor input columns (log scale)")
    layout.light_grid(ax)
    layout.dataset_axis(ax, list(order.dataset), order.task_type, fontsize=7)
    handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=4, color=style.QUANTITY_COLOR,
               label="Non-text"),
        *[Line2D([], [], linestyle="none", marker="D", markersize=3.5,
                 color=style.MODE_COLORS[m], label=f"+ {label}")
          for m, label in (("joint", "joint"), ("per_column", "per-column"), ("both", "both"))],
        Line2D([], [], linestyle="none", marker="s", markersize=4, color=family,
               label="+ embedding, per-column PCA32"),
        Line2D([], [], color=family, marker="|", markersize=6, linewidth=2.4, alpha=0.6,
               label="+ embedding, joint PCA 8–256"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncols=3, fontsize=7, frameon=False)
    return (
        fig,
        "predictor_input_widths",
        "Planned number of predictor input columns for each Experiment 1 and Experiment 2 arm. "
        "Augmented arms add one missing-text flag per text column to the non-text columns. "
        "Decision widths are the same for all twelve extractors (1 feature per request for "
        "binary and regression, K for multiclass) and embedding widths the same for all three "
        "encoders; ticks on the joint-PCA bar mark 8, 16, 32, 64, 128 and 256 components. Arms "
        "wider than 500 columns are excluded for Kumo rather than silently reduced.",
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
    fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, 4.3))
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
    ax.text(0.02, 0.528, "Never sent: non-text values, row targets, class frequencies",
            fontsize=6.5, color=style.DIRECTION_COLORS["higher"], transform=ax.transAxes)
    _arrow(ax, (0.41, 0.72), (0.455, 0.72))
    _box(ax, 0.46, 0.62, 0.15, 0.20, ink, fill=faint)
    ax.text(0.535, 0.72, "Frozen decision\nmodel\n(12 extractors,\nno generation)",
            ha="center", va="center", fontsize=7, transform=ax.transAxes)
    _arrow(ax, (0.615, 0.72), (0.66, 0.72))
    _box(ax, 0.665, 0.505, 0.33, 0.43, ink)
    ax.text(0.68, 0.905, "Features per request", weight="bold", fontsize=7, va="top",
            transform=ax.transAxes)
    outputs = [
        ("Binary", "1", "p(positive class)"),
        ("Multiclass", "K", "class probabilities,\nfixed class order"),
        ("Regression", "1", "expected direction;\nnine probabilities kept"),
    ]
    y = 0.82
    for task, width, body in outputs:
        key = {"Binary": "binary_classification", "Multiclass": "multiclass_classification",
               "Regression": "regression"}[task]
        ax.text(0.68, y, task, fontsize=7, va="top", color=style.TASK_COLORS[key],
                weight="bold", transform=ax.transAxes)
        ax.text(0.785, y, width, fontsize=7, va="top", weight="bold", transform=ax.transAxes)
        ax.text(0.81, y, body, fontsize=6.8, va="top", transform=ax.transAxes)
        y -= 0.075 if "\n" not in body else 0.11
    ax.text(0, 0.46, "b  Predictor input for a row with T = 2 text columns (f features per request)",
            weight="bold", fontsize=8, va="top", transform=ax.transAxes)
    arms = [
        ("Non-text baseline", [("non-text columns", 0.34, style.MISSING_COLOR)]),
        ("+ joint", [("non-text", 0.34, style.MISSING_COLOR), ("T flags", 0.06, faint),
                     ("joint · f", 0.09, joint)]),
        ("+ per-column", [("non-text", 0.34, style.MISSING_COLOR), ("T flags", 0.06, faint),
                          ("A · f", 0.09, per), ("B · f", 0.09, per)]),
        ("+ both", [("non-text", 0.34, style.MISSING_COLOR), ("T flags", 0.06, faint),
                    ("A · f", 0.09, per), ("B · f", 0.09, per), ("joint · f", 0.09, joint)]),
        ("+ frozen embedding", [("non-text", 0.34, style.MISSING_COLOR), ("T flags", 0.06, faint),
                                ("training-only PCA of E5/Qwen vectors", 0.27,
                                 style.FAMILY_COLORS["experiment_2"])]),
    ]
    for i, (name, blocks) in enumerate(arms):
        top = 0.385 - i * 0.075
        ax.text(0.16, top - 0.027, name, ha="right", va="center", fontsize=7,
                transform=ax.transAxes)
        x = 0.175
        for label, width, color in blocks:
            light = color in (faint, style.MISSING_COLOR)
            ax.add_patch(plt.Rectangle((x, top - 0.054), width - 0.004, 0.054, color=color,
                                       transform=ax.transAxes, linewidth=0))
            ax.text(x + width / 2, top - 0.027, label, ha="center", va="center", fontsize=6.3,
                    color=style.TEXT_COLOR if light else style.BACKGROUND_COLOR,
                    transform=ax.transAxes)
            x += width
        ax.text(0.99, top - 0.027, ("Experiment 2" if "embedding" in name else "Experiment 1"),
                ha="right", va="center", fontsize=6.3, color=ink, transform=ax.transAxes)
    return (
        fig,
        "request_and_feature_design",
        "(a) Contents of one provider-neutral decision request and the numeric features it "
        "returns. Every extractor receives the same task context, question, options and named "
        "text; adapters differ only in native prompt framing and readout. (b) Column blocks "
        "entering the six fixed predictors for a row with two text columns. Flags mark missing "
        "text and are shared by every augmented arm; f is 1 for binary and regression and K for "
        "multiclass. Identical requests are answered once and reused across rows. With a single "
        "text column the joint block equals the per-column block and is included once.",
    )


