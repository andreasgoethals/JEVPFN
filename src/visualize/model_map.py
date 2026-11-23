"""Group A synthesis: one map of the 17 decision models from the A1 and A2 summaries.

Marker = architecture family (`style.ARCHITECTURE_MARKERS`), area = verified parameter count
(open small marker when not stated), label = `style.DECISION_SHORT`. Nothing is redrawn from the
experiment notebooks: each point is one model's median or mean already shown there.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src.visualize import layout, style
from src.visualize.decision_diagnostics import short
from src.visualize.outputs import family_handles, family_marker

PANELS = (
    ("consensus_agreement", "classification_auc", "a  Typicality vs quality",
     "Same top answer as consensus (median)", "ROC-AUC, classification (median)"),
    ("mean_top_probability", "wording_excess_joint", "b  Sensitivity vs confidence",
     "Mean top probability (median)", "Wording: excess TV over replay (mean)"),
)
SIZE_REFERENCE = (3, 10, 30)


def _area(params):
    return 10 + 2.2 * params if np.isfinite(params) else 10.0


def _place_labels(ax, xs, ys, labels, colors, fontsize=6.5):
    """Greedy placement in offset points around each marker; keep the first non-overlapping spot.

    Call after the layout is frozen, so pixel boxes are final.
    """
    renderer = ax.figure.canvas.get_renderer()
    scale = ax.figure.dpi / 72
    points = ax.transData.transform(np.column_stack([xs, ys]))
    taken = [(x - 3 * scale, y - 3 * scale, x + 3 * scale, y + 3 * scale) for x, y in points]
    frame = ax.get_window_extent(renderer)
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]

    def clear(r):
        return all(r[2] < o[0] or r[0] > o[2] or r[3] < o[1] or r[1] > o[3] for o in taken)

    for xv, yv, label, color in zip(xs, ys, labels, colors):
        best = None
        for radius in (4, 7, 11, 16, 22):
            for dx, dy in directions:
                text = ax.annotate(label, (xv, yv), xytext=(dx * radius, dy * radius - 2), textcoords="offset points",
                                   fontsize=fontsize, color=color,
                                   ha="left" if dx > 0 else "right" if dx < 0 else "center")
                box = text.get_window_extent(renderer)
                rect = (box.x0, box.y0, box.x1, box.y1)
                inside = frame.x0 <= box.x0 and box.x1 <= frame.x1 and frame.y0 <= box.y0 and box.y1 <= frame.y1
                if inside and clear(rect):
                    best = text
                    taken.append(rect)
                    break
                text.remove()
            if best:
                break
        if best is None:
            ax.annotate(label, (xv, yv), xytext=(4, 2), textcoords="offset points", fontsize=fontsize, color=color)


def decision_model_map(table):
    """Two panels; ``table`` is `group_a.model_map_table` (one row per decision model)."""
    if table.empty:
        return
    usable = [p for p in PANELS if table[[p[0], p[1]]].notna().all(axis=1).any()]
    if not usable:
        return
    fig, axes = plt.subplots(1, len(usable), figsize=(style.WIDTH_FULL, 3.6), squeeze=False)
    axes = axes[0]
    placements = []
    for ax, (x, y, title, xlabel, ylabel) in zip(axes, usable):
        rows = table.dropna(subset=[x, y])
        for row in rows.itertuples():
            family = row.architecture_family if isinstance(row.architecture_family, str) else "not stated"
            marker = family_marker(family)
            size = float(row.parameters_billion) if np.isfinite(row.parameters_billion) else np.nan
            filled = np.isfinite(size)
            if marker == "x":
                ax.scatter(getattr(row, x), getattr(row, y), s=_area(size), marker=marker,
                           color=style.PREDICTOR_INK, linewidths=0.9, zorder=3)
                continue
            ax.scatter(getattr(row, x), getattr(row, y), s=_area(size), marker=marker,
                       facecolors=style.DECISION_POOL_COLOR if filled else "none",
                       edgecolors=style.PREDICTOR_INK, linewidths=0.8, zorder=3)
        ax.set_title(f"{title}, {len(rows)} models", loc="left", fontsize=8)
        ax.set_xlabel(xlabel, fontsize=7.5)
        ax.set_ylabel(ylabel, fontsize=7.5)
        ax.tick_params(labelsize=7)
        if y == "wording_excess_joint":
            ax.axhline(0, color=style.MEDIAN_COLOR, lw=0.7, zorder=1)
        if y == "classification_auc":
            ax.axhline(0.5, color=style.WINDOW_COLOR, lw=0.7, ls=":", zorder=1)
        layout.light_grid(ax, axis="both")
        ax.margins(0.15)
        placements.append((ax, rows))
    families = [f if isinstance(f, str) else "not stated" for f in table.architecture_family]
    handles = family_handles(families)
    handles += [Line2D([], [], ls="none", marker="o", ms=np.sqrt(_area(s)), color=style.DECISION_POOL_COLOR,
                       mec=style.PREDICTOR_INK, label=f"{s}B") for s in SIZE_REFERENCE]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False, fontsize=7,
               columnspacing=1.0, handletextpad=0.3)
    fig.canvas.draw()
    fig.set_layout_engine("none")  # Freeze positions so label boxes are final.
    for ax, (x, y, *_), in zip(axes, usable):
        rows = dict(placements)[ax]
        _place_labels(ax, rows[x].to_numpy(float), rows[y].to_numpy(float), [short(p) for p in rows.provider],
                      [style.DECISION_COLORS.get(p, style.TEXT_COLOR) for p in rows.provider])
    yield fig, "decision_model_map", (
        f"Map of {len(table)} decision models from the A1 and A2 summaries; marker shape = architecture family, "
        "marker area = verified parameter count (hollow small marker: not stated). (a) median over datasets of the "
        "share of distinct joint-request inputs whose most probable answer equals the leave-one-out consensus "
        "(x) against the median stand-alone ROC-AUC (binary) or macro one-vs-rest ROC-AUC (multiclass) over the "
        "nine classification datasets (y; dotted line 0.5); regression quality is rank-only and reported in the "
        "table. (b) median over datasets of the mean top probability on joint requests (x) against the "
        "equal-dataset mean excess total variation of the three rewordings over unchanged replays on joint "
        "requests (y; 0 = replay noise). Each point repeats a value shown in the A1 or A2 notebook."
    )
