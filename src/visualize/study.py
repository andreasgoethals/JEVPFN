"""Print-width coverage, native references and cost views for separate experiments."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src.data.loaders import load_catalog
from src.utils.config import load_yaml
from src.visualize import layout, style


def _names():
    return {r["id"]: r["name"] for r in load_catalog()["datasets"]}


def _ordered_datasets(dataset_ids):
    """Catalog datasets in the project-wide task order, restricted to those present."""
    semantics = load_yaml("dataset_semantics")["datasets"]
    names = _names()
    present = set(dataset_ids)
    rows = [(d, semantics[d]["task_type"]) for d in names if d in present]
    rank = {t: i for i, t in enumerate(layout.TASK_ORDER)}
    rows.sort(key=lambda r: rank[r[1]])
    return [d for d, _ in rows], [t for _, t in rows], names


def coverage_figures(table, title, *, feature_inventory=False):
    if table.empty or not table.completed.gt(0).any():
        # A page of identical 0/N cells adds no information to the coverage table.
        return
    datasets, tasks, names = _ordered_datasets(table.dataset_id)
    models = list(dict.fromkeys(table.model_id))
    fraction = np.full((len(datasets), len(models)), np.nan)
    labels = {}
    for row in table.itertuples():
        if row.dataset_id in datasets:
            i, j = datasets.index(row.dataset_id), models.index(row.model_id)
            fraction[i, j] = row.completed / row.planned if row.planned else np.nan
            labels[i, j] = f"{row.completed:g}/{row.planned:g}"
    fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 1.1 + 0.2 * len(datasets))))
    image = ax.imshow(fraction, vmin=0, vmax=1, cmap=style.OVERVIEW_CMAP, aspect="auto")
    for (i, j), label in labels.items():
        ax.text(j, i, label, ha="center", va="center", fontsize=7,
                color="white" if fraction[i, j] > 0.6 else style.TEXT_COLOR)
    layout.dataset_axis(ax, [names.get(d, d) for d in datasets], tasks, fontsize=7)
    names_by_model = style.DECISION_SHORT if feature_inventory else style.PREDICTOR_LABELS
    ax.set_xticks(range(len(models)), [names_by_model.get(m, m) for m in models],
                  fontsize=7, rotation=40 if len(models) > 6 else 0,
                  ha="left" if len(models) > 6 else "center")
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if feature_inventory:
        label = "Share of representation types present"
        detail = (
            "Each cell counts saved manifest types out of the two planned for a dataset and "
            "extractor (joint and separate text). This local inventory precedes 0B validation."
        )
    else:
        label = "Share of configurations complete"
        detail = (
            "Cells count complete over planned configurations per dataset and predictor. "
            "A configuration requires five defined outer folds of its headline metric. "
            "Zero indicates absence of qualifying saved evidence."
        )
    fig.colorbar(image, ax=ax, label=label, shrink=0.7)
    yield fig, "coverage", f"{title}. {detail}"


def burden_figures(table):
    """Representation width against fit time, one small panel per predictor."""
    if table.empty or "seconds.fit" not in table or "columns.representation" not in table:
        return
    table = table.dropna(subset=["seconds.fit", "columns.representation"])
    models = [m for m in style.PREDICTOR_LABELS if m in set(table.model_id)]
    if not models:
        return
    columns = 3
    rows = int(np.ceil(len(models) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 1.9 * rows + 0.5)),
                             squeeze=False, sharex=True, sharey=True)
    for ax, model in zip(axes.flat, models):
        group = table[table.model_id.eq(model)]
        for phase, part in group.groupby("phase"):
            ax.scatter(part["columns.representation"], part["seconds.fit"], s=8, alpha=0.5,
                       linewidths=0, color=style.FAMILY_COLORS[phase])
        ax.set_xscale("log")
        if group["seconds.fit"].gt(0).all():
            ax.set_yscale("log")
        ax.set_title(style.PREDICTOR_LABELS[model], fontsize=8)
        layout.light_grid(ax, axis="both")
    for ax in axes.flat[len(models):]:
        ax.set_axis_off()
    for ax in axes[-1]:
        ax.set_xlabel("input columns (log)", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("fit time (s, log)", fontsize=7)
    names = {"experiment_1": "Decision features", "experiment_2": "Frozen embedding",
             "experiment_3": "Supervised TAR", "experiment_4": "Native text"}
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=4, color=c, label=names[p])
               for p, c in style.FAMILY_COLORS.items() if p in set(table.phase)]
    fig.legend(handles=handles, loc="outside upper center", ncols=len(handles), fontsize=7,
               frameon=False)
    yield fig, "fit_burden", (
        "Mean outer-fold fit time against the number of input columns after representation and "
        "PCA (before predictor-specific category encoding), for every saved five-fold "
        "configuration; one point per dataset and recipe. Extraction cost is recorded separately "
        "and is not charged to each fit. Datasets and hardware remain identified in the tables."
    )


def qualification_figure(table, profiles=None):
    """0B evidence per predictor: contract, then passed/required full-fold profiles per arm family.

    ``table`` is `notebook_study.predictor_evidence`; ``profiles`` is
    `notebook_study.profile_evidence`. Green: all passed; amber: some; white: none locally;
    grey: not required. Returns ``(fig, name, caption)``.
    """
    from matplotlib.patches import Patch

    profiles = profiles if profiles is not None else pd_empty()
    order = ("no text", "decision", "decision+frozen", "frozen", "classifier", "TAR", "native",
             "column refit", "training size", "hidden", "hidden+decision", "adapted", "Kumo override")
    present = set(profiles.family) if len(profiles) else set()
    families = [f for f in order if f in present] + sorted(present.difference(order))
    models = list(table.model)
    columns = ["Contract", *families]
    cells = {}
    for row in table.itertuples():
        cells[row.model, "Contract"] = ("observed" if row.contract_passed else "unknown",
                                        "passed" if row.contract_passed else "pending")
    for row in profiles.itertuples():
        state = ("observed" if row.passed == row.required else "review" if row.passed else "unknown")
        cells[row.model, row.family] = (state, f"{int(row.passed)}/{int(row.required)}")
    fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 1.3 + 0.24 * len(models))))
    for i, model in enumerate(models):
        for j, column in enumerate(columns):
            state, text = cells.get((model, column), ("not_applicable", ""))
            ax.add_patch(plt.Rectangle((j - 0.46, i - 0.42), 0.92, 0.84, facecolor=style.EVIDENCE_COLORS[state]
                                       if state in ("observed", "review") else style.EVIDENCE_MARKS[state],
                                       edgecolor=style.GRID_COLOR, linewidth=0.6))
            ax.text(j, i, text, ha="center", va="center", fontsize=7)
    ax.set_xlim(-0.5, len(columns) - 0.5)
    ax.set_ylim(len(models) - 0.5, -0.5)
    ax.set_xticks(range(len(columns)), columns, fontsize=7, rotation=30, ha="left")
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(models)), [style.PREDICTOR_LABELS.get(m, m) for m in models], fontsize=7)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    legend = [Patch(facecolor=style.EVIDENCE_COLORS["observed"], label="all passed"),
              Patch(facecolor=style.EVIDENCE_COLORS["review"], label="some passed"),
              Patch(facecolor=style.EVIDENCE_MARKS["unknown"], edgecolor=style.GRID_COLOR, label="none locally"),
              Patch(facecolor=style.EVIDENCE_MARKS["not_applicable"], label="not required")]
    fig.legend(handles=legend, loc="outside lower center", ncols=4, fontsize=7, frameon=False)
    return fig, "predictor_qualification", (
        "Experiment 0B evidence per predictor, read from local records for the current evaluation "
        "identity. First column: the synthetic task-contract fit on VSC. Other columns: passed over "
        "required one-outer-fold profiles on real data per arm family, with the same runtime as the "
        "contract; the Kumo override column is the widest Kumo arm, whose 500-column default recipe "
        "is widened (owner-approved 23-11-2026). Profiles qualify runtimes; they are not results."
    )


def pd_empty():
    import pandas as pd

    return pd.DataFrame(columns=["model", "family", "required", "passed"])
