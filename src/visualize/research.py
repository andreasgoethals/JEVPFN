"""Paper-story figures from measured receipts and the actual experiment plan."""

from __future__ import annotations

from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np

from src.utils.config import load_yaml
from src.visualize import layout, style
from src.visualize.decision_comparison import LABELS

VARIANTS = {
    "paraphrase": "Paraphrase",
    "irrelevant_context": "Unrelated context",
    "embedded_instruction": "Inserted instruction",
    "option_order": "Reversed options",
}
ENCODER_LABELS = {"e5_small_v2": "E5-small-v2", "e5_large_v2": "E5-large-v2",
                  "qwen3_embedding_0_6b": "Qwen3-Embedding-0.6B"}


def study_grid_figure(grid):
    fig, ax = plt.subplots(figsize=style.STUDY_MAP_SIZE)
    ax.set_axis_off()
    counts = grid.set_index("experiment")
    lines = [
        ["0A", "Questions, storage, API/GPU contracts", "Before extraction"],
        ["0B", "Completed features, PCA, full-fold feasibility", "After extraction"],
        ["1", f"{len(load_yaml('experiment_1/main')['candidate_extractors'])} decision models × 3 modes + baseline", "Six fixed predictors"],
        ["2", "3 encoders × 7 PCA recipes", "Six fixed predictors"],
        ["3", "3 original-text systems", "Separate references"],
        ["4", "Question / option / context sensitivity", "Label-blind diagnostics"],
        ["5", "Presence, statistics, shuffled / full outputs", "Separate controls"],
    ]
    table = ax.table(cellText=lines, colLabels=["Stage", "Controlled comparison / check", "Scope"],
                     colWidths=[0.09, 0.56, 0.35], cellLoc="left", colLoc="left",
                     bbox=[0, 0.40, 1, 0.55])
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor(style.GRID_COLOR)
        if row == 0:
            cell.set_facecolor(style.FAINT_COLOR)
            cell.get_text().set_weight("bold")
    for y, phase, label, color in zip((0.32, 0.23, 0.14),
                                     ("experiment_1", "experiment_2", "experiment_3"),
                                     ("Decision", "Embedding", "Original-text"), style.METHOD_COLORS):
        row = counts.loc[phase]
        ax.text(0.02, y, label, color=color, weight="bold", transform=ax.transAxes)
        ax.text(0.28, y, f"{row.configurations:,} configurations", transform=ax.transAxes)
        ax.text(0.69, y, f"{row.outer_fits:,} outer fits", transform=ax.transAxes)
    ax.text(0.5, 0.015, f"Main total: {grid.configurations.sum():,} configurations · "
            f"{grid.outer_fits.sum():,} outer + {grid.binary_inner_fits.sum():,} inner fits",
            ha="center", fontsize=8, transform=ax.transAxes)
    ax.set_title("Study map · planned comparisons on 18 datasets", pad=10)
    return fig, "study_map", (
        "The main executable grid and the roles of acceptance/diagnostic stages. Experiment 1 "
        "shares 108 non-text configurations across extractors; its three augmented modes are "
        "joint, per-column and both. Experiment 2 uses per-column PCA32 or joint "
        "PCA256/128/64/32/16/8 for each encoder. Each configuration has five outer folds; binary "
        "threshold selection adds three inner fits per outer fold. Counts precede feasibility "
        "exclusions and omit Experiment 5 controls and optional concatenations. Fits and "
        "configurations are planned workload units, not independent statistical replicates."
    )


def runtime_evidence_figure(table):
    columns = ["configured", "staged_receipt", "full_bank_tokens", "bounded_runtime", "semantic_review"]
    titles = ["Pinned\nconfig", "Staged\nreceipt", "Full-bank\ntokens", "API/GPU\ncontract", "Human\nreview"]
    labels = {"observed": "Seen", "review": "Review", "failed": "Failed", "unknown": "?",
              "not_applicable": "—"}
    fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 0.75 + 0.22 * len(table))))
    ax.set_axis_off()
    contents = [[LABELS.get(r.model_id, ENCODER_LABELS.get(r.model_id, r.model_id)),
                 *[labels[getattr(r, key)] for key in columns], r.complete_feature_datasets]
                for r in table.itertuples()]
    cells = ax.table(cellText=contents,
                     colLabels=["Runtime", *titles, "Complete datasets\nseparate · joint"],
                     colWidths=[0.27, 0.085, 0.09, 0.10, 0.105, 0.085, 0.265],
                     bbox=[0, 0.07, 1, 0.93], cellLoc="center")
    cells.auto_set_font_size(False)
    cells.set_fontsize(7)
    for (row, col), cell in cells.get_celld().items():
        cell.set_edgecolor(style.GRID_COLOR)
        if row == 0:
            cell.set_facecolor(style.FAINT_COLOR)
        elif 1 <= col <= 5:
            cell.set_facecolor(style.EVIDENCE_COLORS[table.iloc[row - 1][columns[col - 1]]])
        if col == 0:
            cell.get_text().set_ha("left")
    ax.text(0, 0.0, "Seen = receipt available locally; ? = no local evidence, not a model failure.",
            transform=ax.transAxes, fontsize=7)
    return fig, "runtime_evidence", (
        "Selected decision extractors and frozen encoders, using locally available manifests "
        "at notebook execution. A staged receipt is not a rehash of weights. Full-bank counts "
        "require matching prepared-input identities; embedding fixtures are not full-bank decision "
        "counts. Seen API/GPU contracts are bounded historical observations, not final acceptance "
        "of the current code. Review denotes a documented pending human decision. Complete feature "
        "counts are given separately for per-column and joint artifacts; encoder artifacts use a "
        "separate store. Paths, hashes and recorded timestamps accompany the figure in the report."
    )


def prompt_sensitivity_figures(table):
    """One row: distribution change per task, then the regression direction change."""
    if table.empty:
        return
    from matplotlib.lines import Line2D

    from src.data.loaders import load_catalog

    names = {entry["id"]: entry["name"] for entry in load_catalog()["datasets"]}
    tasks = {"BIN_": "binary_classification", "MUL_": "multiclass_classification",
             "REG_": "regression"}
    for (provider, identity), frame in table.groupby(["provider", "source_sha256"], sort=False):
        datasets = frame.dataset_id.drop_duplicates().tolist()
        fig, axes = plt.subplots(1, len(datasets) + 1, figsize=(style.WIDTH_FULL, 2.3),
                                 sharey=True)
        rows = list(VARIANTS)
        for ax, dataset in zip(axes, datasets):
            part = frame[frame.dataset_id.eq(dataset) & frame.variant.ne("base")]
            task = tasks[dataset[:4]]
            positions = [rows.index(v) for v in part.variant]
            ax.hlines(positions, 0, part.total_variation_from_base, color=style.GRID_COLOR,
                      linewidth=2)
            for pos, row in zip(positions, part.itertuples()):
                ax.scatter(row.total_variation_from_base, pos, s=24, zorder=3,
                           marker="o" if row.same_top_option else "X",
                           color=style.TASK_COLORS[task])
            for pos, variant in enumerate(rows):
                if variant not in set(part.variant):
                    ax.text(0.03, pos, "n/a", fontsize=6.5, color=style.REMOVED_COLOR,
                            va="center")
            ax.set(xlim=(-0.03, 1.03), xlabel="Total variation")
            ax.set_title(f"{fill(names.get(dataset, dataset), 22)}", fontsize=7.5,
                         color=style.TASK_COLORS[task])
            layout.light_grid(ax)
        ax = axes[-1]
        regression = frame[frame.expected_direction_change.notna() & frame.variant.ne("base")]
        positions = [rows.index(v) for v in regression.variant]
        ax.axvline(0, color=style.GRID_COLOR)
        ax.hlines(positions, 0, regression.expected_direction_change, color=style.GRID_COLOR,
                  linewidth=2)
        ax.scatter(regression.expected_direction_change, positions, s=24, zorder=3,
                   color=style.TASK_COLORS["regression"])
        bound = max(0.1, float(regression.expected_direction_change.abs().max()) * 1.3)
        ax.set(xlim=(-bound, bound), xlabel="Δ expected direction")
        ax.set_title("Regression direction", fontsize=7.5, color=style.TASK_COLORS["regression"])
        layout.light_grid(ax)
        axes[0].set_yticks(range(len(rows)), list(VARIANTS.values()), fontsize=7)
        axes[0].set_ylim(len(rows) - 0.5, -0.5)
        axes[0].tick_params(axis="y", length=0)
        handles = [Line2D([], [], linestyle="none", marker="o", markersize=4,
                          color=style.MEDIAN_COLOR, label="Same top option"),
                   Line2D([], [], linestyle="none", marker="X", markersize=5,
                          color=style.MEDIAN_COLOR, label="Changed top option")]
        fig.legend(handles=handles, loc="outside upper center", ncols=2, fontsize=7,
                   frameon=False)
        yield fig, f"prompt_sensitivity_{provider}_{identity[:8]}", (
            f"{LABELS[provider]} outputs on one deterministically selected real input per task, "
            f"recorded {str(frame.created_utc.iloc[0])[:10]}, under four interventions that should "
            "not change the answer. Total variation compares the aligned probability vector with "
            "the unmodified request (0 identical, 1 disjoint); option reversal applies only to "
            "multiclass and is mapped back to semantic classes. The right panel gives the change "
            "in regression expected direction (rubric units). Selected probes are not a population "
            f"estimate; receipt SHA-256 {identity[:12]}."
        )


def regression_information_figures(cases):
    """Full nine-level distributions, with their expectation and variance beside each row."""
    if cases.empty:
        return
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "probability", [style.BACKGROUND_COLOR, style.TASK_COLORS["regression"]])
    for (provider, identity), part in cases.groupby(["provider", "source_sha256"], sort=False):
        variants = [p.split("__")[-1] for p in part.probe]
        values = np.array(part.probabilities.tolist())
        fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, 0.75 + 0.36 * len(part)))
        ax.imshow(values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for (i, j), value in np.ndenumerate(values):
            if value >= 0.01:
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.5,
                        color=style.BACKGROUND_COLOR if value >= 0.6 else style.TEXT_COLOR)
        for i, row in enumerate(part.itertuples()):
            ax.text(9.1, i, f"E = {row.mean:+.2f}   Var = {row.variance:.2f}", va="center",
                    fontsize=6.5)
        ax.set_xticks(range(9), [f"{r:+d}" if r else "0" for r in range(-4, 5)], fontsize=7)
        ax.set_yticks(range(len(part)),
                      ["Original" if v == "base" else VARIANTS[v] for v in variants], fontsize=7)
        ax.set_xlim(-0.5, 8.5)
        ax.tick_params(length=0)
        ax.set_xlabel("Directional evidence level")
        yield fig, f"regression_information_{provider}_{identity[:8]}", (
            f"Complete nine-level {LABELS[provider]} distributions for the selected real "
            "regression input and its diagnostic variants, with the probability-weighted mean "
            "(E, the main feature) and variance in rubric units. Probabilities below 0.01 are not "
            "printed. Levels are evidence strengths, not target units; these cases do not "
            "establish a population pattern or calibration."
        )


def pilot_output_figures(table):
    """Label-blind pilot outputs: the spread of each main feature within every dataset."""
    if table.empty:
        return
    from matplotlib.lines import Line2D

    tasks = [t for t in layout.TASK_ORDER if t in set(table.task_type)]
    providers = [p for p in LABELS if p in set(table.provider)]
    datasets = {t: list(dict.fromkeys(table.loc[table.task_type.eq(t), "dataset"]))
                for t in tasks}
    fig, axes = plt.subplots(len(tasks), 1, figsize=(style.WIDTH_FULL, 4.7),
                             gridspec_kw={"height_ratios": [len(datasets[t]) + 1.3 for t in tasks]})
    axes = np.atleast_1d(axes)
    spec = {
        "binary_classification": ((-0.02, 1.02), "Binary: probability that the text supports the positive class"),
        "multiclass_classification": ((-0.02, 1.02), "Multiclass: normalized entropy of the class probabilities (0 = one class)"),
        "regression": ((-4.1, 4.1), "Regression: expected direction (rubric levels −4 … +4)"),
    }
    offsets = dict(zip(providers, np.linspace(-0.22, 0.22, len(providers)) if len(providers) > 1
                       else [0.0]))
    for ax, task in zip(axes, tasks):
        names = datasets[task]
        part = table[table.task_type.eq(task)]
        for provider in providers:
            rows = part[part.provider.eq(provider)]
            y = rows.dataset.map({n: i for i, n in enumerate(names)}) + offsets[provider]
            ax.scatter(rows.value, y, s=9 if provider == "jev" else 16, linewidths=0,
                       alpha=0.75, color=style.DECISION_COLORS[provider],
                       marker="o" if provider == "jev" else "D", zorder=3)
        ax.set_xlim(*spec[task][0])
        ax.set_xlabel(spec[task][1], fontsize=7)
        if task == "regression":
            ax.axvline(0, color=style.GRID_COLOR, linewidth=1)
        layout.light_grid(ax)
        layout.dataset_axis(ax, names, [task] * len(names), fontsize=7, separators=False)
    counts = table.groupby("provider").size()
    handles = [Line2D([], [], linestyle="none", marker="o" if p == "jev" else "D", markersize=4,
                      color=style.DECISION_COLORS[p], label=f"{LABELS[p]} (n = {counts[p]})")
               for p in providers]
    fig.legend(handles=handles, loc="outside upper center", ncols=len(handles), fontsize=7,
               frameon=False)
    yield fig, "pilot_decision_outputs", (
        "Main decision features returned for every verified label-blind pilot request stored "
        "locally, one point per unique request; providers are offset vertically. Requests are "
        "real prepared inputs (per-column and joint) drawn deterministically, not a random "
        "sample, and no target values are read. Responses that fail the production decoder "
        f"({table.attrs.get('invalid_responses', 0)} here) are excluded. Spread within a dataset "
        "shows whether a feature varies across texts; it is not evidence of predictive value."
    )
