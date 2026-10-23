"""Paper-story figures from measured receipts and the actual experiment plan."""

from __future__ import annotations

from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np

from src.visualize import style
from src.visualize.decision_comparison import LABELS

VARIANTS = {
    "paraphrase": "Paraphrase",
    "irrelevant_context": "Unrelated context",
    "embedded_instruction": "Inserted instruction",
    "option_order": "Reversed options",
}
ENCODER_LABELS = {"e5_small_v2": "E5-small-v2", "e5_large_v2": "E5-large-v2",
                  "qwen3_embedding_0_6b": "Qwen3-Embedding-0.6B"}


def representation_families_figure():
    fig, axes = plt.subplots(3, 1, figsize=style.METHOD_MAP_SIZE)
    rows = [
        ("Decision features", "Text + fixed task definition", "Frozen decision model",
         "Compact probabilities / direction", "Target meaning enters; row labels do not."),
        ("Embedding features", "Text + column name", "Frozen E5 / Qwen encoder",
         "Training-only PCA, then predictor", "Weights stay fixed; PCA uses training rows only."),
        ("Original-text systems", "Text + non-text features", "TabSTAR / ConTextTab / AG-MM",
         "Prediction with the complete system", "Representation and predictor change together."),
    ]
    for ax, row, color in zip(axes, rows, style.METHOD_COLORS):
        ax.set_axis_off()
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.text(0, 0.98, row[0], va="top", weight="bold", color=color)
        for x, label in zip((0.16, 0.5, 0.84), row[1:4]):
            ax.text(x, 0.53, fill(label, 24), ha="center", va="center", fontsize=8,
                    bbox={"boxstyle": "round,pad=0.4", "facecolor": style.BACKGROUND_COLOR,
                          "edgecolor": color})
        for start, end in ((0.32, 0.35), (0.65, 0.69)):
            ax.annotate("", xy=(end, 0.53), xytext=(start, 0.53),
                        arrowprops={"arrowstyle": "->", "color": color})
        ax.text(0.5, 0.04, row[4], ha="center", fontsize=7)
    return fig, "representation_families", (
        "Three representation families in Experiments 1–3. The first two augment the same six "
        "predictors and retain the original non-text features. Decision models receive documentary "
        "target meanings but no row targets; frozen encoders receive text without those task "
        "definitions. Original-text systems are separate system references. This diagram describes "
        "the interventions, not observed performance or supervised encoder fine-tuning."
    )


def study_grid_figure(grid):
    fig, ax = plt.subplots(figsize=style.STUDY_MAP_SIZE)
    ax.set_axis_off()
    counts = grid.set_index("experiment")
    lines = [
        ["0A", "Questions, storage, API/GPU contracts", "Before extraction"],
        ["0B", "Completed features, PCA, full-fold feasibility", "After extraction"],
        ["1", "11 decision models × 3 modes + baseline", "Six fixed predictors"],
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
    fig, ax = plt.subplots(figsize=style.EVIDENCE_MATRIX_SIZE)
    ax.set_axis_off()
    contents = [[LABELS.get(r.model_id, ENCODER_LABELS.get(r.model_id, r.model_id)),
                 *[labels[getattr(r, key)] for key in columns], r.complete_feature_datasets]
                for r in table.itertuples()]
    cells = ax.table(cellText=contents,
                     colLabels=["Runtime", *titles, "Complete datasets\nseparate · joint"],
                     colWidths=[0.27, 0.085, 0.09, 0.10, 0.105, 0.085, 0.265],
                     bbox=[0, 0.12, 1, 0.86], cellLoc="center")
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
    ax.text(0, 0.045, "Seen = receipt available locally; ? = no local evidence, not a model failure.",
            transform=ax.transAxes, fontsize=7)
    ax.set_title("Local evidence inventory · bounded tests ≠ full extraction", pad=10)
    return fig, "runtime_evidence", (
        "Eleven decision extractors and three frozen encoders, using locally available manifests "
        "at notebook execution. A staged receipt is not a rehash of weights. Full-bank counts "
        "require matching prepared-input identities; embedding fixtures are not full-bank decision "
        "counts. Seen API/GPU contracts are bounded historical observations, not final acceptance "
        "of the current code. Review denotes a documented pending human decision. Complete feature "
        "counts are given separately for per-column and joint artifacts; encoder artifacts use a "
        "separate store. Paths, hashes and recorded timestamps accompany the figure in the report."
    )


def prompt_sensitivity_figures(table):
    if table.empty:
        return
    from src.data.loaders import load_catalog

    names = {entry["id"]: entry["name"] for entry in load_catalog()["datasets"]}
    for (provider, identity), frame in table.groupby(["provider", "source_sha256"], sort=False):
        fig, axes = plt.subplots(2, 3, figsize=style.PROMPT_SENSITIVITY_SIZE,
                                  gridspec_kw={"height_ratios": [1.5, 1]})
        datasets = frame.dataset_id.drop_duplicates().tolist()
        for index, (ax, dataset) in enumerate(zip(axes[0], datasets)):
            part = frame[frame.dataset_id.eq(dataset) & frame.variant.ne("base")]
            y = [list(VARIANTS).index(v) for v in part.variant]
            ax.hlines(y, 0, part.total_variation_from_base, color=style.GRID_COLOR)
            for pos, row in zip(y, part.itertuples()):
                ax.scatter(row.total_variation_from_base, pos, marker="o" if row.same_top_option else "X",
                           color=style.MEDIAN_COLOR if row.same_top_option else style.P95_COLOR, s=25)
            ax.set_yticks(range(len(VARIANTS)), list(VARIANTS.values()), fontsize=7)
            ax.set_ylim(len(VARIANTS) - 0.5, -0.5)
            for pos, variant in enumerate(VARIANTS):
                if variant not in set(part.variant):
                    ax.text(0.04, pos, "n/a", fontsize=7, color=style.REMOVED_COLOR, va="center")
            ax.set(xlim=(-0.03, 1.03), xlabel="Total variation")
            task = ("Binary" if dataset.startswith("BIN_") else "Multiclass"
                    if dataset.startswith("MUL_") else "Regression")
            ax.set_title(task + "\n" + fill(names.get(dataset, dataset), 20), fontsize=8)
            if index:
                ax.tick_params(axis="y", labelleft=False)
        for ax in axes[1]:
            ax.set_axis_off()
        axes[1, 0].text(0, 0.82, "One selected real input per task\nCircle: same top option\nCross: changed top option",
                         fontsize=7, va="top", transform=axes[1, 0].transAxes)
        reg = frame[frame.expected_direction_change.notna() & frame.variant.ne("base")]
        ax = axes[1, 1]
        ax.set_axis_on()
        ax.axvline(0, color=style.GRID_COLOR)
        ax.scatter(reg.expected_direction_change, range(len(reg)), color=style.TOTAL_COLOR,
                   s=style.AUDIT_MARKER_SIZE)
        ax.set_yticks(range(len(reg)), [VARIANTS[v] for v in reg.variant], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Direction change (rubric units)", fontsize=7)
        ax.set_title("Regression only", fontsize=8)
        axes[1, 2].text(0, 0.82, "Selected probes only.\nNo target labels used.\nNo population flip rate.",
                         fontsize=7, va="top", transform=axes[1, 2].transAxes)
        fig.suptitle(f"{LABELS[provider]} · real-text sensitivity · {str(frame.created_utc.iloc[0])[:10]}")
        yield fig, f"prompt_sensitivity_{provider}_{identity[:8]}", (
            "Changes from each fixed real-text baseline under the listed interventions. Total "
            "variation compares aligned full probability vectors (0 identical, 1 disjoint); crosses "
            "mark changed top options. The lower panel reports regression expected-direction "
            "change in rubric units. Each task has one deterministically selected input, not a "
            "representative sample. Rows align across tasks; n/a marks inapplicable variants. "
            "Reversed options apply to multiclass only. Full dataset/row "
            f"and source identity are retained in the report; receipt SHA-256 starts {identity[:12]}. "
            "A changed output alone is not proof of an incorrect answer or semantic equivalence."
        )


def regression_information_figures(cases):
    if cases.empty:
        return
    for (provider, identity), part in cases.groupby(["provider", "source_sha256"], sort=False):
        fig, axes = plt.subplots(1, 2, figsize=style.REGRESSION_CASE_SIZE,
                                  gridspec_kw={"width_ratios": [1, 1.5]})
        variants = [p.split("__")[-1] for p in part.probe]
        for i, row in enumerate(part.itertuples()):
            axes[0].scatter(row.mean, row.variance, color=style.MEDIAN_COLOR,
                            marker=["o", "s", "^", "D"][i % 4], s=25,
                            label="Original" if variants[i] == "base" else VARIANTS[variants[i]])
        span = max(float(part["mean"].max() - part["mean"].min()), 0.1)
        axes[0].set(xlabel="Expected direction", ylabel="Variance (rubric units²)",
                    xlim=(part["mean"].min() - span * 0.3, part["mean"].max() + span * 0.3),
                    ylim=(0, max(part.variance.max() * 1.25, 0.1)))
        fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=2,
                   fontsize=7)
        image = axes[1].imshow(np.array(part.probabilities.tolist()), cmap=style.OVERVIEW_CMAP,
                               vmin=0, vmax=1, aspect="auto")
        axes[1].set_xticks(range(9), [f"{r:+d}" for r in range(-4, 5)], fontsize=7)
        axes[1].set_yticks(range(len(part)), ["Original" if v == "base" else VARIANTS[v] for v in variants], fontsize=7)
        axes[1].set_xlabel("Directional evidence level")
        for (i, j), value in np.ndenumerate(np.array(part.probabilities.tolist())):
            if value >= 0.01:
                axes[1].text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                             color=style.BACKGROUND_COLOR if value >= 0.6 else style.TEXT_COLOR)
        fig.colorbar(image, ax=axes[1], label="Probability", shrink=0.75)
        fig.suptitle(f"{LABELS[provider]} · retained regression distributions")
        yield fig, f"regression_information_{provider}_{identity[:8]}", (
            "Expected direction and directional variance (left) and the complete nine-level "
            "probability vectors (right) for the same real regression input and its diagnostic "
            "variants. Probabilities come from the saved receipt, not synthetic examples. "
            "The left axes zoom to these observed cases. A near-zero expectation does not "
            "determine the full distribution; neither axis "
            "is in target units. These cases do not establish a population uncertainty pattern "
            "or calibrated target probabilities."
        )


def measured_usage_figure(usage):
    if usage.empty:
        return
    from src.data.loaders import load_catalog

    catalog = {entry["id"]: entry["name"] for entry in load_catalog()["datasets"]}
    names = [catalog[r.dataset_id] for r in usage.itertuples()]
    fig, ax = plt.subplots(figsize=style.DASHBOARD_SIZE)
    y = np.arange(len(usage))
    bars = ax.barh(y, usage.reported_input_tokens / 1000, color=style.MEDIAN_COLOR)
    ax.set_yticks(y, [fill(n, 25) for n in names], fontsize=7)
    ax.invert_yaxis()
    ax.bar_label(bars, labels=[f"{n} jobs" for n in usage.stored_jobs], padding=3, fontsize=7)
    ax.set_xlim(0, usage.reported_input_tokens.max() / 1000 * 1.3)
    ax.set_xlabel("Reported input tokens in stored Jev jobs (thousands)")
    ax.set_title(f"Measured diagnostic usage · recorded charges ${usage.recorded_charges_usd.sum():.6f}")
    yield fig, "measured_jev_diagnostic_usage", (
        "Cumulative input tokens from local Jev extraction-job records, grouped by dataset; "
        "annotations count distinct stored jobs, including resumed jobs once. Reported charges "
        "are retained cumulative accounting, separate from reservations. Earlier ambiguous "
        "attempts may lack usage; synthetic/control receipts are outside this dataset-cache view. "
        "This deterministic pilot is not a representative full-extraction cost estimate."
    )
