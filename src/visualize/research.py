"""Paper-story figures from measured receipts and the actual experiment plan."""

from __future__ import annotations

from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np

from src.visualize import layout, style
from src.visualize.decision_comparison import LABELS

VARIANTS = {
    "paraphrase": "Paraphrase",
    "irrelevant_context": "Unrelated context",
    "embedded_instruction": "Inserted instruction",
    "option_order": "Reversed options",
}
ENCODER_LABELS = {"e5_small_v2": "E5-small-v2", "e5_large_v2": "E5-large-v2",
                  "qwen3_embedding_0_6b": "Qwen3-Embedding-0.6B", "e5_small_v2_tar": "TAR E5-small-v2"}


def study_grid_figure(grid):
    """Four comparisons at A4 print width, with measured plan counts."""
    counts = grid.set_index("experiment")
    fig, axes = plt.subplots(2, 2, figsize=(style.WIDTH_FULL, 5.0))
    rows = [
        ("experiment_1", "Decision features", "Selected extractors; per-column, joint or both.\nOne shared non-text reference; six predictors."),
        ("experiment_2", "Frozen embeddings", "Three pretrained encoders; six predictors.\nPer-column PCA32 and joint PCA64."),
        ("experiment_3", "Supervised TAR", "E5-small-v2 LoRA using training labels.\nSame six predictors and two PCA recipes."),
        ("experiment_4", "Native text systems", "TabSTAR v1.1, ConTextTab and AG-MM.\nText and structured inputs, end-to-end."),
    ]
    for ax, (phase, name, detail) in zip(axes.flat, rows):
        ax.set_axis_off()
        color, row = style.FAMILY_COLORS[phase], counts.loc[phase]
        ax.text(0, .9, f"Experiment {phase[-1]} · {name}", color=color, weight="bold", fontsize=9)
        ax.text(0, .68, detail, fontsize=8, va="top", linespacing=1.5)
        ax.text(0, .33, f"{row.configurations:,} configurations\n{row.outer_fits:,} outer predictor fits",
                fontsize=8, va="top", linespacing=1.5)
    fig.suptitle("Four comparisons on the same 18 datasets and five outer folds", fontsize=10)
    return fig, "study_map", (
        "Experiments 1–3 hold the downstream predictor fixed; Experiment 4 compares complete "
        "systems. PCA uses outer-training rows only; frozen embeddings additionally refit PCA "
        "inside binary threshold folds. TAR trains 90 shared fold-specific adapters, uses "
        "upstream early stopping and a fixed binary threshold, with no inner encoder refits. "
        "Counts precede capacity exclusions and exclude encoder preparation. Headline metrics "
        "are binary ROC-AUC, multiclass macro OvR ROC-AUC and regression R²."
    )


def _model_color(model_id):
    return style.DECISION_COLORS.get(model_id, style.EMBEDDING_COLORS.get(model_id, style.TEXT_COLOR))


def runtime_evidence_figure(table):
    """Acceptance gates (left) beside the measured B200 footprint of each runtime (right)."""
    from matplotlib.patches import Patch

    gates = [("bounded_runtime", "Contract"), ("known_answer", "Known answers"),
             ("robustness", "Robustness"), ("token_audit", "Token audit"),
             ("accepted", "Accepted")]
    names = [LABELS.get(m, ENCODER_LABELS.get(m, m)) for m in table.model_id]
    y = np.arange(len(table))
    fig, axes = plt.subplots(1, 4, figsize=(style.WIDTH_FULL, style.MAX_HEIGHT), sharey=True,
                             gridspec_kw={"width_ratios": [1.6, 1, 1, 1]})
    ax = axes[0]
    for j, (column, _) in enumerate(gates):
        ax.scatter(np.full(len(table), j), y, marker="s", s=58,
                   color=[style.EVIDENCE_MARKS[s] for s in table[column]],
                   edgecolors=style.GRID_COLOR, linewidths=0.6)
    ax.set_xticks(range(len(gates)), [label for _, label in gates], fontsize=7, rotation=90)
    ax.xaxis.tick_top()
    ax.set_xlim(-0.6, len(gates) - 0.4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_yticks(y, names, fontsize=7)
    ax.set_ylim(len(table) - 0.5, -0.5)
    ax.tick_params(length=0)
    for tick, model in zip(ax.get_yticklabels(), table.model_id):
        tick.set_color(_model_color(model))
    boundary = int(table.kind.ne("encoder").sum())
    panels = [("peak_gpu_gb", "Peak GPU memory (GB)"),
              ("load_seconds", "Model loading (s)"),
              ("contract_seconds", "Full contract run (s)")]
    for ax, (column, label) in zip(axes[1:], panels):
        values = table[column].astype(float)
        colors = [_model_color(m) for m in table.model_id]
        ax.hlines(y, 0, values, color=colors, linewidth=1.6, alpha=0.45)
        ax.scatter(values, y, color=colors, s=16, zorder=3)
        ax.set_xlim(0, np.nanmax(values) * 1.15 if np.isfinite(values).any() else 1)
        ax.set_xlabel(label, fontsize=7)
        layout.light_grid(ax)
        for i, kind in enumerate(table.kind):
            if kind == "hosted":
                ax.text(0.02, i, "hosted API", va="center", fontsize=7, color=style.REMOVED_COLOR,
                        transform=ax.get_yaxis_transform())
            elif kind == "encoder" and not np.isfinite(values.iloc[i]):
                ax.text(0.02, i, "not recorded", va="center", fontsize=7,
                        color=style.REMOVED_COLOR, transform=ax.get_yaxis_transform())
    for ax in axes:
        ax.axhline(boundary - 0.5, color=style.GRID_COLOR, linewidth=0.9)
    legend = [Patch(facecolor=style.EVIDENCE_MARKS["observed"], label="passed / recorded"),
              Patch(facecolor=style.EVIDENCE_MARKS["review"], label="pin differs: review"),
              Patch(facecolor=style.EVIDENCE_MARKS["failed"], label="failed"),
              Patch(facecolor=style.EVIDENCE_MARKS["unknown"], edgecolor=style.GRID_COLOR,
                    label="no local receipt"),
              Patch(facecolor=style.EVIDENCE_MARKS["not_applicable"], label="not applicable")]
    fig.legend(handles=legend, loc="outside lower center", ncols=3, fontsize=7, frameon=False)
    gpus = sorted({g for g in table.gpu_name.dropna()})
    return fig, "runtime_evidence", (
        "Experiment 0A gates for the selected decision models and three frozen encoders, read from "
        "the receipts available locally. Runtime contracts load the pinned checkpoint and answer "
        "synthetic and real requests; known-answer probes require the expected answer on synthetic "
        "inputs whose answer is unambiguous; robustness probes rerun real inputs under harmless "
        "interventions; the "
        "token audit counts every prepared request with the provider's own tokenizer; acceptance is "
        "the signed review that enables full open-weight extraction (Jev: the bounded 1.14 pilot). "
        f"Right panels give the contract run on {', '.join(gpus) or 'the recorded GPU'}: peak "
        "allocated memory, checkpoint loading time and total contract duration."
    )


def _profile(ax, values, position, color):
    q05, q25, q50, q75, q95 = np.nanpercentile(values, [5, 25, 50, 75, 95])
    ax.hlines(position, q05, q95, color=color, linewidth=0.9)
    ax.add_patch(plt.Rectangle((q25, position - 0.3), q75 - q25, 0.6, facecolor=color,
                               alpha=0.25, edgecolor=color, linewidth=0.8))
    ax.vlines(q50, position - 0.3, position + 0.3, color=color, linewidth=1.6)


TASK_FEATURES = {
    "binary_classification": ((0, 1), "p(positive class)"),
    "multiclass_classification": ((0, 1), "normalized entropy\n(0 = all mass on one class)"),
    "regression": ((-4, 4), "expected direction\n(−4 lower … +4 higher)"),
}


def pilot_output_figures(shared):
    """Each model's main feature on identical inputs: does a model separate texts at all?"""
    if shared.empty:
        return
    providers = shared.attrs.get("providers") or [p for p in LABELS if p in set(shared.provider)]
    tasks = [t for t in layout.TASK_ORDER if t in set(shared.task_type)]
    fig, axes = plt.subplots(1, len(tasks), figsize=(style.WIDTH_FULL, 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(0)
    for ax, task in zip(axes, tasks):
        part = shared[shared.task_type.eq(task)]
        for i, provider in enumerate(providers):
            values = part.loc[part.provider.eq(provider), "value"].to_numpy(float)
            color = style.DECISION_COLORS[provider]
            ax.scatter(values, i + rng.uniform(-0.22, 0.22, len(values)), s=3, color=color,
                       alpha=0.35, linewidths=0, zorder=1)
            _profile(ax, values, i, color)
        limits, label = TASK_FEATURES[task]
        pad = 0.03 * (limits[1] - limits[0])
        ax.set_xlim(limits[0] - pad, limits[1] + pad)
        if task == "regression":
            ax.axvline(0, color=style.GRID_COLOR, linewidth=1)
        ax.set_xlabel(label, fontsize=7)
        ax.set_title(f"{layout.TASK_LABELS[task]} · {part.input.nunique()} inputs", fontsize=8,
                     color=style.TASK_COLORS[task])
        layout.light_grid(ax)
    axes[0].set_yticks(range(len(providers)), [LABELS[p] for p in providers], fontsize=7)
    axes[0].set_ylim(len(providers) - 0.5, -0.5)
    axes[0].tick_params(axis="y", length=0)
    for tick, provider in zip(axes[0].get_yticklabels(), providers):
        tick.set_color(style.DECISION_COLORS[provider])
    yield fig, "decision_output_profiles", (
        "Main decision feature returned by each model for the same label-blind 0A pilot inputs "
        "(identical semantic requests across selected models at their pinned versions; per-column "
        "and joint requests from all 18 datasets). Boxes span p25–p75 with the median; lines span "
        "p5–p95; points are individual inputs. A narrow profile means a model returns similar "
        "values whatever the text; differences in location reflect each model's own framing and "
        "calibration. No target value is read."
    )


def pilot_agreement_figure(agreement):
    """Pairwise agreement between models on identical inputs, one matrix per task."""
    if not agreement:
        return
    tasks = [t for t in layout.TASK_ORDER if t in agreement]
    providers = list(next(iter(agreement.values()))[0].index)
    short = [style.DECISION_SHORT.get(p, p) for p in providers]
    fig, axes = plt.subplots(1, len(tasks), figsize=(style.WIDTH_FULL, 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap(style.AGREEMENT_DIVERGING)
    # Cell values stay legible (7 pt) only for small rosters; larger ones are tabulated.
    annotate = len(providers) <= 8
    for ax, task in zip(axes, tasks):
        matrix, n = agreement[task]
        # Strict lower triangle, without its empty first row and last column.
        values = matrix.loc[providers[1:], providers[:-1]].to_numpy(float)
        mask = np.triu(np.ones_like(values, dtype=bool), k=1)
        image = ax.imshow(np.ma.masked_array(values, mask), cmap=cmap, vmin=-1, vmax=1)
        for (i, j), value in np.ndenumerate(values):
            if annotate and i >= j and np.isfinite(value):
                ax.text(j, i, f"{value:.2f}".replace("0.", ".").replace("-.", "−."),
                        ha="center", va="center", fontsize=7,
                        color=style.BACKGROUND_COLOR if abs(value) > 0.6 else style.TEXT_COLOR)
        ax.set_xticks(range(len(providers) - 1), short[:-1], rotation=90, fontsize=7)
        ax.set_yticks(range(len(providers) - 1), short[1:], fontsize=7)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        measure = "same top class" if task == "multiclass_classification" else "Spearman ρ"
        ax.set_title(f"{layout.TASK_LABELS[task]}\n{measure} · {n} inputs", fontsize=8,
                     color=style.TASK_COLORS[task])
        for tick, provider in zip(ax.get_xticklabels(), providers[:-1]):
            tick.set_color(style.DECISION_COLORS[provider])
    for tick, provider in zip(axes[0].get_yticklabels(), providers[1:]):
        tick.set_color(style.DECISION_COLORS[provider])
    fig.colorbar(image, ax=list(axes), shrink=0.8, pad=0.015, label="Spearman ρ, or share with the same top class")
    yield fig, "decision_model_agreement", (
        "Agreement between every pair of decision models on the identical pilot inputs of the "
        "previous figure. Binary and regression: Spearman rank correlation of the main feature "
        "(1 = same ordering of texts). Multiclass: share of inputs on which both models give the "
        "highest probability to the same class (chance is roughly 1/K). Panel titles give the number "
        f"of inputs; exact values are in the accompanying table. Coefficients from as few as {min(n for _, n in agreement.values())} inputs are "
        "imprecise. Low agreement means the models would hand the predictor different information; "
        "it does not say which is correct."
    )


def robustness_matrix_figure(changes):
    """Distribution change of every model under each harmless intervention, same inputs."""
    if changes.empty:
        return
    from matplotlib.colors import LinearSegmentedColormap

    from src.data.loaders import load_catalog

    names = {entry["id"]: entry["name"] for entry in load_catalog()["datasets"]}
    providers = [p for p in LABELS if p in set(changes.provider)]
    datasets = list(dict.fromkeys(changes.dataset_id))
    tasks = {"BIN_": "binary_classification", "MUL_": "multiclass_classification",
             "REG_": "regression"}
    columns = [(d, v) for d in datasets for v in VARIANTS
               if ((changes.dataset_id == d) & (changes.variant == v)).any()]
    values = np.full((len(providers), len(columns)), np.nan)
    flipped = np.zeros_like(values, dtype=bool)
    for record in changes[changes.variant.ne("base")].itertuples():
        if record.provider in providers and (record.dataset_id, record.variant) in columns:
            i, j = providers.index(record.provider), columns.index((record.dataset_id, record.variant))
            values[i, j] = record.total_variation_from_base
            flipped[i, j] = not record.same_top_option
    cmap = LinearSegmentedColormap.from_list("tv", [style.BACKGROUND_COLOR, style.WINDOW_COLOR])
    fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, 3.9))
    ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=0, vmax=0.6, aspect="auto")
    for (i, j), value in np.ndenumerate(values):
        if np.isfinite(value):
            ax.text(j, i, f"{value:.2f}" + ("*" if flipped[i, j] else ""), ha="center",
                    va="center", fontsize=7, weight="bold" if flipped[i, j] else "normal",
                    color=style.BACKGROUND_COLOR if value > 0.4 else style.TEXT_COLOR)
    short = {"paraphrase": "Paraphrase", "irrelevant_context": "Unrelated\ncontext",
             "embedded_instruction": "Inserted\ninstruction", "option_order": "Reversed\noptions"}
    ax.set_xticks(range(len(columns)), [short[v] for _, v in columns], fontsize=7)
    ax.set_yticks(range(len(providers)), [LABELS[p] for p in providers], fontsize=7)
    for tick, provider in zip(ax.get_yticklabels(), providers):
        tick.set_color(style.DECISION_COLORS[provider])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    start = 0
    for dataset in datasets:
        width = sum(d == dataset for d, _ in columns)
        task = tasks[dataset[:4]]
        ax.text(start + (width - 1) / 2, -0.85, f"{layout.TASK_LABELS[task]} · {names.get(dataset, dataset)}",
                ha="center", va="bottom", fontsize=7, color=style.TASK_COLORS[task], weight="bold")
        if start:
            ax.axvline(start - 0.5, color=style.BACKGROUND_COLOR, linewidth=3)
        start += width
    ax.set_ylim(len(providers) - 0.5, -1.2)
    yield fig, "robustness_across_models", (
        "Total variation between each model's probability vector for an unmodified real input "
        "and for the same input under an intervention that should not change the answer (0 = "
        "identical, 1 = disjoint); one deterministically selected input per task, the same for "
        "every model. An asterisk marks a changed top answer. Reversed options apply only to "
        "multiclass and are mapped back to the original classes. Single probes illustrate "
        "sensitivity; they are not population failure rates."
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
                    ax.text(0.03, pos, "n/a", fontsize=7, color=style.REMOVED_COLOR,
                            va="center")
            ax.set(xlim=(-0.03, 1.03), xlabel="Total variation")
            ax.set_title(f"{fill(names.get(dataset, dataset), 22)}", fontsize=8,
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
        ax.set_title("Regression direction", fontsize=8, color=style.TASK_COLORS["regression"])
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
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                        color=style.BACKGROUND_COLOR if value >= 0.6 else style.TEXT_COLOR)
        for i, row in enumerate(part.itertuples()):
            ax.text(9.1, i, f"E = {row.mean:+.2f}   Var = {row.variance:.2f}", va="center",
                    fontsize=7)
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
