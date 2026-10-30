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
    """Research questions, controlled comparisons and grid sizes of the three experiments."""
    from matplotlib.patches import FancyBboxPatch

    counts = grid.set_index("experiment")
    extractors = len(load_yaml("experiment_1/main")["candidate_extractors"])
    fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, 3.3))
    # A pure schematic: fixed axes, so long text lines never shrink the drawing area.
    fig.set_layout_engine("none")
    ax.set_position([0.01, 0.01, 0.98, 0.98])
    ax.set_axis_off()
    ax.set(xlim=(0, 1), ylim=(0, 1))
    experiments = [
        ("experiment_1", "Experiment 1 · RQ1", "Do decision features add\ninformation to a fixed predictor?",
         ["non-text columns, alone or with", f"decision features from {extractors} models,",
          "asked per column, jointly or both"], "6 predictors"),
        ("experiment_2", "Experiment 2 · RQ2", "How do they compare with\nfrozen general embeddings?",
         ["E5-small, E5-large or Qwen3", "embeddings, PCA to 32 per column", "or 256 … 8 jointly"],
         "same 6 predictors"),
        ("experiment_3", "Experiment 3 · RQ3", "How do the pipelines compare\nwith native text systems?",
         ["TabSTAR v1.1, ConTextTab and", "AutoGluon-Multimodal read the", "raw text themselves"],
         "3 complete systems"),
    ]
    for k, (phase, head, question, body, predictor) in enumerate(experiments):
        x, color = 0.005 + k * 0.333, style.FAMILY_COLORS[phase]
        row = counts.loc[phase]
        ax.add_patch(FancyBboxPatch((x, 0.2), 0.32, 0.62, boxstyle="round,pad=0,rounding_size=0.015",
                                    facecolor=style.BACKGROUND_COLOR, edgecolor=color, linewidth=1.1))
        ax.add_patch(FancyBboxPatch((x, 0.69), 0.32, 0.13, boxstyle="round,pad=0,rounding_size=0.015",
                                    facecolor=color, edgecolor=color, linewidth=1.1))
        ax.text(x + 0.16, 0.755, head, ha="center", va="center", color=style.BACKGROUND_COLOR,
                weight="bold", fontsize=8)
        ax.text(x + 0.015, 0.665, question, va="top", fontsize=7.2, weight="bold", linespacing=1.15)
        ax.text(x + 0.015, 0.55, "\n".join(body), va="top", fontsize=6.8, linespacing=1.25)
        ax.text(x + 0.015, 0.335, predictor, va="top", fontsize=6.8, color=color, weight="bold")
        ax.text(x + 0.015, 0.265, f"{row.configurations:,} configurations · "
                f"{row.outer_fits:,} outer fits", va="top", fontsize=6.8)
    ax.text(0.005, 0.95, "0A runtime gates  →  full extraction  →  0B feature and predictor "
            "checks  →  experiments", fontsize=7, color=style.QUANTITY_COLOR, va="center")
    ax.text(0.005, 0.1, "Shared protocol: 18 MulTaBench text datasets · the same 5 shuffled "
            "outer folds · ROC-AUC, macro OvR ROC-AUC or R² · paired per dataset",
            fontsize=6.8, va="center")
    ax.text(0.005, 0.03, f"Total {grid.configurations.sum():,} configurations · "
            f"{grid.outer_fits.sum():,} outer fits + {grid.binary_inner_fits.sum():,} binary inner "
            "fits for F1 thresholds", fontsize=6.8, va="center", color=style.QUANTITY_COLOR)
    return fig, "study_map", (
        "Design of the three experiments. Each answers one research question with the rows, "
        "splits and endpoint held fixed. Experiment 1 shares one non-text baseline per predictor "
        "across the decision models; Experiment 2 holds the same predictors fixed and swaps the "
        "text representation; Experiment 3 changes the whole system. Configurations are planned "
        "before width exclusions (Kumo refuses inputs above 500 columns); one configuration is "
        "one dataset × predictor × representation, each with five outer fits."
    )


def _model_color(model_id):
    return style.DECISION_COLORS.get(model_id, style.EMBEDDING_COLORS.get(model_id, style.TEXT_COLOR))


def runtime_evidence_figure(table):
    """Acceptance gates (left) beside the measured B200 footprint of each runtime (right)."""
    from matplotlib.patches import Patch

    gates = [("bounded_runtime", "Contract"), ("known_answer", "Known\nanswers"),
             ("robustness", "Robust-\nness"), ("token_audit", "Token\naudit"),
             ("accepted", "Accepted")]
    names = [LABELS.get(m, ENCODER_LABELS.get(m, m)) for m in table.model_id]
    y = np.arange(len(table))
    fig, axes = plt.subplots(1, 4, figsize=(style.WIDTH_FULL, 3.9), sharey=True,
                             gridspec_kw={"width_ratios": [1.6, 1, 1, 1]})
    ax = axes[0]
    for j, (column, _) in enumerate(gates):
        ax.scatter(np.full(len(table), j), y, marker="s", s=58,
                   color=[style.EVIDENCE_MARKS[s] for s in table[column]],
                   edgecolors=style.GRID_COLOR, linewidths=0.6)
    ax.set_xticks(range(len(gates)), [label for _, label in gates], fontsize=6)
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
                ax.text(0.02, i, "hosted API", va="center", fontsize=6, color=style.REMOVED_COLOR,
                        transform=ax.get_yaxis_transform())
            elif kind == "encoder" and not np.isfinite(values.iloc[i]):
                ax.text(0.02, i, "not recorded", va="center", fontsize=6,
                        color=style.REMOVED_COLOR, transform=ax.get_yaxis_transform())
    for ax in axes:
        ax.axhline(boundary - 0.5, color=style.GRID_COLOR, linewidth=0.9)
    legend = [Patch(facecolor=style.EVIDENCE_MARKS["observed"], label="passed / recorded"),
              Patch(facecolor=style.EVIDENCE_MARKS["review"], label="pin differs: review"),
              Patch(facecolor=style.EVIDENCE_MARKS["failed"], label="failed"),
              Patch(facecolor=style.EVIDENCE_MARKS["unknown"], edgecolor=style.GRID_COLOR,
                    label="no local receipt"),
              Patch(facecolor=style.EVIDENCE_MARKS["not_applicable"], label="not applicable")]
    fig.legend(handles=legend, loc="outside lower center", ncols=5, fontsize=6.5, frameon=False)
    gpus = sorted({g for g in table.gpu_name.dropna()})
    return fig, "runtime_evidence", (
        "Experiment 0A gates for the twelve decision models and three frozen encoders, read from "
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
        "(identical payloads answered by all twelve models at their pinned versions; per-column "
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
    fig, axes = plt.subplots(1, len(tasks), figsize=(style.WIDTH_FULL, 2.75), sharey=True)
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap(style.AGREEMENT_DIVERGING)
    for ax, task in zip(axes, tasks):
        matrix, n = agreement[task]
        # Strict lower triangle, without its empty first row and last column.
        values = matrix.loc[providers[1:], providers[:-1]].to_numpy(float)
        mask = np.triu(np.ones_like(values, dtype=bool), k=1)
        image = ax.imshow(np.ma.masked_array(values, mask), cmap=cmap, vmin=-1, vmax=1)
        for (i, j), value in np.ndenumerate(values):
            if i >= j and np.isfinite(value):
                ax.text(j, i, f"{value:.2f}".replace("0.", ".").replace("-.", "−."),
                        ha="center", va="center", fontsize=4.8,
                        color=style.BACKGROUND_COLOR if abs(value) > 0.6 else style.TEXT_COLOR)
        ax.set_xticks(range(len(providers) - 1), short[:-1], rotation=90, fontsize=6)
        ax.set_yticks(range(len(providers) - 1), short[1:], fontsize=6)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        measure = "same top class" if task == "multiclass_classification" else "Spearman ρ"
        ax.set_title(f"{layout.TASK_LABELS[task]}\n{measure} · {n} inputs", fontsize=7.5,
                     color=style.TASK_COLORS[task])
        for tick, provider in zip(ax.get_xticklabels(), providers[:-1]):
            tick.set_color(style.DECISION_COLORS[provider])
    for tick, provider in zip(axes[0].get_yticklabels(), providers[1:]):
        tick.set_color(style.DECISION_COLORS[provider])
    fig.colorbar(image, ax=list(axes), shrink=0.8, pad=0.015, label="agreement")
    yield fig, "decision_model_agreement", (
        "Agreement between every pair of decision models on the identical pilot inputs of the "
        "previous figure. Binary and regression: Spearman rank correlation of the main feature "
        "(1 = same ordering of texts). Multiclass: share of inputs on which both models give the "
        "highest probability to the same class (chance is roughly 1/K). Panel titles give the number "
        f"of inputs; coefficients from as few as {min(n for _, n in agreement.values())} inputs are "
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
    fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, 3.6))
    ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=0, vmax=0.6, aspect="auto")
    for (i, j), value in np.ndenumerate(values):
        if np.isfinite(value):
            ax.text(j, i, f"{value:.2f}" + ("*" if flipped[i, j] else ""), ha="center",
                    va="center", fontsize=6, weight="bold" if flipped[i, j] else "normal",
                    color=style.BACKGROUND_COLOR if value > 0.4 else style.TEXT_COLOR)
    short = {"paraphrase": "Paraphrase", "irrelevant_context": "Unrelated\ncontext",
             "embedded_instruction": "Inserted\ninstruction", "option_order": "Reversed\noptions"}
    ax.set_xticks(range(len(columns)), [short[v] for _, v in columns], fontsize=6.3)
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
