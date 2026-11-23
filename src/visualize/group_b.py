"""Group B figures (B1-B5), drawn at final A4 width from the per-experiment exports.

Every generator yields ``(figure, name, caption)`` and yields nothing for an empty table, so
the notebooks show a waiting state instead of an empty figure. Gains are always "arm minus
the same predictor's baseline (non-text columns only)" in the fixed headline metric; datasets
carry equal weight and intervals resample datasets.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src.visualize import style

PREDICTORS = style.MAIN_PREDICTORS
PREDICTOR_COLORS = dict(zip(PREDICTORS, ("#4E79A7", "#F28E2B", "#59A14F", "#B07AA1", "#9C755F",
                                         "#76B7B2", "#E15759")))
MARKERS = dict(zip(PREDICTORS, ("o", "s", "D", "^", "v", "P", "X")))
FAMILY_LABELS = {
    "decision/joint/median": "Decision, joint (median model)",
    "decision/per_column/median": "Decision, per column (median model)",
    "decision/both/median": "Decision, both (median model)",
    "decision+frozen/both/median": "Decision both + E5-small PCA32 (median model)",
}


def _label(representation: str) -> str:
    if representation in FAMILY_LABELS:
        return FAMILY_LABELS[representation]
    parts = representation.split("/")
    names = {"e5_small_v2": "E5-small", "e5_large_v2": "E5-large", "qwen3_embedding_0_6b": "Qwen3-Emb-0.6B",
             "e5_small_v2_tar": "TAR E5-small"}
    if parts[0] == "frozen":
        return f"Frozen {names.get(parts[1], parts[1])}, {parts[2].replace('per_column', 'per column ').replace('joint', 'joint ')}"
    if parts[0] == "tar":
        return f"TAR E5-small, {parts[2].replace('per_column', 'per column ').replace('joint', 'joint ')}"
    if parts[0] == "probe":
        return f"Label-trained classifier, {names.get(parts[1], parts[1])}, {parts[2].replace('_', ' ')}"
    return representation


def _legend(fig, models):
    handles = [Line2D([], [], ls="none", marker=MARKERS.get(m, "o"), ms=4,
                      color=PREDICTOR_COLORS.get(m, "black"), label=style.PREDICTOR_LABELS.get(m, m))
               for m in models]
    fig.legend(handles=handles, loc="outside lower center", ncol=min(4, len(handles)), frameon=False,
               fontsize=7.5)


def gain_forest(summary, representations, *, name, what, share=False):
    """Equal-dataset mean gain with dataset-bootstrap 95% intervals, one marker per predictor."""
    if summary is None or summary.empty:
        return
    data = summary[summary.representation.isin(representations)]
    if data.empty:
        return
    rows = [r for r in representations if r in set(data.representation)]
    models = [m for m in PREDICTORS if m in set(data.model_id)]
    column, low, high = (("mean_headroom_share", "share_ci_low", "share_ci_high") if share
                         else ("mean_raw_gain", "raw_ci_low", "raw_ci_high"))
    fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 1.2 + 0.42 * len(rows))),
                           layout="constrained")
    offsets = np.linspace(-0.3, 0.3, max(len(models), 1))
    for offset, model in zip(offsets, models):
        frame = data[data.model_id.eq(model)].set_index("representation").reindex(rows)
        y = np.arange(len(rows)) + offset
        ax.errorbar(frame[column], y, xerr=[frame[column] - frame[low], frame[high] - frame[column]],
                    fmt=MARKERS.get(model, "o"), ms=3.5, lw=0.8, color=PREDICTOR_COLORS.get(model),
                    capsize=0)
    ax.axvline(0, color=style.MEDIAN_COLOR, lw=0.8)
    ax.set_yticks(range(len(rows)), [_label(r) for r in rows], fontsize=7.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.grid(axis="x", color=style.GRID_COLOR, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Share of baseline headroom gained" if share
                  else "Gain over the no-text baseline (ROC-AUC, macro AUC or R²)", fontsize=8)
    _legend(fig, models)
    yield fig, name, (
        f"{what}: equal-dataset mean {'share of the baseline headroom (gain divided by one minus the baseline score)' if share else 'gain in the headline metric'} "
        "over the same predictor's no-text baseline, with 95% percentile intervals from resampling "
        "datasets (2,000 draws); one marker per predictor. Medians are taken over all seventeen "
        "decision models per dataset; no model is selected.")


def win_tie_loss(summary, representations, *, name, what):
    """Datasets on which the arm wins, ties (|gain| <= 0.001) or loses, per predictor."""
    if summary is None or summary.empty:
        return
    data = summary[summary.representation.isin(representations)]
    if data.empty:
        return
    rows = [r for r in representations if r in set(data.representation)]
    fig, axes = plt.subplots(1, len(rows), sharey=True, layout="constrained",
                             figsize=(style.WIDTH_FULL, 2.4), squeeze=False)
    colors = ("#1B7837", "#B9BDC5", "#B2182B")
    for ax, representation in zip(axes[0], rows):
        frame = data[data.representation.eq(representation)].set_index("model_id").reindex(
            [m for m in PREDICTORS if m in set(data.model_id)])
        left = np.zeros(len(frame))
        for column, color in zip(("wins", "ties", "losses"), colors):
            ax.barh(range(len(frame)), frame[column], left=left, color=color, height=0.7)
            left += frame[column].fillna(0).to_numpy()
        ax.set_title(_label(representation).replace(" (median model)", ""), loc="left", fontsize=7.5)
        ax.set_xlabel("Datasets", fontsize=8)
        ax.set_yticks(range(len(frame)), [style.PREDICTOR_LABELS.get(m, m) for m in frame.index],
                      fontsize=7.5)
        ax.set_ylim(len(frame) - 0.5, -0.5)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=label)
                        for c, label in zip(colors, ("Win", "Tie", "Loss"))],
               loc="outside lower center", ncol=3, frameon=False, fontsize=7.5)
    yield fig, name, (
        f"{what}: number of datasets on which the median decision model's arm scores above, within "
        "0.001 of, or below the same predictor's no-text baseline (five-fold mean headline metric).")


def model_by_predictor(gains, *, mode="both", name, what):
    """Equal-dataset mean gain of every decision model (rows) in every predictor (columns)."""
    if gains is None or gains.empty:
        return
    data = gains[gains.representation.str.startswith(f"decision/{mode}/")
                 & ~gains.representation.str.endswith("/median")]
    if data.empty:
        return
    data = data.assign(provider=data.representation.str.split("/").str[2])
    table = data.pivot_table(index="provider", columns="model_id", values="raw_gain", aggfunc="mean")
    table = table[[m for m in PREDICTORS if m in table.columns]]
    table = table.loc[table.mean(axis=1).sort_values(ascending=False).index]
    limit = float(np.nanmax(np.abs(table.to_numpy()))) or 1.0
    fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 1.0 + 0.2 * len(table))),
                           layout="constrained")
    image = ax.imshow(table.to_numpy(), cmap=style.EFFECT_CMAP, vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(table.shape[1]), [style.PREDICTOR_LABELS.get(m, m) for m in table.columns],
                  fontsize=7.5, rotation=30, ha="right")
    ax.set_yticks(range(len(table)), table.index, fontsize=7.5)
    fig.colorbar(image, ax=ax, shrink=0.8).set_label("Mean gain over baseline", fontsize=8)
    yield fig, name, (
        f"{what}: equal-dataset mean gain of each decision model's {mode.replace('_', '-')} features over "
        "the same predictor's no-text baseline (headline metric). Rows are ordered by their mean over "
        "predictors for readability only; nothing is selected.")


def association_heatmap(associations, *, name, what):
    """Spearman rho between label-free dataset properties and the median decision gain."""
    if associations is None or associations.empty or associations.spearman_rho.isna().all():
        return
    data = associations[associations.representation.eq("decision/both/median")]
    if data.empty:
        return
    table = data.pivot_table(index="property", columns="model_id", values="spearman_rho")
    qs = data.pivot_table(index="property", columns="model_id", values="q_value_bh")
    table = table[[m for m in PREDICTORS if m in table.columns]]
    fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 1.0 + 0.2 * len(table))),
                           layout="constrained")
    image = ax.imshow(table.to_numpy(), cmap=style.EFFECT_CMAP, vmin=-1, vmax=1, aspect="auto")
    for (i, j), value in np.ndenumerate(qs.reindex(index=table.index, columns=table.columns).to_numpy()):
        if np.isfinite(value) and value <= 0.10:
            ax.text(j, i, "*", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(table.shape[1]), [style.PREDICTOR_LABELS.get(m, m) for m in table.columns],
                  fontsize=7.5, rotation=30, ha="right")
    ax.set_yticks(range(len(table)), [p.replace("_", " ") for p in table.index], fontsize=7)
    fig.colorbar(image, ax=ax, shrink=0.8).set_label("Spearman ρ over datasets", fontsize=8)
    yield fig, name, (
        f"{what}: Spearman rank correlation, across datasets, between each label-free dataset property "
        "and the gain of the median decision model's both-view features, per predictor. An asterisk "
        "marks a Benjamini-Hochberg q-value of at most 0.10 across all properties and predictors; "
        "these are associations, not causal effects.")


INPUT_SET_TEXT = {
    "pre_extraction": ("median text length only", "whether to acquire decision features"),
    "pilot": ("median text length and decision redundancy on a fixed 200-row label-free pilot",
              "whether to acquire them after a 200-request pilot"),
    "post_extraction": ("median text length and decision redundancy on all rows",
                        "whether to use decision features already extracted"),
}


def lodo_scatter(predictions, *, name, what, input_set=None):
    """Leave-one-dataset-out predicted versus actual gain, one panel per predictor."""
    if predictions is None or predictions.empty:
        return
    if input_set is not None and "input_set" in predictions:
        predictions = predictions[predictions.input_set.eq(input_set)]
        if predictions.empty:
            return
    models = [m for m in PREDICTORS if m in set(predictions.model_id)]
    columns = 4
    rows = int(np.ceil(len(models) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 1.7 * rows + 0.4)),
                             layout="constrained", squeeze=False, sharex=True, sharey=True)
    for ax, model in zip(axes.flat, models):
        frame = predictions[predictions.model_id.eq(model)]
        ax.scatter(frame.predicted, frame.actual, s=10, color=PREDICTOR_COLORS.get(model), linewidths=0)
        ax.axhline(0, color=style.GRID_COLOR, lw=0.6)
        ax.axvline(0, color=style.GRID_COLOR, lw=0.6)
        ax.set_title(style.PREDICTOR_LABELS.get(model, model), loc="left", fontsize=7.5)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(models):]:
        ax.axis("off")
    fig.supxlabel("Predicted gain (held-out dataset)", fontsize=8)
    fig.supylabel("Actual gain", fontsize=8)
    inputs, decision = INPUT_SET_TEXT.get(
        input_set, ("two prespecified label-free properties", "whether to use decision features"))
    yield fig, name, (
        f"{what}: each dataset's gain from the median decision model's both-view features, predicted "
        f"before seeing it by a ridge rule fitted on the other datasets' {inputs}, against the actual "
        f"gain; the rule decides {decision}. Points in the upper-right and lower-left quadrants are "
        "correct use/don't-use decisions.")


def size_curves(size_gains, *, name, what, outcome="raw_gain", curve="main", main_max=2048):
    """Mean gain against training size (log2), per predictor; decision = median over models.

    `main`: every dataset at sizes up to `main_max`; `extended`: only the datasets sampled at
    sizes above it, at every size, so each curve keeps one fixed set of datasets.
    """
    if size_gains is None or size_gains.empty:
        return
    data = size_gains.copy()
    if curve == "main":
        data = data[data.training_size.le(main_max)]
    else:
        extended = set(data.loc[data.training_size.gt(main_max), "dataset_id"])
        data = data[data.dataset_id.isin(extended)]
    if data.empty:
        return
    parts = data.representation.str.split("/")
    data["arm"] = np.where(parts.str[0].eq("decision"), "Decision both (median model)",
                           [_label(r) for r in data.representation])
    data = data.groupby(["dataset_id", "model_id", "arm", "training_size"], as_index=False)[outcome].median()
    means = data.groupby(["model_id", "arm", "training_size"], as_index=False).agg(
        value=(outcome, "mean"), datasets=("dataset_id", "nunique"))
    models = [m for m in PREDICTORS if m in set(means.model_id)]
    arms = sorted(set(means.arm), key=lambda a: (not a.startswith("Decision"), a))
    palette = dict(zip(arms, ("#1B7837", "#A64B72", "#4E79A7", "#B35806", "#9467BD", "#558B2F")))
    columns = 4
    rows = int(np.ceil(len(models) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 1.8 * rows + 0.7)),
                             layout="constrained", squeeze=False, sharex=True, sharey=True)
    for ax, model in zip(axes.flat, models):
        for arm in arms:
            frame = means[means.model_id.eq(model) & means.arm.eq(arm)].sort_values("training_size")
            ax.plot(np.log2(frame.training_size), frame.value, marker="o", ms=2.5, lw=1,
                    color=palette.get(arm))
        ax.axhline(0, color=style.MEDIAN_COLOR, lw=0.6)
        ax.set_title(style.PREDICTOR_LABELS.get(model, model), loc="left", fontsize=7.5)
        ax.tick_params(labelsize=7)
        sizes = sorted(set(means.training_size))
        ax.set_xticks(np.log2(sizes), [f"{s // 1024}k" if s >= 1024 else str(s) for s in sizes],
                      fontsize=6, rotation=90)
    for ax in axes.flat[len(models):]:
        ax.axis("off")
    handles = [Line2D([], [], color=palette[a], lw=1.2, label=a) for a in arms]
    fig.supxlabel("Labelled training rows", fontsize=8)
    fig.supylabel("Gain over same-size baseline" if outcome == "raw_gain" else "Share of headroom", fontsize=8)
    if len(models) < axes.size:
        axes.flat[-1].legend(handles=handles, loc="center", frameon=False, fontsize=6.5)
    else:
        fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False, fontsize=7)
    datasets = data.dataset_id.nunique()
    scope = (f"{'Main' if curve == 'main' else 'Extended'} curve, {datasets} datasets. ")
    yield fig, name, (
        f"{what}. {scope}Mean gain over the same-size, same-draw no-text baseline against the number of sampled "
        "labelled training rows (nested stratified subsamples; test rows unchanged), averaged over draws "
        "within a dataset and then over datasets with equal weight. The decision line is the per-dataset "
        "median over all seventeen decision models.")


def native_references(folds_headline, *, name, what):
    """Native-text systems' headline metric per dataset beside the seven baselines."""
    if folds_headline is None or folds_headline.empty:
        return
    per = folds_headline.groupby(["dataset_id", "model_id", "representation"], as_index=False).value.mean()
    native = per[per.representation.str.startswith("native/")]
    base = per[per.representation.eq("baseline")]
    if native.empty:
        return
    datasets = sorted(set(native.dataset_id))
    fig, ax = plt.subplots(figsize=(style.WIDTH_FULL, min(style.MAX_HEIGHT, 1.0 + 0.2 * len(datasets))),
                           layout="constrained")
    for i, dataset in enumerate(datasets):
        values = base[base.dataset_id.eq(dataset)].value
        ax.scatter(values, np.full(len(values), i), s=8, color=style.MISSING_COLOR, linewidths=0, zorder=2)
        for model, frame in native[native.dataset_id.eq(dataset)].groupby("model_id"):
            ax.scatter(frame.value, [i], s=16, marker=style.NATIVE_MARKERS.get(model, "o"),
                       color=style.NATIVE_COLORS.get(model, "black"), zorder=3, linewidths=0)
    ax.set_yticks(range(len(datasets)), [d.split("_TEXT_")[-1].replace("_", " ").title() for d in datasets],
                  fontsize=7)
    ax.set_ylim(len(datasets) - 0.5, -0.5)
    ax.set_xlabel("Five-fold mean headline metric (ROC-AUC, macro AUC or R²)", fontsize=8)
    ax.grid(axis="x", color=style.GRID_COLOR, lw=0.6)
    handles = [Line2D([], [], ls="none", marker="o", ms=3, color=style.MISSING_COLOR,
                      label="Seven predictors, no text")]
    handles += [Line2D([], [], ls="none", marker=style.NATIVE_MARKERS[m], ms=4, color=style.NATIVE_COLORS[m],
                       label=style.PREDICTOR_LABELS[m]) for m in style.NATIVE_COLORS]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=7.5)
    yield fig, name, (
        f"{what}: five-fold mean headline metric of each native-text system per dataset (coloured), "
        "beside the no-text baseline of each of the seven fixed predictors (grey). Descriptive "
        "whole-system references; they share no predictor with the decision-feature arms.")


def render(save, generator):
    """Save (PDF + caption), display and close every figure of a generator."""
    import matplotlib.pyplot as plt

    from src.utils.notebook_display import display_local

    for fig, name, caption in generator:
        save(fig, name, caption=caption)
        display_local(fig)
        plt.close(fig)
