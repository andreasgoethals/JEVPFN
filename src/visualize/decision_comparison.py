"""Figures from saved features only; missing extraction never becomes invented results."""

from __future__ import annotations

import inspect
import json
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator, PercentFormatter

from src.evaluation.decision_comparison import dataset_balanced
from src.visualize import style

LABELS = {
    "jev": "Jev",
    "liquid": "Liquid d1-3B",
    "liquid_lfm3": "Liquid d1-2B",
    "perplexity": "Perplexity",
    "cloudflare": "Clef-flash",
    "intern": "Intern-Decision-4B",
    "strands": "Strands",
    "h2o": "H2O-Lightning-4B",
    "sii": "SII-Decision-27B",
    "kev": "Kev-27B",
    "decision3": "Decision 3.0 Turbo",
    "nimble": "Nimble-9B",
}
TASKS = {
    "binary_classification": "Binary",
    "multiclass_classification": "Multiclass",
    "regression": "Regression",
}
MODES = {"per_column": "Separate text columns", "joint": "Joint text"}


def _matrix(data, *, title, label, vmin=0, vmax=1, cmap=None, percent=False):
    size = style.comparison_size(len(data), rotated_labels=True)
    fig, ax = plt.subplots(figsize=size)
    im = ax.imshow(
        data.to_numpy(dtype=float),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap or style.AGREEMENT_CMAP,
        aspect="auto",
    )
    ax.set_xticks(
        range(len(data.columns)),
        [LABELS.get(str(c), str(c)) for c in data.columns],
        rotation=35,
        ha="right",
    )
    ax.set_yticks(range(len(data)), [LABELS.get(str(i), str(i)) for i in data.index])
    for i, row in enumerate(data.to_numpy(dtype=float)):
        for j, v in enumerate(row):
            ax.text(
                j,
                i,
                "—" if not np.isfinite(v) else (f"{v:.0%}" if percent else f"{v:.3f}"),
                ha="center",
                va="center",
                fontsize=style.COMPARISON_MATRIX_FONT,
                color=style.BACKGROUND_COLOR
                if np.isfinite(v) and (v - vmin) / (vmax - vmin) > 0.65
                else style.TEXT_COLOR,
            )
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=label, shrink=0.8)
    return fig


def coverage_figures(result, options):
    inventory = result.table("inventory")
    if not inventory.empty and inventory.status.eq("not_created").all():
        # A compact status table replaces identical empty dataset/provider heatmaps.
        fig, ax = plt.subplots(figsize=style.COMPARISON_PENDING_SIZE)
        ax.set_axis_off()
        rows = []
        for provider in options["providers"]:
            rows.append(
                [
                    LABELS[provider],
                    *[
                        f"0 / {inventory[(inventory.provider == provider) & (inventory.representation == mode)].dataset.nunique()} datasets"
                        for mode in options["representations"]
                    ],
                ]
            )
        table = ax.table(
            cellText=rows,
            colLabels=["Extractor", *[MODES[m] for m in options["representations"]]],
            bbox=[0, 0, 1, 0.94],
            cellLoc="left",
            colLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor(style.GRID_COLOR)
            if row == 0:
                cell.set_facecolor(style.GRID_COLOR)
        ax.set_title("Feature extraction pending · no measured agreement yet", pad=12)
        yield (
            fig,
            "coverage_pending",
            (
                "No selected decision-feature artifacts have been created. Counts give complete saved datasets "
                "per extractor and representation. Agreement, distributions and accuracy are not yet measurable."
            ),
        )
        return
    for mode in options["representations"]:
        frame = inventory[inventory.representation.eq(mode)].copy()
        frame["available"] = frame.status.eq("available").astype(int)
        table = frame.pivot(index="dataset", columns="provider", values="available").reindex(
            columns=options["providers"]
        )
        fig, ax = plt.subplots(figsize=style.COMPARISON_COVERAGE_SIZE)
        ax.imshow(
            table,
            cmap=ListedColormap([style.GRID_COLOR, style.TOTAL_COLOR]),
            vmin=0,
            vmax=1,
            aspect="auto",
        )
        ax.set_xticks(
            range(len(table.columns)), [LABELS[p] for p in table.columns], rotation=35, ha="right"
        )
        ax.set_yticks(range(len(table)), table.index)
        for i, row in enumerate(table.to_numpy()):
            for j, value in enumerate(row):
                ax.text(
                    j,
                    i,
                    "Saved" if value else "Pending",
                    ha="center",
                    va="center",
                    fontsize=style.COMPARISON_MATRIX_FONT,
                )
        ax.set_title(f"Saved feature coverage · {MODES[mode]}")
        yield (
            fig,
            f"coverage_{mode}",
            (
                f"Availability of complete cached feature artifacts for {MODES[mode].lower()}. "
                "Pending means no selected artifact exists; it is not a measured model failure. "
                "Only saved, checksum-verified artifacts enter subsequent comparisons."
            ),
        )


def pair_figures(result, options):
    pairs = result.table("pairs")
    if pairs.empty:
        return
    metrics = ["agreement", "total_variation", "js_divergence", "scalar_mae", "pearson", "spearman"]
    overall = dataset_balanced(
        pairs, ["task", "representation", "weighting", "model_a", "model_b"], metrics
    )
    for (task, mode), subset in overall.groupby(["task", "representation"], sort=False):
        for metric in ["agreement", "total_variation", "js_divergence"] + (
            ["scalar_mae", "pearson"] if task == "regression" else []
        ):
            data = subset[subset.weighting.eq("unique_inputs")]
            names = [p for p in options["providers"] if p in set(data.model_a) | set(data.model_b)]
            matrix = pd.DataFrame(np.nan, index=names, columns=names)
            for row in data.itertuples():
                matrix.loc[row.model_a, row.model_b] = getattr(row, metric)
                matrix.loc[row.model_b, row.model_a] = getattr(row, metric)
            # Diagonal left blank: no observations of self-agreement are added to the evidence.
            label = {
                "agreement": "Same class / same direction",
                "total_variation": "Total variation (0–1)",
                "js_divergence": "Jensen–Shannon divergence (bits)",
                "scalar_mae": "Mean absolute direction difference",
                "pearson": "Pearson correlation of expected directions",
            }[metric]
            fig = _matrix(
                matrix,
                title=f"{TASKS[task]} · {MODES[mode]}\n{label}",
                label=label,
                vmax=8 if metric == "scalar_mae" else 1,
                vmin=-1 if metric == "pearson" else 0,
                percent=metric == "agreement",
                cmap=style.AGREEMENT_CMAP
                if metric in {"agreement", "pearson"}
                else style.DISTANCE_CMAP,
            )
            counts = data.datasets
            yield (
                fig,
                f"pairs_{task}_{mode}_{metric}",
                (
                    f"{label} between decision models on matched {MODES[mode].lower()} inputs for {TASKS[task].lower()} tasks. "
                    f"Each input counts once within its text group; groups are averaged within dataset, then datasets equally ({int(counts.min())}–{int(counts.max())} per pair). "
                    "Missing cells are unavailable comparisons. Counts and row-weighted estimates accompany the figure in the report. "
                    "Classification agreement uses 0.5 for binary probabilities and argmax for multiclass; regression agreement uses the configured sign categories."
                ),
            )
        # Show how much duplicates change the measured agreement, without substituting one estimate.
        row = subset[subset.weighting.eq("rows")].set_index(["model_a", "model_b"])
        unique = subset[subset.weighting.eq("unique_inputs")].set_index(["model_a", "model_b"])
        matched = unique.index.intersection(row.index, sort=False)
        for page, start in enumerate(range(0, len(matched), 12), 1):
            fig, ax = plt.subplots(figsize=style.comparison_size(len(matched[start : start + 12])))
            labels = []
            for i, key in enumerate(matched[start : start + 12]):
                a, b = 100 * unique.loc[key, "agreement"], 100 * row.loc[key, "agreement"]
                ax.plot([a, b], [i, i], color=style.GRID_COLOR)
                ax.scatter(
                    a, i, facecolors="none", edgecolors=style.QUANTITY_COLOR,
                    s=style.AUDIT_MARKER_SIZE, label="Unique inputs" if i == 0 else None,
                )
                ax.scatter(
                    b, i, color=style.QUANTITY_COLOR, marker="x",
                    s=style.AUDIT_MARKER_SIZE, label="Source rows" if i == 0 else None,
                )
                labels.append(f"{LABELS[key[0]]} / {LABELS[key[1]]}")
            ax.set_yticks(range(len(labels)), labels)
            ax.invert_yaxis()
            ax.set_xlim(0, 100)
            ax.set_xlabel("Agreement (%) · equal weight per dataset")
            ax.set_title(f"Duplicate sensitivity · {TASKS[task]} · {MODES[mode]} · page {page}",
                         pad=30)
            ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=7)
            yield (
                fig,
                f"reuse_{task}_{mode}_{page}",
                "Agreement with each source row counted versus exact inputs counted once per text group. "
                "Pairs are paginated at twelve per page; only pairs with both weightings are plotted. "
                "All metric calculations use the complete matched data.",
            )


def consensus_figures(result, options):
    consensus, votes = result.table("consensus"), result.table("votes")
    if consensus.empty:
        return
    balanced = dataset_balanced(
        consensus,
        ["task", "representation", "weighting", "provider"],
        ["distance_to_others", "farthest_share"],
    )
    for (task, mode), frame in balanced[balanced.weighting.eq("unique_inputs")].groupby(
        ["task", "representation"], sort=False
    ):
        frame = frame.sort_values("distance_to_others")
        fig, axes = plt.subplots(1, 2, sharey=True, figsize=style.COMPARISON_MATRIX_SIZE)
        for ax, metric, label in zip(
            axes,
            ["distance_to_others", "farthest_share"],
            ["Mean distance to other models", "Most different inputs (%)"],
        ):
            ax.barh(
                [LABELS[p] for p in frame.provider],
                frame[metric],
                color=[style.DECISION_COLORS[p] for p in frame.provider],
            )
            ax.set_xlabel(label)
            ax.set_xlim(left=0)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="upper"))
        axes[1].xaxis.set_major_formatter(PercentFormatter(1))
        fig.suptitle(f"{TASKS[task]} · {MODES[mode]}\nDifferent does not mean less accurate")
        yield (
            fig,
            f"outliers_{task}_{mode}",
            (
                "Distance to the arithmetic mean of the other models on the complete configured roster. "
                "Classification uses total variation; regression uses absolute expected-direction difference. "
                "Exact ties for furthest receive fractional credit. Unique inputs, text groups and datasets receive equal weight at their respective aggregation levels."
            ),
        )
        sub = votes[
            (votes.task == task)
            & (votes.representation == mode)
            & (votes.weighting == "unique_inputs")
        ]
        counts = dataset_balanced(sub, ["largest_agreeing_group"], ["fraction"])
        raw = sub.groupby("largest_agreeing_group")["count"].sum()
        fig, ax = plt.subplots(figsize=style.figsize())
        ax.bar(
            counts.largest_agreeing_group.astype(str),
            counts.fraction * 100,
            color=style.MEDIAN_COLOR,
        )
        for i, r in enumerate(counts.itertuples()):
            ax.text(i, r.fraction * 100 + 1, f"{int(raw[r.largest_agreeing_group]):,}", ha="center")
        ax.set_ylim(0, 110)
        ax.set_yticks(range(0, 101, 20))
        ax.set_xlabel("Number of models sharing the most common class/direction")
        ax.set_ylabel("Inputs (%) · equal dataset weight")
        ax.set_title(f"How many agree? · {TASKS[task]} · {MODES[mode]}")
        yield (
            fig,
            f"votes_{task}_{mode}",
            (
                "Size of the largest agreeing group of decision models. Heights are dataset-balanced percentages; "
                "annotations are absolute unique-input counts summed across text groups (not a global deduplication). "
                "Only inputs present for every configured model are included. Agreement is not correctness."
            ),
        )
    # Per-dataset view prevents the overall average from hiding domain-specific outliers.
    for mode in options["representations"]:
        sub = consensus[
            (consensus.representation == mode) & (consensus.weighting == "unique_inputs")
        ]
        if sub.empty:
            continue
        table = (
            sub.groupby(["dataset", "provider"])
            .farthest_share.mean()
            .unstack()
            .reindex(columns=options["providers"])
        )
        fig = _matrix(
            table,
            title=f"Most different model by dataset · {MODES[mode]}",
            label="Share of inputs furthest from the other models",
            percent=True,
        )
        fig.set_size_inches(*style.COMPARISON_COVERAGE_SIZE)
        yield (
            fig,
            f"outliers_by_dataset_{mode}",
            "Tie-adjusted fraction of inputs for which a model is furthest from the others' mean, per dataset. Text groups receive equal weight; every configured model is required.",
        )


def distribution_figures(result, options):
    frame = result.table("distributions")
    if frame.empty:
        return
    # One dataset per page keeps variable text-column counts and physical scales legible.
    for (dataset, task, mode), group in frame[frame.weighting.eq("unique_inputs")].groupby(
        ["dataset", "task", "representation"], sort=False
    ):
        for variable in ("main_scalar", "entropy"):
            sub = group[group.variable.eq(variable)]
            stats, labels, colors = [], [], []
            for row in sub.itertuples():
                stats.append(
                    {
                        "med": row.median,
                        "q1": row.q1,
                        "q3": row.q3,
                        "whislo": row.p01,
                        "whishi": row.p99,
                        "fliers": [],
                    }
                )
                labels.append(f"{LABELS[row.provider]} · {row.text_column}")
                colors.append(style.DECISION_COLORS[row.provider])
            # Paginate long per-column datasets instead of squeezing labels onto one plot.
            for page, start in enumerate(range(0, len(stats), 12), 1):
                fig, ax = plt.subplots(figsize=style.COMPARISON_MATRIX_SIZE)
                orientation = (
                    {"orientation": "horizontal"}
                    if "orientation" in inspect.signature(ax.bxp).parameters
                    else {"vert": False}
                )
                artists = ax.bxp(
                    stats[start : start + 12],
                    **orientation,
                    patch_artist=True,
                    showfliers=False,
                    medianprops={"color": style.TEXT_COLOR},
                )
                for patch, color in zip(artists["boxes"], colors[start : start + 12]):
                    patch.set_facecolor(color)
                ax.set_yticks(
                    range(1, len(stats[start : start + 12]) + 1), labels[start : start + 12]
                )
                ax.set_xlim(
                    (-4, 4) if variable == "main_scalar" and task == "regression" else (0, 1)
                )
                label = (
                    "Normalized entropy"
                    if variable == "entropy"
                    else (
                        "Expected direction"
                        if task == "regression"
                        else "Positive-class probability"
                        if task == "binary_classification"
                        else "Largest class probability"
                    )
                )
                ax.set_xlabel(label)
                ax.set_title(f"{dataset}\n{MODES[mode]} · {label} · page {page}")
                dataset_id = sub.iloc[0].dataset_id
                yield (
                    fig,
                    f"distribution_{dataset_id}_{mode}_{variable}_{page}",
                    (
                        f"{label} by decision model and text input. Boxes span the 25th–75th percentiles, lines show medians and whiskers the 1st–99th percentiles. "
                        "Each exact input counts once; extrema and row-weighted statistics are retained in the report. "
                        "Entropy describes concentration of the reported probabilities, not measured calibration or correctness."
                    ),
                )


def example_figures(result, options):
    examples = result.table("examples")
    if examples.empty:
        return
    for task, frame in examples.groupby("task", sort=False):
        row = frame.sort_values(
            ["disagreement", "dataset_id", "row_id"], ascending=[False, True, True]
        ).iloc[0]
        probabilities = json.loads(row.probabilities)
        table = pd.DataFrame.from_dict(
            probabilities, orient="index", columns=[str(c) for c in json.loads(row.labels)]
        )
        if len(table.columns) > 9:
            # Thirty classes need rows, not thirty overlapping annotations across A4 width.
            table = table.T
        for page, start in enumerate(range(0, len(table), 12), 1):
            fig = _matrix(
                table.iloc[start : start + 12],
                title=f"{TASKS[task]} disagreement example · page {page}\n"
                      f"{row.dataset} · {row.row_id} · {row.text_column}",
                label="Reported probability",
            )
            yield (
                fig,
                f"disagreement_example_{task}_{page}",
                (
                    "Complete option-probability vectors for a highly disagreeing input, "
                    "paginated at twelve rows with the same 0–1 color scale. "
                    "The report retains its original CSV row ID, text group and payload hash. "
                    "This is a label-blind descriptive example, not a selected prediction error. "
                    "Other high-disagreement examples are retained in the table."
                ),
            )
    scatter = result.table("scatter")
    if scatter.empty:
        return
    for (task, mode), group in scatter.groupby(["task", "representation"], sort=False):
        names = [p for p in options["providers"] if p in set(group.model_a) | set(group.model_b)]
        pairs = list(combinations(names, 2))
        for page, start in enumerate(range(0, len(pairs), 3), 1):
            fig, axes = plt.subplots(1, 3, figsize=style.COMPARISON_SCATTER_FIGURE_SIZE)
            for ax, (a, b) in zip(axes, pairs[start : start + 3]):
                sub = group[(group.model_a == a) & (group.model_b == b)]
                if sub.empty:
                    ax.set_axis_off()
                    continue
                ax.scatter(
                    sub.x,
                    sub.y,
                    s=style.COMPARISON_SCATTER_SIZE,
                    alpha=style.COMPARISON_SCATTER_ALPHA,
                    color=style.MEDIAN_COLOR,
                    rasterized=True,
                )
                lo, hi = (-4, 4) if task == "regression" else (0, 1)
                ax.plot([lo, hi], [lo, hi], color=style.GRID_COLOR)
                ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel=LABELS[a], ylabel=LABELS[b])
                ax.set_aspect("equal")
            for ax in axes[len(pairs[start : start + 3]) :]:
                ax.set_axis_off()
            fig.suptitle(f"{TASKS[task]} · {MODES[mode]} · paired outputs")
            yield (
                fig,
                f"scatter_{task}_{mode}_{page}",
                (
                    "Matched model outputs: positive-class probability for binary tasks, maximum class probability for multiclass, and expected direction for regression. "
                    "The diagonal indicates equal values. Scatter points are deterministically subsampled within each dataset/text group; all reported metrics use every input. "
                    "Equal multiclass confidence does not imply the same selected class."
                ),
            )
