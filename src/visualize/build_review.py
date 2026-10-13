"""Extraction costs and main feature counts, using measured dataset statistics."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from src.visualize import style


def extraction_figure(budget, price):
    table = budget.sort_values("wire_tokens_combined_approx")
    fig, axes = plt.subplots(1, 2, figsize=style.DASHBOARD_SIZE, sharey=True)
    y = np.arange(len(table))
    axes[0].barh(y, table.requests_for_combined_build, color=style.MEDIAN_COLOR)
    axes[1].barh(y, table.wire_tokens_combined_approx / 1e6, color=style.P95_COLOR)
    axes[0].set_yticks(y, table.dataset)
    axes[0].set_xlabel("Unique requests · per-column + joint")
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k"))
    axes[1].set_xlabel("Estimated input tokens (millions)")
    for i, tokens in enumerate(table.wire_tokens_combined_approx):
        label = f" ${tokens * price / 1e6:.2f}" if price is not None else ""
        axes[1].text(tokens / 1e6, i, label, va="center")
    axes[1].set_xlim(0, table.wire_tokens_combined_approx.max() / 1e6 * 1.25)
    fig.suptitle("Selected datasets: unique extraction workload and cost")
    return (
        fig,
        "extraction_by_dataset",
        f"Unique requests and approximate documented API-body tokens after exact reuse for the {len(table)} selected datasets. Both representations share cached requests. Costs use the configured input price; tokens use ceil(JSON characters/4), excluding unknown billing overhead and retries.",
    )


def regression_scale_figure(labels):
    fig, ax = plt.subplots(figsize=style.REGRESSION_RUBRIC_SIZE)
    ax.set_axis_off()
    ax.set_title("Regression: nine distinct evidence levels, not target units")
    for index, (level, wording) in enumerate(labels.items()):
        ax.text(
            0.02,
            0.94 - index * 0.1,
            f"{int(level):+d}",
            color=style.MEDIAN_COLOR
            if int(level) < 0
            else style.P95_COLOR
            if int(level) > 0
            else style.TEXT_COLOR,
            transform=ax.transAxes,
            va="top",
            weight="bold",
        )
        ax.text(
            0.1,
            0.94 - index * 0.1,
            wording,
            wrap=True,
            transform=ax.transAxes,
            va="top",
            fontsize=style.RUBRIC_FONT,
        )
    return (
        fig,
        "regression_evidence_levels",
        "The fixed nine-level regression rubric describes evidence for relatively lower or higher target values. Zero represents no directional evidence; nonzero levels differ in the specificity and strength of the supplied evidence. No target statistics or row targets define these levels.",
    )
