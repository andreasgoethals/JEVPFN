"""Compact burden figures at the template's final A4 dimensions."""

import matplotlib.pyplot as plt

from src.visualize import style


def burden_figures(summary, characters_per_token: float = 4.0):
    ordered = summary.sort_values("combined_tokens_p95")
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, style.AUDIT_RATIO))
    ax.scatter(
        ordered.combined_tokens_median,
        ordered.dataset,
        color=style.MEDIAN_COLOR,
        s=style.AUDIT_MARKER_SIZE,
        label="Median",
    )
    ax.scatter(
        ordered.combined_tokens_p95,
        ordered.dataset,
        color=style.P95_COLOR,
        s=style.AUDIT_MARKER_SIZE,
        label="95th percentile",
        marker="x",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Approximate combined text tokens per row (log scale)")
    ax.legend()
    caption = (
        "Median and 95th percentile of combined text tokens per row for the 20 core text "
        "datasets. Tokens are approximated by the ceiling of the total text characters "
        f"per row divided by {characters_per_token:g}; missing text contributes zero. "
        "The horizontal axis is logarithmic."
    )
    yield fig, "row_text_burden", caption
    ordered = summary.sort_values("total_text_tokens")
    fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_FULL, style.AUDIT_RATIO))
    ax.barh(ordered.dataset, ordered.total_text_tokens / 1e6, color=style.TOTAL_COLOR)
    ax.set_xlabel("Approximate full-dataset text tokens (millions)")
    caption = (
        "Total approximate text tokens across all rows in each of the 20 core text datasets. "
        f"The totals sum the ceiling of combined text characters per row divided by {characters_per_token:g}; "
        "task metadata, questions and non-text features are excluded."
    )
    yield fig, "total_text_burden", caption
