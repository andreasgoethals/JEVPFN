"""Dataset text burden: typical row size and total volume on one aligned figure."""

import matplotlib.pyplot as plt
import numpy as np

from src.visualize import style


def burden_figures(summary, characters_per_token=4.0):
    ordered = summary.sort_values("total_text_tokens")
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=style.DASHBOARD_SIZE)
    y = np.arange(len(ordered))
    for column, label, color, marker in [
        ("combined_tokens_median", "Median", style.MEDIAN_COLOR, "o"),
        ("combined_tokens_p95", "95th percentile", style.P95_COLOR, "x"),
    ]:
        axes[0].scatter(
            ordered[column], y, color=color, marker=marker, s=style.AUDIT_MARKER_SIZE, label=label
        )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Combined text tokens per row (log scale)")
    axes[0].legend(loc="lower right")
    bars = axes[1].barh(y, ordered.total_text_tokens / 1e6, color=style.TOTAL_COLOR)
    axes[1].bar_label(bars, labels=[f"{v / 1e6:.2f}" for v in ordered.total_text_tokens], padding=3)
    axes[1].set_xlim(0, ordered.total_text_tokens.max() / 1e6 * 1.15)
    axes[1].set_xlabel("Full-dataset text tokens (millions)")
    axes[0].set_yticks(y, ordered.dataset)
    fig.suptitle("Typical row length and total text volume before exact reuse")
    yield (
        fig,
        "text_burden",
        (
            "Combined-text median and 95th percentile per row (left), and total text volume (right) "
            f"for all 20 core datasets. Tokens use ceil(combined characters/{characters_per_token:g}); "
            "empty text contributes zero. Questions and task metadata are excluded. "
            "These are raw text volumes before reuse, not API billing counts."
        ),
    )
