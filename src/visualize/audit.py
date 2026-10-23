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
    axes[1].scatter(ordered.total_text_tokens / 1e6, y, color=style.TOTAL_COLOR,
                    s=style.AUDIT_MARKER_SIZE)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Full-dataset text tokens (millions; log scale)")
    from src.utils.config import load_config

    excluded = load_config("feature_creation")["dataset_selection"]["exclude"]
    labels = ordered.dataset + np.where(ordered.dataset_id.isin(excluded), " · removed", "")
    axes[0].set_yticks(y, labels)
    for index in np.flatnonzero(ordered.dataset_id.isin(excluded).to_numpy()):
        for ax in axes:
            ax.axhspan(index - 0.45, index + 0.45, color=style.FAINT_COLOR, zorder=-1)
    fig.suptitle("Typical text length and full-dataset burden")
    yield (
        fig,
        "text_burden",
        (
            "Combined-text median and 95th percentile per row (left), and total text volume (right; log scale) "
            f"for all 20 core datasets. Tokens use ceil(combined characters/{characters_per_token:g}); "
            "empty text contributes zero. Questions and task metadata are excluded. "
            "Shaded rows identify the two datasets removed from experiments. These are raw text "
            "volumes before reuse, not API billing counts."
        ),
    )
