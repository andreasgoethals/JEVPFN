"""Actual saved question-file footprint; no repeated dataset-dimension chart."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from src.visualize import style


def prepared_storage_figure(table):
    table = table.copy()
    table["size_mib"] = [Path(p).stat().st_size / 2**20 for p in table.path]
    table = table.sort_values("size_mib")
    fig, axes = plt.subplots(1, 2, figsize=style.DASHBOARD_SIZE, sharey=True)
    axes[0].barh(table.dataset, table.unique_inputs, color=style.MEDIAN_COLOR)
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}k"))
    axes[0].set_xlabel("Distinct active requests per extractor")
    bars = axes[1].barh(table.dataset, table.size_mib, color=style.TOTAL_COLOR)
    axes[1].bar_label(bars, labels=[f"{v:,.1f}" for v in table.size_mib], padding=3)
    axes[1].set_xlim(0, table.size_mib.max() * 1.2)
    axes[1].set_xlabel("SQLite file size (MiB; all saved versions)")
    fig.suptitle(f"{len(table)} frozen input files · {table.size_mib.sum():,.0f} MiB on disk")
    return (
        fig,
        "prepared_input_storage",
        (
            "Distinct requests in the active input set and measured SQLite file sizes for each selected dataset. "
            "Files also retain historical question versions and row-to-input links; size is therefore not "
            "solely the active text volume. These shared inputs are stored once for all extractors."
        ),
    )
