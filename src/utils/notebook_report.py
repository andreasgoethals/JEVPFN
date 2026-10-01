"""Complete local notebook reports, including displayed tables, JSON and plotted data.

Reports can contain source examples and are always gitignored. Individual reports survive
partial reruns; rebuilding the combined report never requires rerunning the notebooks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import paths
from src.utils.files import atomic_text

_sections: list[str] = []


def begin_report() -> None:
    _sections.clear()


def report_section(title: str) -> None:
    _sections.append(f"\n{'=' * 88}\n{title}\n{'=' * 88}")


def record(value, title: str | None = None) -> None:
    if isinstance(value, pd.DataFrame):
        text = value.to_string(index=False, max_rows=None, max_cols=None, max_colwidth=None)
    elif isinstance(value, pd.Series):
        text = value.to_string(max_rows=None)
    elif isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, ensure_ascii=False, indent=2, default=_json_default)
    else:
        text = str(value)
    _sections.append((f"\n{title}\n" if title else "") + text)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def record_figure(fig, name: str, caption: str) -> None:
    """Preserve the numbers drawn by the project's lines, histograms, bars and error bars."""
    axes = []
    for ax in fig.axes:
        item = {
            "title": ax.get_title(),
            "x_label": ax.get_xlabel(),
            "y_label": ax.get_ylabel(),
            "x_scale": ax.get_xscale(),
            "y_scale": ax.get_yscale(),
            "x_limits": ax.get_xlim(),
            "y_limits": ax.get_ylim(),
            "x_tick_labels": [t.get_text() for t in ax.get_xticklabels()],
            "y_tick_labels": [t.get_text() for t in ax.get_yticklabels()],
            "annotations": [{"text": t.get_text(), "position": t.get_position()} for t in ax.texts],
            "lines": [
                {
                    "label": line.get_label(),
                    "x": np.asarray(line.get_xdata()),
                    "y": np.asarray(line.get_ydata()),
                }
                for line in ax.lines
            ],
            "patches": [],
            "collections": [],
        }
        for patch in ax.patches:
            if hasattr(patch, "get_height"):
                item["patches"].append(
                    {
                        "x": patch.get_x(),
                        "y": patch.get_y(),
                        "width": patch.get_width(),
                        "height": patch.get_height(),
                    }
                )
            else:
                item["patches"].append({"vertices": patch.get_path().vertices})
        for collection in ax.collections:
            item["collections"].append(
                {"segments": collection.get_segments()}
                if hasattr(collection, "get_segments")
                else {"offsets": collection.get_offsets().tolist()}
            )
        item["images"] = [np.asarray(im.get_array()) for im in ax.images]
        item["tables"] = [
            {str(key): cell.get_text().get_text() for key, cell in table.get_celld().items()}
            for table in ax.tables
        ]
        axes.append(item)
    record({"caption": caption, "axes": axes}, f"Figure: {name}")


def finish_report(notebook: str, summary: str = "") -> None:
    """Print the complete report in the last cell and save it for interactive/parallel use."""
    from src.utils.locking import file_lock
    from src.utils.run_notebooks import discover, write_all_results, write_captions

    text = f"{notebook}\n" + "\n\n".join(_sections) + "\n\n" + summary
    print(text)
    atomic_text(paths.reports_dir(notebook), text)
    # Independent notebook processes finish concurrently. Protect aggregate replacement on Windows.
    with file_lock(paths.outputs_dir() / ".reports.lock"):
        names = discover()
        write_captions(names)
        write_all_results(names)
