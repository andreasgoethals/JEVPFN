"""Display rich objects and retain their full contents in ignored local reports."""

from __future__ import annotations

import html
import json


def display_local(*objects) -> None:
    import pandas as pd
    from IPython import get_ipython
    from IPython.display import HTML, display
    from matplotlib.figure import Figure

    from src.utils.notebook_report import record

    for obj in objects:
        if not isinstance(obj, Figure):
            record(obj)
        if get_ipython() is not None:
            if isinstance(obj, pd.DataFrame) and isinstance(obj.index, pd.RangeIndex):
                display(HTML(obj.to_html(index=False, max_rows=40, max_cols=30, escape=True)))
            else:
                display(obj)


def display_json(title: str, value) -> None:
    from IPython.display import HTML

    body = html.escape(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    from IPython import get_ipython
    from IPython.display import display

    from src.utils.notebook_report import record

    record(value, title)
    if get_ipython() is not None:
        display(
            HTML(f"<details><summary>{html.escape(title)}</summary><pre>{body}</pre></details>")
        )
