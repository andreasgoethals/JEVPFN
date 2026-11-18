"""Label-free text-field taxonomy for RQ2-E moderators, fixed before any result.

MulTaBench reviewers asked to distinguish short categorical strings, IDs, names,
long descriptions and reviews. Each text column gets one type from its non-empty
values only (never the target), using fixed, documented thresholds:

1. numeric_text: at least 90% of values parse as numbers after removing , $ % and spaces.
2. categorical: at most 50 distinct values, or fewer than 5% distinct values.
3. identifier_or_code: at least 90% of values contain no whitespace and median length <= 16.
4. short_text: median length below 60 characters.
5. long_text: everything else.
"""

from __future__ import annotations

import pandas as pd

FIELD_TYPES = ("numeric_text", "categorical", "identifier_or_code", "short_text", "long_text")
NUMERIC_SHARE = 0.9
CATEGORICAL_MAX_DISTINCT = 50
CATEGORICAL_MAX_RATIO = 0.05
CODE_NO_SPACE_SHARE = 0.9
CODE_MAX_MEDIAN_LENGTH = 16
SHORT_MAX_MEDIAN_LENGTH = 60


def classify_text_column(values: pd.Series) -> dict:
    text = values.astype("string").str.strip()
    text = text[text.notna() & text.ne("")]
    if text.empty:
        return {"field_type": "empty", "nonempty": 0, "distinct": 0, "median_chars": 0.0}
    distinct = int(text.nunique())
    median = float(text.str.len().median())
    cleaned = text.str.replace(r"[,$%\s]", "", regex=True)
    numeric = pd.to_numeric(cleaned, errors="coerce").notna().mean()
    if numeric >= NUMERIC_SHARE:
        kind = "numeric_text"
    elif distinct <= CATEGORICAL_MAX_DISTINCT or distinct / len(text) < CATEGORICAL_MAX_RATIO:
        kind = "categorical"
    elif (~text.str.contains(r"\s", regex=True)).mean() >= CODE_NO_SPACE_SHARE and median <= CODE_MAX_MEDIAN_LENGTH:
        kind = "identifier_or_code"
    elif median < SHORT_MAX_MEDIAN_LENGTH:
        kind = "short_text"
    else:
        kind = "long_text"
    return {"field_type": kind, "nonempty": int(len(text)), "distinct": distinct,
            "distinct_ratio": distinct / len(text), "median_chars": median,
            "numeric_share": float(numeric)}


def dataset_field_types(dataset) -> pd.DataFrame:
    """One row per text column of a loaded dataset; the target is never read."""
    frame = dataset.features
    rows = [{"dataset_id": dataset.metadata.name if hasattr(dataset.metadata, "name") else None,
             "field": str(column), **classify_text_column(frame[column])}
            for column in dataset.metadata.text_columns]
    return pd.DataFrame(rows)


def field_type_table(entries=None) -> pd.DataFrame:
    """Taxonomy for the selected benchmark datasets (default: Experiment 1 selection)."""
    from src.data.loaders import load_catalog, load_dataset
    from src.data.selection import selected_entries
    from src.utils.config import load_yaml

    entries = entries or selected_entries(load_catalog(), load_yaml("experiment_1/main"))
    tables = []
    for entry in entries:
        table = dataset_field_types(load_dataset(entry, allow_download=False))
        table["dataset_id"] = entry["id"]
        tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
