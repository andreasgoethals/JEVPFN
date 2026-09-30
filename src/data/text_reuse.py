"""Exact text-input reuse, without targets, prompts or model inference.

Equality follows the request builder's typed JSON and missing-text conventions.
Column names and task metadata remain part of the request: equal values in
different fields or datasets are not automatically interchangeable.
"""

from __future__ import annotations

import pandas as pd

from src.utils.serialization import canonical_json, text_value


def semantic_codes(features: pd.DataFrame) -> pd.DataFrame:
    """Encode a column at a time without retaining a second copy of long strings."""
    return pd.DataFrame(
        {
            col: pd.factorize(features[col].map(lambda v: canonical_json(text_value(v))))[0]
            for col in features
        },
        index=features.index,
    )


def text_reuse_profile(features: pd.DataFrame, *, dataset_id: str, top_n: int = 5) -> dict:
    """Profile per-field and joint duplicates under one fixed question/model/config.

    A joint input omits missing fields. A joint input with just one available field
    therefore reuses that field's per-column response. No approximate matching,
    whitespace trimming of nonempty strings, or cross-field merging is performed.
    """
    if not len(features.columns):
        raise ValueError("Text reuse requires at least one official text column.")
    codes = semantic_codes(features)
    present = features.apply(lambda s: s.map(text_value).notna())
    rows = len(features)
    records = []
    groups = [("per_column", [col]) for col in features] + [("joint", list(features))]
    overlap = 0
    for mode, columns in groups:
        availability = present[columns].sum(axis=1).to_numpy()
        active = availability > 0
        selected = codes.loc[active, columns]
        counts = selected.value_counts(sort=False).to_numpy()
        nonempty, unique = int(active.sum()), len(counts)
        keep = active & ~codes[columns].duplicated().to_numpy()
        if mode == "joint":
            overlap = int((keep & (availability == 1)).sum())
        records.append(
            {
                "dataset_id": dataset_id,
                "mode": mode,
                "text_columns": columns,
                "rows": rows,
                "nonempty_rows": nonempty,
                "missing_or_empty_rows": rows - nonempty,
                "missing_or_empty_pct": 100 * (rows - nonempty) / rows if rows else 0.0,
                "unique_nonempty_inputs": unique,
                "unique_nonempty_ratio": unique / nonempty if nonempty else 0.0,
                "repeated_nonempty_rows": nonempty - unique,
                "reuse_savings_pct": 100 * (nonempty - unique) / nonempty if nonempty else 0.0,
                "repeated_distinct_inputs": int((counts > 1).sum()),
                "rows_in_repeated_groups": int(counts[counts > 1].sum()),
                "most_common_input_count": int(counts.max()) if unique else 0,
                "top_input_frequencies": sorted((int(v) for v in counts), reverse=True)[:top_n],
            }
        )
    per_column, joint = records[:-1], records[-1]
    per_nonempty = sum(r["nonempty_rows"] for r in per_column)
    per_unique = sum(r["unique_nonempty_inputs"] for r in per_column)
    slots = rows * len(features.columns)
    distinct_both = per_unique + joint["unique_nonempty_inputs"] - overlap
    logical_both = per_nonempty + joint["nonempty_rows"]
    summary = {
        "per_column_nonempty_inputs": per_nonempty,
        "per_column_unique_inputs": per_unique,
        "per_column_repeated_inputs": per_nonempty - per_unique,
        "per_column_missing_or_empty_pct": 100 * (slots - per_nonempty) / slots if slots else 0.0,
        "per_column_reuse_savings_pct": 100 * (per_nonempty - per_unique) / per_nonempty
        if per_nonempty
        else 0.0,
        "joint_nonempty_inputs": joint["nonempty_rows"],
        "joint_unique_inputs": joint["unique_nonempty_inputs"],
        "joint_unique_ratio": joint["unique_nonempty_ratio"],
        "joint_repeated_inputs": joint["repeated_nonempty_rows"],
        "joint_reuse_savings_pct": joint["reuse_savings_pct"],
        "joint_missing_or_empty_pct": joint["missing_or_empty_pct"],
        "joint_single_field_unique_overlap": overlap,
        "combined_nonempty_inputs": logical_both,
        "combined_unique_inputs": distinct_both,
        "combined_reuse_savings_pct": 100 * (logical_both - distinct_both) / logical_both
        if logical_both
        else 0.0,
    }
    return {"summary": summary, "groups": records}
