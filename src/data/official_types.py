"""Dependency-light implementation of MulTaBench's published text definition.

Source: MulTaBench 3bb95079a6146ed35d45187e98bcad5b063273a1,
multabench/preprocessing/feat_types.py::detect_text_features, _is_text_feature,
is_date_feature; numerical precheck: TabSTAR 1.1.15::is_numerical_feature,
is_mostly_numerical, is_numeric. See docs/DATA_SOURCES.md.

This follows the upstream algorithm, not a natural-language detector. We preserve CSV
column order instead of upstream set iteration. No upstream model package is imported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def valid_values(series: pd.Series) -> list:
    return [
        v
        for v in series
        if isinstance(v, str)
        or (not pd.isna(v) and (not isinstance(v, (int, float, np.number)) or np.isfinite(v)))
    ]


def is_numerical(series: pd.Series) -> bool:
    values = valid_values(series)
    if not values:
        return False
    if pd.api.types.is_numeric_dtype(series.dtype):
        return True
    unique = set(values)
    # This deliberately retains upstream's digit-only rule for numeric strings.
    return len(unique) > 50 and sum(not _numeric(v) for v in unique) <= 1


def _numeric(value) -> bool:
    if isinstance(value, str):
        return value.isdigit()
    try:
        float(value)
        return value is not None
    except (TypeError, ValueError):
        return False


def is_date(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return True
    if pd.api.types.is_numeric_dtype(series.dtype):
        return False
    values = valid_values(series)[:1000]
    if not values or not all(any(c in "-/:" or c.isalpha() for c in str(v)) for v in values):
        return False
    try:
        return (
            pd.to_datetime(pd.Series(values), errors="coerce", format="mixed").notna().mean()
            >= 0.99
        )
    except (TypeError, ValueError, OverflowError):
        return False


def official_text_columns(features: pd.DataFrame) -> list[str]:
    found = []
    for col in features:
        series = features[col]
        if pd.api.types.is_datetime64_any_dtype(series.dtype) or is_numerical(series):
            continue
        values = valid_values(series)
        unique = len(set(values))
        if values and (unique >= 100 or unique / len(values) >= 0.8) and not is_date(series):
            found.append(col)
    return found
