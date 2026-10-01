"""Stable, strict JSON for requests, fingerprints and local audit files."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.files import atomic_text


def scalar(value):
    """Preserve types; missing/nonfinite CSV values become JSON null, never 'nan'."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported request value type: {type(value).__name__}")


def canonical_json(value) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> Path:
    """Atomic replacement prevents interrupted output from appearing complete."""
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return path


def text_value(value):
    """Treat null/nonfinite and whitespace-only cells as missing; preserve other text exactly."""
    value = scalar(value)
    return None if value is None or (isinstance(value, str) and not value.strip()) else value
