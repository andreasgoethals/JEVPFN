"""Shared decision-model labels for figures of saved features.

The earlier comparison pages (coverage, pairs, consensus, distributions, examples) were retired
on 23 November 2026; A1 renders the compact export through `src.visualize.outputs`.
"""

from __future__ import annotations

from src.visualize import style

LABELS = style.DECISION_SHORT
TASKS = {
    "binary_classification": "Binary",
    "multiclass_classification": "Multiclass",
    "regression": "Regression",
}
MODES = {"per_column": "Separate text columns", "joint": "Joint text"}
