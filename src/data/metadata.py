"""A small label-free task schema, kept separate from descriptive target audits."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from src.utils.serialization import canonical_json, digest, scalar


@dataclass(frozen=True)
class TaskMetadata:
    dataset_id: str
    dataset_name: str
    target: str
    task_type: str
    classes: tuple
    text_columns: tuple[str, ...]
    non_text_columns: tuple[str, ...]

    def __post_init__(self):
        columns = self.text_columns + self.non_text_columns
        if not self.text_columns or len(columns) != len(set(columns)) or self.target in columns:
            raise ValueError("Text and non-text features must be disjoint and exclude the target.")
        if self.task_type not in {
            "binary_classification",
            "multiclass_classification",
            "regression",
        }:
            raise ValueError("Unknown task type.")
        if self.task_type == "regression" and self.classes:
            raise ValueError("Regression metadata cannot carry target values.")
        if self.task_type == "binary_classification" and len(self.classes) != 2:
            raise ValueError("Binary metadata requires exactly two class labels.")
        if self.task_type == "multiclass_classification" and len(self.classes) < 3:
            raise ValueError("Multiclass metadata requires at least three class labels.")

    def task_context(self) -> dict:
        # Explicit allowlist: frequencies, quantiles and the row target have no input slot.
        return {
            "dataset": self.dataset_name,
            "target": self.target,
            "task_type": self.task_type,
            "class_labels": list(self.classes),
        }

    @property
    def fingerprint(self) -> str:
        return digest(asdict(self))


def extract_metadata(
    frame: pd.DataFrame,
    official: dict,
    *,
    dataset_id: str,
    dataset_name: str,
    text_columns: list[str],
) -> TaskMetadata:
    """Only distinct classification labels are read; no target summaries enter this schema."""
    target = official["target"]
    if not frame.columns.is_unique or target not in frame:
        raise ValueError("Missing target or duplicate column names.")
    if not set(text_columns) <= set(frame.columns) - {target}:
        raise ValueError("An official text column is absent or is the target.")
    if official["task_type"] == "reg":
        if not pd.api.types.is_numeric_dtype(frame[target]):
            raise ValueError("Official regression target is not numerical.")
        task, classes = "regression", ()
    elif official["task_type"] == "cls":
        labels = [scalar(v) for v in frame[target].dropna().unique()]
        if any(v is None for v in labels):
            raise ValueError("Classification labels cannot be nonfinite.")
        classes = tuple(sorted(labels, key=canonical_json))
        if len(classes) < 2:
            raise ValueError("Classification requires at least two distinct labels.")
        if official.get("num_classes", len(classes)) != len(classes):
            raise ValueError("Observed class vocabulary disagrees with official metadata.")
        task = "binary_classification" if len(classes) == 2 else "multiclass_classification"
    else:
        raise ValueError(f"Unknown official task_type: {official['task_type']!r}")
    return TaskMetadata(
        dataset_id,
        dataset_name,
        target,
        task,
        classes,
        tuple(text_columns),
        tuple(c for c in frame if c != target and c not in text_columns),
    )
