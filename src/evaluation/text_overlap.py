"""Evaluation-only text overlap diagnostics; never influence prompts, splits or fitting."""

import numpy as np
import pandas as pd

from src.evaluation.metrics import classification_metrics, regression_metrics
from src.utils.serialization import digest, text_value

OVERLAP_VERSION = "exact_named_text_overlap_v1"


def text_identities(dataset):
    """Hash exact named nonempty text, independently of labels and question versions.

    Equal joint IDs use the same equality as prepared inputs within a dataset. A
    joint combination may be new while some of its individual text columns are familiar.
    Empty text is not a semantic identity and is classified separately.
    """
    columns = dataset.metadata.text_columns
    records = []
    for row in dataset.frame.loc[:, list(columns)].itertuples(index=False, name=None):
        text = {
            col: value for col, raw in zip(columns, row) if (value := text_value(raw)) is not None
        }
        records.append(
            {
                "joint": digest(text) if text else None,
                **{
                    f"column_{i}": digest({col: text[col]}) if col in text else None
                    for i, col in enumerate(columns)
                },
            }
        )
    return pd.DataFrame(records, index=[f"csv:{i}" for i in range(len(dataset.frame))])


def fold_overlap(identities, train, test):
    """Return flags for outer-test rows against that fold's training rows only."""
    if (
        set(train) & set(test)
        or len(set(train)) != len(train)
        or len(set(test)) != len(test)
        or any(
            type(i) not in (int, np.int32, np.int64) or not 0 <= i < len(identities)
            for i in [*train, *test]
        )
    ):
        raise ValueError("Overlap diagnostics need valid disjoint train/test positions")
    left, right = identities.iloc[train], identities.iloc[test]
    flags = pd.DataFrame(index=right.index)
    flags["text_present"] = right.joint.notna()
    flags["joint_text_seen_in_train"] = (
        right.joint.isin(set(left.joint.dropna())) & flags.text_present
    )
    flags["any_text_column_seen_in_train"] = False
    for column in identities.columns.drop("joint"):
        flags["any_text_column_seen_in_train"] |= (
            right[column].isin(set(left[column].dropna())) & right[column].notna()
        )
    return flags


def cohort_scores(metadata, y_true, predictions, flags, *, selected_threshold=None):
    """Rescore already-made predictions; thresholds remain fixed, with no extra fit."""
    y, prediction = np.asarray(y_true), np.asarray(predictions)
    if len(flags) != len(y) or len(prediction) != len(y):
        raise ValueError("Text-overlap flags and predictions must align")
    masks = {
        "seen_joint": flags.text_present & flags.joint_text_seen_in_train,
        "unseen_joint": flags.text_present & ~flags.joint_text_seen_in_train,
        "all_columns_unseen": flags.text_present & ~flags.any_text_column_seen_in_train,
        "empty_text": ~flags.text_present,
    }
    result = {}
    for name, mask in masks.items():
        chosen = mask.to_numpy(dtype=bool)
        count = int(chosen.sum())
        if not count:
            scores = {
                "n_observations": 0,
                "metrics": {},
                "undefined": {},
                "status": "no_eligible_rows",
                "reason": "No outer-test rows in this text cohort",
            }
        elif metadata.task_type == "regression":
            scores = regression_metrics(y[chosen], prediction[chosen])
        else:
            scores = classification_metrics(
                y[chosen],
                prediction[chosen],
                classes=metadata.classes,
                positive_class=metadata.classes[-1]
                if metadata.task_type == "binary_classification"
                else None,
                selected_threshold=selected_threshold,
            )
        result[name] = {**scores, "fraction_of_outer_test": count / len(y) if len(y) else 0.0}
    return {
        "version": OVERLAP_VERSION,
        "cohorts": result,
        "interpretation": "Conditional diagnostics on unchanged outer-test predictions; unseen joint text may still contain familiar individual columns. No split or threshold is changed.",
    }
