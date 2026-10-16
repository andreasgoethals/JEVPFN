"""Conditional reporting uses raw text and frozen predictions, never new splits or fits."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data.metadata import extract_metadata
from src.evaluation.text_overlap import cohort_scores, fold_overlap, text_identities


def dataset():
    frame = pd.DataFrame(
        {
            "a": ["known", "novel", "known", "never", " ", "known "],
            "b": ["old", "new", "new", "fresh", None, "old"],
            "y": [0, 1, 0, 1, 0, 1],
            "numeric": [1, 2, 3, 4, 5, 6],
        }
    )
    meta = extract_metadata(
        frame,
        {"task_type": "cls", "target": "y"},
        dataset_id="fixture",
        dataset_name="Fixture",
        text_columns=["a", "b"],
    )
    return SimpleNamespace(frame=frame, metadata=meta)


def test_joint_novelty_is_different_from_individual_column_novelty():
    data = dataset()
    flags = fold_overlap(text_identities(data), [0, 1], [2, 3, 4, 5])
    assert flags.joint_text_seen_in_train.tolist() == [False] * 4
    assert flags.any_text_column_seen_in_train.tolist() == [True, False, False, True]
    assert flags.text_present.tolist() == [True, True, False, True]
    assert flags.index.tolist() == ["csv:2", "csv:3", "csv:4", "csv:5"]


def test_overlap_identity_ignores_targets_non_text_and_preserves_whitespace():
    data = dataset()
    before = text_identities(data)
    data.frame["y"] = 999
    data.frame["numeric"] = None
    pd.testing.assert_frame_equal(before, text_identities(data))
    assert before.joint.iloc[0] != before.joint.iloc[5]


@pytest.mark.parametrize(
    "task", ["binary_classification", "multiclass_classification", "regression"]
)
def test_all_task_metrics_use_same_predictions_and_fixed_threshold(task):
    data = dataset()
    meta = replace(
        data.metadata,
        task_type=task,
        classes=()
        if task == "regression"
        else ((0, 1) if task.startswith("binary") else (0, 1, 2)),
    )
    flags = fold_overlap(text_identities(data), [0, 1], [2, 3, 4, 5])
    p = (
        np.array([0.3, 0.8, 0.6, 0.1])
        if task == "regression"
        else np.full((4, len(meta.classes)), 1 / len(meta.classes))
    )
    report = cohort_scores(
        meta, [0, 1, 0, 1], p, flags, selected_threshold=0.8 if task.startswith("binary") else None
    )
    cohorts = report["cohorts"]
    assert cohorts["seen_joint"]["status"] == "no_eligible_rows"
    assert cohorts["unseen_joint"]["n_observations"] == 3
    assert cohorts["all_columns_unseen"]["n_observations"] == 1
    assert cohorts["empty_text"]["n_observations"] == 1
    assert cohorts["all_columns_unseen"]["fraction_of_outer_test"] == 0.25
    if task == "binary_classification":
        assert cohorts["unseen_joint"]["decisions"]["validation_selected"]["threshold"] == 0.8
    elif task == "regression":
        assert cohorts["all_columns_unseen"]["metrics"]["r2"] is None
    else:
        assert cohorts["all_columns_unseen"]["metrics"]["roc_auc_ovr_macro"] is None


def test_exact_duplicate_joint_can_be_seen_without_leaking_test_identity():
    data = dataset()
    data.frame.loc[2, ["a", "b"]] = data.frame.loc[0, ["a", "b"]]
    flags = fold_overlap(text_identities(data), [0], [1, 2, 3, 4, 5])
    assert flags.joint_text_seen_in_train.tolist() == [False, True, False, False, False]
    with pytest.raises(ValueError, match="disjoint"):
        fold_overlap(text_identities(data), [0, 1], [1, 2])
