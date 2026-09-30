"""Reuse statistics must match paid-request identity, including absent joint fields."""

import numpy as np
import pandas as pd
import pytest

from src.data.loaders import LoadedDataset
from src.data.metadata import extract_metadata
from src.data.text_reuse import text_reuse_profile
from src.jev.requests import build_requests
from src.utils.config import load_config


@pytest.mark.parametrize(
    "task,labels",
    [
        ("cls", [0, 1, 0, 1, 0, 1, 0, 1]),
        ("cls", ["a", "b", "c", "a", "b", "c", "a", "a"]),
        ("reg", [1.1, 2.2, 1.1, 0, 3, 4, 5, 6]),
    ],
)
def test_reuse_matches_full_request_keys(task, labels):
    frame = pd.DataFrame(
        {
            "a": ["same", "same", "same", None, " ", 1, "1", "same "],
            "b": ["text", "text", None, "text", np.nan, None, None, "text"],
            "number": range(8),
            "target": labels,
        }
    )
    official = {"target": "target", "task_type": task}
    metadata = extract_metadata(
        frame, official, dataset_id="test", dataset_name="Test", text_columns=["a", "b"]
    )
    dataset = LoadedDataset(frame, metadata, official, {}, [])
    profile = text_reuse_profile(dataset.features[["a", "b"]], dataset_id="test")
    cfg = load_config("feature_creation/default")
    requests = [
        r for i in range(len(frame)) for r in build_requests(dataset.features, metadata, i, cfg)
    ]
    summary = profile["summary"]
    assert summary["combined_nonempty_inputs"] == len(requests)
    assert summary["combined_unique_inputs"] == len({r.cache_key for r in requests})
    assert summary["joint_unique_inputs"] == 6
    assert summary["joint_single_field_unique_overlap"] == 4
    assert profile["groups"][0]["unique_nonempty_inputs"] == 4  # Preserve type and whitespace.
    assert profile["groups"][0]["top_input_frequencies"] == [3, 1, 1, 1]
    assert profile["groups"][0]["missing_or_empty_rows"] == 2


def test_single_field_modes_coincide_and_all_empty_inputs_cost_nothing():
    profile = text_reuse_profile(pd.DataFrame({"text": ["x", "x", None]}), dataset_id="x")
    assert profile["summary"]["combined_unique_inputs"] == 1
    assert profile["summary"]["joint_single_field_unique_overlap"] == 1
    empty = text_reuse_profile(pd.DataFrame({"text": [None, " \t", float("inf")]}), dataset_id="x")
    assert empty["summary"]["combined_unique_inputs"] == 0
    assert empty["summary"]["joint_missing_or_empty_pct"] == 100
    assert empty["groups"][0]["top_input_frequencies"] == []
