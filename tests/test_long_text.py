"""Quantiles, empty-input denominators, row identity and complete longest-cell storage."""

import numpy as np
import pandas as pd
import pytest

from src.data.audit import AuditCollection, _save_lengths, audit_dataset, request_lengths
from src.data.loaders import LoadedDataset
from src.data.long_text import long_text_audit
from src.data.metadata import extract_metadata
from src.utils.config import load_config


def test_long_text_quantiles_and_maxima_match_actual_nonempty_cells(tmp_path):
    cfg = load_config()
    frame = pd.DataFrame(
        {
            "text": ["x", "z" * 2000, None, "z" * 2000],
            "other": ["abc", None, " ", "abcdef"],
            "y": [0, 1, 0, 1],
        }
    )
    official = {"target": "y", "task_type": "cls", "num_classes": 2}
    meta = extract_metadata(
        frame, official, dataset_id="TEST_TEXT", dataset_name="Test", text_columns=["text", "other"]
    )
    dataset = LoadedDataset(frame, meta, official, {}, [])
    tables, details = audit_dataset(dataset, cfg)
    arrays = request_lengths(dataset)
    np.testing.assert_array_equal(arrays["characters_0"], [1, 2000, 0, 2000])
    _save_lengths(tmp_path / "TEST_TEXT_lengths.npz", arrays)
    collection = AuditCollection(
        **{
            key: pd.DataFrame([value] if key == "summary" else value)
            for key, value in tables.items()
        },
        details={"TEST_TEXT": details},
        folder=tmp_path,
    )
    datasets, columns, examples = long_text_audit(collection, cfg)
    first = columns.iloc[0]
    assert first.nonempty_cells == 3 and first.missing_or_empty_cells == 1
    assert first.characters_p99 == pytest.approx(np.quantile([1, 2000, 2000], 0.99))
    assert first.characters_max == 2000 and first.longest_row_id == "csv:1"
    assert datasets.iloc[0].joint_characters_max == 2006
    assert examples[0]["row_id"] == "csv:1"
    assert examples[0]["truncated_for_display"] is True
    assert len(examples[0]["text_preview"]) < examples[0]["characters"]
    assert (tmp_path / "long_text_columns.csv").is_file()


def test_numeric_official_text_and_empty_column_lengths_are_supported():
    cfg = load_config()
    frame = pd.DataFrame({"text": [123, np.nan, 4], "empty": [None, "", " "], "y": [0, 1, 0]})
    official = {"target": "y", "task_type": "cls", "num_classes": 2}
    meta = extract_metadata(
        frame, official, dataset_id="TEST_TEXT", dataset_name="Test", text_columns=["text", "empty"]
    )
    dataset = LoadedDataset(frame, meta, official, {}, [])
    arrays = request_lengths(dataset)
    np.testing.assert_array_equal(arrays["characters_0"], [5, 0, 3])  # CSV float dtype preserved.
    tables, details = audit_dataset(dataset, cfg)
    assert "empty" not in details["longest_text_examples"]
    assert tables["text"][1]["nonempty_tokens_p99"] is None
