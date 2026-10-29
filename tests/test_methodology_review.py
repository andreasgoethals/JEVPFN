"""Regression cases for representation and evaluation boundary consistency."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data.text_embeddings import MODEL_NAMESPACES, encoder_config, input_rows
from src.evaluation.features import add_text_presence
from src.evaluation.protocol import shared_splits
from src.utils.serialization import text_value


def test_text_presence_shares_request_missingness_without_erasing_literal_strings():
    values = [None, pd.NA, np.nan, np.inf, -np.inf, " \t", 0, 0.0, "nan", "inf", "two words"]
    dataset = SimpleNamespace(
        frame=pd.DataFrame({"text": values}),
        metadata=SimpleNamespace(text_columns=("text",)),
    )
    matrix = add_text_presence(pd.DataFrame(index=dataset.frame.index), dataset)
    expected_missing = [int(text_value(v) is None) for v in values]
    assert expected_missing == [1] * 6 + [0] * 5
    assert matrix.iloc[:, 0].tolist() == expected_missing
    import json
    import zlib

    unique, columns = input_rows(dataset)
    keys = json.loads(zlib.decompress(columns[0][1]))
    assert [int(key is None) for key in keys] == expected_missing
    assert set(unique.values()) == {"0", "0.0", "nan", "inf", "two words"}


@pytest.mark.parametrize("model,encoder", MODEL_NAMESPACES.items())
def test_encoder_provenance_points_to_selected_checkpoint(model, encoder):
    cfg = encoder_config(encoder)
    assert cfg["model"] == model
    assert cfg["source"] == f"https://huggingface.co/{model}"


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_nonfinite_regression_targets_fail_before_split_or_model_execution(value):
    dataset = SimpleNamespace(
        frame=pd.DataFrame({"target": [1.0, 2.0, value, 4.0, 5.0]}),
        metadata=SimpleNamespace(target="target", task_type="regression"),
    )
    with pytest.raises(ValueError, match="Regression targets must be finite"):
        shared_splits(dataset)
