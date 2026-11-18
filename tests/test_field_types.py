"""Label-free RQ2-E moderators are fixed before results and never read targets."""

import numpy as np
import pandas as pd

from src.evaluation.field_types import classify_text_column
from src.evaluation.row_benefit import prediction_shift_bands


def test_field_taxonomy_rules_in_priority_order():
    assert classify_text_column(pd.Series(["1,200", "$35", "40%", "7"] * 30))["field_type"] == "numeric_text"
    assert classify_text_column(pd.Series(["Ohio", "Utah", "Iowa"] * 100))["field_type"] == "categorical"
    codes = pd.Series([f"Z{i:05d}" for i in range(400)])
    assert classify_text_column(codes)["field_type"] == "identifier_or_code"
    names = pd.Series([f"The great title number {i}" for i in range(400)])
    assert classify_text_column(names)["field_type"] == "short_text"
    reviews = pd.Series([("word " * 30) + str(i) for i in range(400)])
    assert classify_text_column(reviews)["field_type"] == "long_text"
    assert classify_text_column(pd.Series(["", None, "  "]))["field_type"] == "empty"


def test_prediction_shift_uses_predictions_only():
    ref = pd.DataFrame({"p_c000": [0.5, 0.9, 0.5], "p_c001": [0.5, 0.1, 0.5]})
    aug = pd.DataFrame({"p_c000": [0.49, 0.6, 0.0], "p_c001": [0.51, 0.4, 1.0]})
    bands = prediction_shift_bands(ref, aug, ["a", "b"], np.nan)
    assert bands.tolist() == ["shift_0", "shift_3", "shift_3"]
    reg_ref = pd.DataFrame({"prediction": [1.0, 1.0, 1.0]})
    reg_aug = pd.DataFrame({"prediction": [1.0, 1.1, 2.0]})
    assert prediction_shift_bands(reg_ref, reg_aug, [], 1.0).tolist() == ["shift_0", "shift_1", "shift_3"]
    assert set(prediction_shift_bands(reg_ref, reg_aug, [], 0.0)) == {"undefined"}
