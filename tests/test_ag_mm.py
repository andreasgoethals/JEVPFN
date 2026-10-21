"""Train-only/offline contracts against a fake publisher interface; no AG-MM import."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.evaluation.models import FoldInput, _backend
from src.utils.serialization import file_sha256


@pytest.mark.parametrize(
    "task", ["binary_classification", "multiclass_classification", "regression"]
)
def test_agmm_fixed_recipe_official_text_and_class_alignment(monkeypatch, tmp_path, task):
    import sys

    calls = {}

    class Predictor:
        def __init__(self, **kwargs):
            calls["constructor"] = kwargs
            self.class_labels = ["z", "a"]

        def fit(self, **kwargs):
            calls["fit"] = kwargs

        def predict_proba(self, x, **kwargs):
            calls["test"] = x.copy()
            assert kwargs == {"as_pandas": True, "as_multiclass": True}
            return pd.DataFrame({"a": [0.2] * len(x), "z": [0.8] * len(x)})

        def predict(self, x, **kwargs):
            return np.zeros(len(x))

    monkeypatch.setitem(
        sys.modules, "autogluon.multimodal", SimpleNamespace(MultiModalPredictor=Predictor)
    )
    encoder_dir = tmp_path / "weights"
    encoder_dir.mkdir()
    asset = encoder_dir / "config.json"
    asset.write_text("{}")
    settings = dict(
        seed=12,
        device="cuda",
        threads=2,
        parameters={},
        text_checkpoint_path=str(encoder_dir),
        assets=[dict(path=str(asset), sha256=file_sha256(asset))],
    )
    x = pd.DataFrame(
        {
            "text": [None, "genuine text", "other"],
            "code": ["12", "10", "12"],
            "value": [1.0, 2.0, np.nan],
        },
        index=[3, 8, 9],
    )
    encoder = FoldInput().fit(x)
    encoder.text_columns_ = ["text"]
    backend = _backend("ag_mm", task, settings, encoder, len(x))
    workspace = backend.workspace.name
    try:
        y = pd.Series([0.1, 0.2, 0.3] if task == "regression" else ["z", "a", "z"], index=x.index)
        backend.fit(x, y)
        fit = calls["fit"]
        assert fit["train_data"].index.tolist() == x.index.tolist()
        assert fit["train_data"][backend.label].tolist() == y.tolist()
        assert fit["train_data"].text.tolist() == ["", "genuine text", "other"]
        assert fit["tuning_data"] is None and fit["hyperparameter_tune_kwargs"] is None
        assert fit["column_types"] == {
            "text": "text",
            "code": "categorical",
            "value": "numerical",
            backend.label: "numerical" if task == "regression" else "categorical",
        }
        assert calls["constructor"]["presets"] == "medium_quality"
        assert calls["constructor"]["use_ensemble"] is False
        assert (
            str(encoder_dir)
            == calls["constructor"]["hyperparameters"]["model.hf_text.checkpoint_name"]
        )
        test = x.iloc[:1].copy()
        test["text"] = "unseen test only"
        if task != "regression":
            np.testing.assert_allclose(backend.predict_proba(test), [[0.8, 0.2]])
            assert backend.label not in calls["test"]
        assert backend.predict(test).shape == (1,)
        assert "unseen test only" not in str(fit["train_data"].to_dict())
    finally:
        backend.workspace.cleanup()
    from pathlib import Path

    assert not Path(workspace).exists()
    (encoder_dir / "unreviewed.bin").write_bytes(b"not reviewed")
    with pytest.raises(ValueError, match="Every AG-MM encoder file"):
        _backend("ag_mm", task, settings, encoder, len(x))
