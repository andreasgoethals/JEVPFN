"""Real data-boundary and orchestration checks; no checkpoint imports or ML fits."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data.loaders import LoadedDataset
from src.data.metadata import extract_metadata
from src.evaluation.campaign import benchmark_plan, run_arm
from src.evaluation.features import original_features
from src.evaluation.models import (
    MAIN_MODELS,
    TEXT_MODELS,
    FoldInput,
    PredictorAdapter,
    _backend,
    _limix_config,
    estimator_factory,
    verify_runtime,
)
from src.utils.config import load_yaml


@pytest.fixture
def dataset():
    frame = pd.DataFrame(
        {
            "amount": [1.0, np.nan, 4.0],
            "category": ["x", None, "x"],
            "text": ["first", "", "last"],
            "target": ["b", "a", "b"],
        }
    )
    official = {"target": "target", "task_type": "cls", "num_classes": 2}
    meta = extract_metadata(
        frame, official, dataset_id="fixture", dataset_name="Fixture", text_columns=["text"]
    )
    return LoadedDataset(frame, meta, official, {"data_identity": "fixture"}, [])


def test_category_vocabulary_is_train_only_and_missing_stays_missing():
    train = pd.DataFrame({"n": [1, np.nan], "c": ["train", None]})
    test = pd.DataFrame({"n": [999, np.nan], "c": ["test_only", "train"]})
    encoder = FoldInput().fit(train)
    assert encoder.vocabulary_["c"] == {'"train"': 0}
    result = encoder.transform(test)
    assert np.isnan(result.loc[0, "c"]) and result.loc[1, "c"] == 0
    assert np.isnan(result.loc[1, "n"])
    assert "test_only" not in str(encoder.vocabulary_)
    assert encoder.transform(train, strings=True).loc[1, "c"] == "null"


@pytest.mark.parametrize("model_id", MAIN_MODELS + TEXT_MODELS)
@pytest.mark.parametrize(
    "task", ["binary_classification", "multiclass_classification", "regression"]
)
def test_adapters_preserve_all_rows_classes_and_original_text_only_for_baselines(
    dataset, model_id, task
):
    if task == "multiclass_classification":
        dataset.frame["target"] = ["a", "b", "c"]
        dataset.metadata = replace(dataset.metadata, task_type=task, classes=("a", "b", "c"))
    elif task == "regression":
        dataset.frame["target"] = [0.2, 0.4, 0.8]
        dataset.metadata = replace(dataset.metadata, task_type=task, classes=())
    calls = []

    class Backend:
        def fit(self, x, y):
            assert len(x) == len(dataset.frame) == len(y)
            assert list(x.index) == list(y.index)
            if model_id in TEXT_MODELS:
                assert x["text"].tolist() == dataset.frame["text"].tolist()
                assert y.tolist() == dataset.frame["target"].tolist()
                self.classes_ = list(reversed(dataset.metadata.classes))
            else:
                assert "text" not in x
                self.classes_ = list(reversed(range(len(dataset.metadata.classes))))
            calls.append(len(x))

        def predict_proba(self, x):
            # Reversed class vocabulary: only its first column is one.
            p = np.zeros((len(x), len(self.classes_)))
            p[:, 0] = 1
            return p

        def predict(self, *args, **kwargs):
            if model_id == "limix_2":
                assert len(args[0]) == len(args[1]) == len(dataset.frame)
                assert kwargs["task_type"] == (
                    "Regression" if task == "regression" else "Classification"
                )
                return (
                    np.zeros(len(args[2]))
                    if task == "regression"
                    else np.tile(
                        [1 / len(dataset.metadata.classes)] * len(dataset.metadata.classes),
                        (len(args[2]), 1),
                    )
                )
            return np.zeros(len(args[0]))

    features = original_features(dataset)
    if model_id in MAIN_MODELS:
        features = features.drop(columns="text")
    adapter = PredictorAdapter(
        model_id, dataset.metadata, {}, backend_loader=lambda *args: Backend()
    )
    adapter.fit(features, dataset.frame.target)
    assert adapter.n_training_rows_ == len(features)
    if task == "regression":
        assert adapter.predict(features).shape == (3,)
    else:
        p = adapter.predict_proba(features)
        assert p.shape == (3, len(dataset.metadata.classes))
        if model_id != "limix_2":
            assert (p[:, -1] == 1).all()


def test_no_raw_text_in_main_arm_and_no_model_import_without_permission(dataset):
    adapter = PredictorAdapter(
        "linear", dataset.metadata, {}, backend_loader=lambda *args: pytest.fail("must not load")
    )
    with pytest.raises(ValueError, match="Original text"):
        adapter.fit(dataset.features, dataset.frame.target)
    with pytest.raises(RuntimeError, match="disabled"):
        estimator_factory("tabpfn_3_5", dataset.metadata, {})


def test_plan_eight_arms_six_engines_three_original_text_baselines():
    plan = benchmark_plan()
    assert len({a["dataset_id"] for a in plan}) == 18
    assert len(plan) == 5994
    for model in MAIN_MODELS:
        row = [a for a in plan if a["dataset_id"] == plan[0]["dataset_id"] and a["model"] == model]
        assert len([a for a in row if a["provider"] is None]) == 22
        for provider in load_yaml("extractors")["candidates"]:
            assert len([a for a in row if a["provider"] == provider]) == 3
    text = [a for a in plan if a["model"] in TEXT_MODELS]
    assert len(text) == 54
    assert len([a for a in text if a["provider"] is None]) == 54
    assert "ag_mm" in {a["model"] for a in text}
    with pytest.raises(RuntimeError, match="disabled"):
        run_arm(plan[0], allow_model_runs=True)


def test_constructor_contracts_and_full_context(monkeypatch, tmp_path):
    from src.evaluation import models
    from src.utils.serialization import file_sha256

    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"fake checkpoint: never imported")
    settings = {
        "seed": 123,
        "device": "cuda",
        "threads": 2,
        "parameters": {},
        "checkpoints": {
            "binary_classification": {"path": str(checkpoint), "sha256": file_sha256(checkpoint)}
        },
        "assets": [{"path": str(checkpoint), "sha256": file_sha256(checkpoint)}],
        "checkpoint_filename": "pinned.pt",
        "pretrain_dataset_or_path": "pinned-tabstar",
    }
    calls = []

    class Constructor:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.tokenizer = SimpleNamespace(random_seed=42)

        @classmethod
        def create_default_for_version(cls, version, **kwargs):
            assert version == "V3_5"
            return cls(**kwargs)

    modules = {
        "tabpfn": SimpleNamespace(TabPFNClassifier=Constructor),
        "tabpfn.constants": SimpleNamespace(ModelVersion=SimpleNamespace(V3_5="V3_5")),
        "tabicl": SimpleNamespace(TabICLClassifier=Constructor),
        "sap_rpt_oss": SimpleNamespace(SAP_RPT_OSS_Classifier=Constructor),
        "tabstar.tabstarv1p1_model": SimpleNamespace(TabSTARv1p1Classifier=Constructor),
    }
    monkeypatch.setattr(models.importlib, "import_module", modules.__getitem__)
    encoder = FoldInput().fit(pd.DataFrame({"n": [1, 2], "c": ["a", "b"]}))
    _backend("tabpfn_3_5", "binary_classification", settings, encoder, 9000)
    assert calls[-1]["inference_config"] == {"TRANSFORM_TEXT": False}
    assert calls[-1]["model_path"] == str(checkpoint)
    _backend("tabicl_v2", "binary_classification", settings, encoder, 9000)
    assert calls[-1]["allow_auto_download"] is False
    model = _backend("contexttab", "binary_classification", settings, encoder, 9000)
    assert calls[-1]["max_context_size"] == 9000 and calls[-1]["bagging"] == 1
    assert model.seed == model.tokenizer.random_seed == 123
    monkeypatch.setenv("JEVPFN_LOCAL_STORAGE_ROOT", str(tmp_path))
    _backend("tabstar", "binary_classification", settings, encoder, 9000)
    output_one = calls[-1]["output_dir"]
    _backend("tabstar", "binary_classification", settings, encoder, 9000)
    assert calls[-1]["output_dir"] != output_one
    assert "6. experiment_3" in output_one
    _backend(
        "tabstar",
        "binary_classification",
        settings | {"output_phase": "experiment_0"},
        encoder,
        9000,
    )
    assert "3. experiment_0" in calls[-1]["output_dir"]
    checkpoint.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="SHA256"):
        _backend("tabicl_v2", "binary_classification", settings, encoder, 9000)


def test_limix_rejects_context_retrieval_and_runtime_requires_pins(tmp_path):
    config = tmp_path / "inference.json"
    config.write_text(json.dumps({"pipelines": [{"retrieval_config": {"use_retrieval": True}}]}))
    with pytest.raises(ValueError, match="all-training-rows"):
        _limix_config(config)
    with pytest.raises(ValueError, match="pins"):
        verify_runtime({})


def test_limix_backend_uses_torch_device_and_training_only_adapter(monkeypatch, tmp_path):
    from src.evaluation import models
    from src.evaluation.limix_preprocessing import TrainOnlyLimiX
    from src.utils.serialization import file_sha256

    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"never loaded")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "pipelines": [{"retrieval_config": {"use_retrieval": False}}],
                "feature_encoder_mode": "train_unknown_nan",
                "feature_view_strategy": "disabled",
            }
        )
    )
    captured = {}

    class Predictor:
        __module__ = "inference.v2_0.predictor"

        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.get_categorical_features_indices = lambda x: []

    device = SimpleNamespace(type="cuda")
    modules = {
        "inference.predictor": SimpleNamespace(LimiXPredictor=Predictor),
        "torch": SimpleNamespace(device=lambda name: device),
    }
    monkeypatch.setattr(models.importlib, "import_module", modules.__getitem__)
    def pin(path):
        return {"path": str(path), "sha256": file_sha256(path)}

    settings = {
        "seed": 7,
        "device": "cuda",
        "threads": 4,
        "parameters": {},
        "checkpoints": {"regression": pin(checkpoint)},
        "inference_configs": {"regression": pin(config)},
    }
    backend = _backend(
        "limix_2", "regression", settings, FoldInput().fit(pd.DataFrame({"a": [1, 2]})), 2
    )
    assert isinstance(backend, TrainOnlyLimiX)
    assert captured["device"] is device
    assert captured["preprocess_num_jobs"] == 4
    assert captured["use_data_cache"] is False
