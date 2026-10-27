"""Native B200 checkpoint resolution without loading any predictor or weight."""

from types import SimpleNamespace

import pandas as pd
import pytest

from src.evaluation import checkpoint_paths, models
from src.utils import paths
from src.utils.serialization import file_sha256


@pytest.fixture
def trees(tmp_path, monkeypatch):
    durable = tmp_path / "lustre"
    native = tmp_path / "gpfs" / "JEVPFN" / "checkpoints"
    durable_checkpoints = durable / "JEVPFN" / "checkpoints"
    native.mkdir(parents=True)
    durable_checkpoints.mkdir(parents=True)
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "mindwell")
    monkeypatch.setenv("JEVPFN_DURABLE_ROOT", str(durable))
    monkeypatch.setattr(paths, "checkpoints_dir", lambda *parts: native.joinpath(*parts))
    return durable_checkpoints, native


def test_durable_and_relative_pins_select_native_verified_copy(trees):
    durable, native = trees
    relative = "model/weights.bin"
    copied = native / relative
    copied.parent.mkdir()
    copied.write_bytes(b"staged reviewed weights")
    # The durable source need not be read or even mounted for checkpoint hashing.
    pin = {"path": str(durable / relative), "sha256": file_sha256(copied)}
    assert models._checkpoint({"checkpoints": {"regression": pin}}, "regression") == str(copied)
    models._verify_assets({"assets": [pin]})
    assert checkpoint_paths.checkpoint_path(relative) == copied
    assert checkpoint_paths.checkpoint_path(copied) == copied
    copied.write_bytes(b"wrong staged copy")
    with pytest.raises(ValueError, match="SHA256"):
        models._checkpoint({"checkpoints": {"regression": pin}}, "regression")


def test_external_and_traversing_pins_are_rejected(trees, tmp_path):
    durable, _ = trees
    for value in (
        tmp_path / "other-project/model.bin",
        "../../outside.bin",
        durable.parent / "outside.bin",
    ):
        with pytest.raises(ValueError, match="outside|escapes"):
            checkpoint_paths.checkpoint_path(value)


def test_missing_native_asset_never_falls_back_to_durable(trees):
    durable, _ = trees
    weight = durable / "existing.bin"
    weight.write_bytes(b"must not read this from the GPU")
    with pytest.raises(ValueError, match="stage it on CPU"):
        checkpoint_paths.model_checkpoint_argument(weight)


def test_mindwell_cannot_keep_durable_as_its_active_checkpoint_tree(trees, monkeypatch):
    durable, _ = trees
    monkeypatch.setattr(paths, "checkpoints_dir", lambda: durable)
    with pytest.raises(ValueError, match="native staged"):
        checkpoint_paths.checkpoint_path("weights.bin")


def test_local_and_wice_arguments_remain_compatible(trees, monkeypatch):
    durable, native = trees
    for cluster in ("", "wice"):
        monkeypatch.setenv("SLURM_CLUSTER_NAME", cluster)
        assert checkpoint_paths.checkpoint_path(durable / "weights.bin") == durable / "weights.bin"
        assert checkpoint_paths.checkpoint_path("weights.bin") == native / "weights.bin"
        assert (
            checkpoint_paths.model_checkpoint_argument("publisher-local-alias")
            == "publisher-local-alias"
        )


@pytest.mark.parametrize(
    "model_id,argument",
    [("tabstar", "pretrain_dataset_or_path"), ("contexttab", "checkpoint_filename")],
)
def test_native_predictor_constructor_uses_staged_asset(trees, monkeypatch, model_id, argument):
    durable, native = trees
    asset = native / "native.bin"
    asset.write_bytes(b"staged native checkpoint")
    calls = {}

    class FakePredictor:
        def __init__(self, **kwargs):
            calls.update(kwargs)
            self.tokenizer = SimpleNamespace()

    module = SimpleNamespace(TabSTARv1p1Classifier=FakePredictor, SAP_RPT_OSS_Classifier=FakePredictor)
    monkeypatch.setattr(models.importlib, "import_module", lambda _: module)
    monkeypatch.setattr(paths, "results_dir", lambda *args, **kwargs: native / "test-results")
    settings = {
        "seed": 1,
        "device": "cuda",
        "threads": 1,
        argument: str(durable / "native.bin"),
        "assets": [{"path": str(durable / "native.bin"), "sha256": file_sha256(asset)}],
    }
    models._backend(
        model_id,
        "binary_classification",
        settings,
        models.FoldInput().fit(pd.DataFrame({"x": [1, 2]})),
        2,
    )
    assert calls["pretrain_dataset_or_path" if model_id == "tabstar" else "checkpoint"] == str(
        asset
    )


def test_limix_loads_both_weights_and_recipe_from_native_copy(trees, monkeypatch):
    import json

    durable, native = trees
    weight = native / "limix.bin"
    recipe = native / "inference.json"
    weight.write_bytes(b"reviewed weights")
    recipe.write_text(
        json.dumps(
            {
                "pipelines": [{"retrieval_config": {"use_retrieval": False}}],
                "feature_encoder_mode": "train_unknown_nan",
                "feature_view_strategy": "disabled",
            }
        )
    )
    calls = {}

    class FakePredictor:
        __module__ = "inference.v2_0.predictor"
        gpu_ids = [0]

        def get_categorical_features_indices(self, values):
            return []

    def constructor(**kwargs):
        calls.update(kwargs)
        return FakePredictor()

    monkeypatch.setattr(
        models.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(device=lambda value: ("torch.device", value))
            if name == "torch"
            else SimpleNamespace(LimiXPredictor=constructor)
        ),
    )
    settings = {
        "seed": 1,
        "device": "cuda",
        "threads": 1,
        "checkpoints": {
            "regression": {"path": str(durable / weight.name), "sha256": file_sha256(weight)}
        },
        "inference_configs": {
            "regression": {"path": str(durable / recipe.name), "sha256": file_sha256(recipe)}
        },
    }
    models._backend(
        "limix_2", "regression", settings, models.FoldInput().fit(pd.DataFrame({"x": [1, 2]})), 2
    )
    assert calls["model_path"] == str(weight)
    assert calls["inference_config"] == json.loads(recipe.read_text())
    assert calls["device"] == ("torch.device", "cuda")
    assert calls["preprocess_num_jobs"] == 1


def test_link_cannot_escape_native_checkpoint_tree(trees, tmp_path):
    _, native = trees
    external = tmp_path / "unapproved.bin"
    external.write_bytes(b"outside")
    link = native / "linked.bin"
    try:
        link.symlink_to(external)
    except OSError:
        pytest.skip("Native symlink creation unavailable")
    with pytest.raises(ValueError, match="escapes"):
        checkpoint_paths.checkpoint_path("linked.bin")
