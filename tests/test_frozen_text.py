"""Synthetic encoders verify reuse/row identity/recovery without downloading weights."""

import copy
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.data import text_embeddings as embeddings
from src.data.loaders import LoadedDataset
from src.data.metadata import extract_metadata
from src.evaluation.embedding_control import EmbeddingControl
from src.utils.config import load_yaml


def fixture_data():
    frame = pd.DataFrame(
        {
            "a": ["alpha", "alpha", None, "beta"],
            "b": ["beta", " ", "gamma", "alpha"],
            "x": [1, 2, 3, 4],
            "y": [0.0, 1.0, 2.0, 3.0],
        }
    )
    official = {"target": "y", "task_type": "reg"}
    meta = extract_metadata(
        frame, official, dataset_id="fixture", dataset_name="Fixture", text_columns=["a", "b"]
    )
    return LoadedDataset(frame, meta, official, {"data_identity": "fixed"}, [])


class Encoder:
    identity = {"runtime": "synthetic"}

    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def __call__(self, texts):
        if self.fail and self.calls:
            raise RuntimeError("interrupted")
        self.calls.extend(texts)
        return np.array([[len(t), ord(t[0]), 0.0] for t in texts])


@pytest.fixture
def recipe():
    return load_yaml("feature_creation/text_embeddings") | {"dimensions": 3, "batch_size": 1}


def test_resume_skips_completed_text_deduplicates_and_retains_missing(
    isolated_output, tmp_path, recipe
):
    dataset, path = fixture_data(), tmp_path / "fixture.sqlite3"
    encoder = Encoder(fail=True)
    with pytest.raises(RuntimeError, match="interrupted"):
        embeddings.build(dataset, recipe, encoder, destination=path)
    with pytest.raises(ValueError, match="Incomplete"):
        embeddings.validate(path, require_complete=True)
    resumed = Encoder()
    identity = embeddings.build(dataset, recipe, resumed, destination=path)
    assert encoder.calls + resumed.calls == ["alpha", "beta", "gamma"]
    assert embeddings.build(dataset, recipe, resumed, destination=path) == identity
    assert len(resumed.calls) == 2
    features = embeddings.load(dataset, recipe, source=path)
    assert list(features.index) == ["csv:0", "csv:1", "csv:2", "csv:3"]
    np.testing.assert_array_equal(features.iloc[0, :3], features.iloc[1, :3])
    assert features.iloc[2, :3].isna().all() and features.iloc[1, 3:].isna().all()
    dataset.frame["y"] = [99, 98, 97, 96]
    assert embeddings.input_rows(dataset) == embeddings.input_rows(fixture_data())
    changed = copy.deepcopy(dataset)
    changed.frame.loc[0, "a"] = "different"
    with pytest.raises(ValueError, match="does not match"):
        embeddings.load(changed, recipe, source=path)


def test_corruption_and_conflicting_merge_are_rejected(isolated_output, tmp_path, recipe):
    source, local = tmp_path / "source.sqlite3", tmp_path / "local.sqlite3"
    embeddings.build(fixture_data(), recipe, Encoder(), destination=source)
    embeddings.merge(source, local)
    embeddings.merge(source, local)
    pd.testing.assert_frame_equal(
        embeddings.load(fixture_data(), recipe, source=source),
        embeddings.load(fixture_data(), recipe, source=local),
    )
    with sqlite3.connect(source) as db:
        db.execute("UPDATE embeddings SET vector=? WHERE text='alpha'", (b"corrupt",))
    with pytest.raises((ValueError, TypeError)):
        embeddings.merge(source, local)
    embeddings.validate(local, require_complete=True)


def test_embedding_transfer_is_in_combined_bundle(isolated_output, tmp_path, monkeypatch, recipe):
    from src.cluster.transfer import export_bundle, import_bundle
    from src.utils import paths

    dataset = fixture_data()
    embeddings.build(dataset, recipe, Encoder())
    bundle = export_bundle()
    monkeypatch.delenv("VSC_DATA")
    monkeypatch.delenv("JEVPFN_STAGING_ROOT")
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path / "local")
    result = import_bundle(bundle)
    assert result["embedding_caches"][0]["status"] == "copied"
    assert embeddings.load(dataset, recipe).shape == (4, 6)
    assert import_bundle(bundle)["embedding_caches"][0]["status"] == "merged"


def test_pca_uses_only_training_rows_and_preserves_non_text_values():
    class Model:
        def fit(self, x, y):
            self.seen = x

        def predict(self, x):
            return x.x.to_numpy()

    train = pd.DataFrame(
        {
            "x": [1, 2, 3],
            "decision": [0.1, 0.2, np.nan],
            "e0": [0.0, 2.0, np.nan],
            "e1": [2.0, 4.0, np.nan],
        }
    )
    test = pd.DataFrame({"x": [4], "decision": [0.8], "e0": [10000.0], "e1": [20000.0]})
    wrapped = EmbeddingControl(Model(), [["e0", "e1"]], components=2).fit(train, [0, 1, 2])
    np.testing.assert_allclose(wrapped.pcas[0].mean_, [1, 3])
    np.testing.assert_array_equal(wrapped.predict(test), [4])
    assert wrapped.estimator.seen.x.equals(train.x)
    assert wrapped.estimator.seen.decision.equals(train.decision)
    assert wrapped.transform(test).decision.equals(test.decision)
    assert wrapped.estimator.seen.iloc[2]["text_presence__000__missing"] == 1
    assert (
        wrapped.estimator.seen.iloc[2]["e5_pca__000__000"]
        != wrapped.estimator.seen.iloc[2]["e5_pca__000__000"]
    )
    # A fresh inner fit must not reuse the outer PCA basis.
    inner = EmbeddingControl(Model(), [["e0", "e1"]], components=2).fit(train.iloc[[1, 2]], [1, 2])
    assert inner.pcas == [None]


def test_long_embedding_text_is_fully_covered_with_room_for_prefix_and_special_tokens():
    class Tokenizer:
        def encode(self, text, **_):
            return [90, 91] if text == "query: " else list(range(1300))

        def num_special_tokens_to_add(self, _):
            return 2

    prefix, chunks = embeddings.token_chunks(
        Tokenizer(), "long text", prefix="query: ", max_tokens=512
    )
    assert prefix == [90, 91]
    assert [len(c) for c in chunks] == [508, 508, 284]
    assert [token for chunk in chunks for token in chunk] == list(range(1300))
    assert all(len(prefix) + len(chunk) + 2 <= 512 for chunk in chunks)
    with pytest.raises(ValueError, match="No room"):
        embeddings.token_chunks(Tokenizer(), "long text", prefix="query: ", max_tokens=4)


def test_encoder_batches_across_texts_and_preserves_chunk_weighting():
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    calls = []

    class Batch(dict):
        def to(self, _):
            return self

    class Tokenizer:
        def encode(self, text, **_):
            return [2] if text == "query: " else list(range(3, 3 + len(text.split())))

        def num_special_tokens_to_add(self, _):
            return 2

        def prepare_for_model(self, ids, **_):
            return {"input_ids": [1, *ids, 1]}

        def pad(self, rows, **_):
            width = max(len(r["input_ids"]) for r in rows)
            ids = [r["input_ids"] + [0] * (width - len(r["input_ids"])) for r in rows]
            tensor = torch.tensor(ids)
            return Batch(input_ids=tensor, attention_mask=tensor.ne(0).int())

    def model(**inputs):
        ids = inputs["input_ids"].float()
        calls.append(len(ids))
        return SimpleNamespace(last_hidden_state=torch.stack([ids, torch.ones_like(ids)], -1))

    encoder = object.__new__(embeddings.E5Encoder)
    encoder.cfg = {
        "dimensions": 2,
        "batch_size": 2,
        "max_tokens": 6,
        "prefix": "query: ",
        "device": "cpu",
    }
    encoder.torch, encoder.tokenizer, encoder.model = torch, Tokenizer(), model
    texts = ["one", "two", "a b c d e f g"]
    batched = encoder(texts)
    assert calls == [2, 2, 1]  # Three chunks in the long text, two short texts together.
    calls.clear()
    encoder.cfg["batch_size"] = 1
    np.testing.assert_allclose(encoder(texts), batched, atol=1e-7)


def test_combined_join_preserves_both_artifact_identities_and_missingness():
    from src.evaluation.features import add_embeddings

    base = pd.DataFrame({"x": [1, 2], "jev": [0.5, np.nan]}, index=["csv:0", "csv:1"])
    base.attrs["feature_artifact_ids"] = ["decision-artifact"]
    encoded = pd.DataFrame({"e": [9, np.nan]}, index=base.index)
    encoded.attrs.update(feature_artifact_ids=["embedding-artifact"], embedding_groups=[["e"]])
    result = add_embeddings(base, encoded)
    assert result.attrs["feature_artifact_ids"] == ["decision-artifact", "embedding-artifact"]
    assert result.loc["csv:1", ["jev", "e"]].isna().all()
    assert result.attrs["embedding_groups"] == [["e"]]
    with pytest.raises(ValueError, match="row IDs/order"):
        add_embeddings(base, encoded.iloc[::-1])
    with pytest.raises(ValueError, match="collide"):
        add_embeddings(base, encoded.rename(columns={"e": "x"}))
    with pytest.raises(ValueError, match="unique row"):
        add_embeddings(base, encoded.set_axis(["csv:0", "csv:0"]))
    encoded.attrs["embedding_groups"] = [["e"], ["e"]]
    with pytest.raises(ValueError, match="exactly once"):
        add_embeddings(base, encoded)


@pytest.mark.parametrize(
    "model", ["catboost", "tabpfn_3_5", "tabicl_v2", "limix_2", "kumo_tabular", "linear"]
)
@pytest.mark.parametrize("mode", ["joint", "per_column", "both", None])
def test_campaign_embedding_factory_for_every_predictor_and_decision_mode(
    isolated_output, monkeypatch, model, mode
):
    from types import SimpleNamespace

    from src.evaluation import campaign

    dataset = fixture_data()
    name = "non_text_plus_frozen_e5" + (f"_and_{mode}" if mode else "")
    arm = dict(
        dataset_id="fixture",
        model=model,
        provider="jev" if mode else None,
        feature_set=name,
        phase="experiment_1",
    )
    real_load = campaign.load_yaml
    selector = dict(
        provider="jev",
        model="jev-1.13.0",
        version="pin",
        question_bank_id="bank",
        question_variant="accepted",
    )

    def config(name):
        cfg = copy.deepcopy(real_load(name))
        if name == "experiment_0/debug":
            cfg["full_fold_profile_enabled"] = True
        if name == "experiment_1/main":
            cfg["feature_selectors"]["jev"] = selector
        if name == "experiment_1/models":
            cfg["models"][model]["runtime"]["device"] = "cpu"
        return cfg

    monkeypatch.setattr(campaign, "load_yaml", config)
    monkeypatch.setattr(campaign, "benchmark_plan", lambda **_: [arm])
    monkeypatch.setattr(campaign, "load_catalog", lambda: {"datasets": [{"id": "fixture"}]})
    monkeypatch.setattr(campaign, "load_dataset", lambda *_, **kw: dataset)
    monkeypatch.setattr(campaign, "shared_splits", lambda *_, **kw: {})
    selected = []

    def selection(path, selector, representations):
        selected.extend(representations)
        return []

    monkeypatch.setattr(campaign, "load_feature_selection", selection)
    expected_modes = [] if mode is None else ["per_column", "joint"] if mode == "both" else [mode]
    base = dataset.frame[["x"]].set_axis([f"csv:{i}" for i in range(4)])
    if mode:
        base["decision"] = [0.2, 0.4, np.nan, 0.6]
    base.attrs["feature_artifact_ids"] = [f"{m}-artifact" for m in expected_modes]

    def assemble(data, blocks, name, **kw):
        assert name == (f"non_text_plus_{mode}" if mode else "non_text")
        return base.copy()

    monkeypatch.setattr(campaign, "assemble_feature_set", assemble)
    encoded = pd.DataFrame(
        {"e0": [0.0, 2.0, np.nan, 3.0], "e1": [2.0, 4.0, np.nan, 8.0]}, index=base.index
    )
    encoded.attrs.update(feature_artifact_ids=["e5-artifact"], embedding_groups=[["e0", "e1"]])
    monkeypatch.setattr(embeddings, "load", lambda *args: encoded)
    received = []
    monkeypatch.setattr(
        campaign,
        "estimator_factory",
        lambda *args, **kw: (
            lambda: SimpleNamespace(
                fit=lambda x, y: received.append(x.copy()), predict=lambda x: x.x.to_numpy()
            )
        ),
    )

    def evaluate(data, features, factory, **kw):
        assert features.attrs["feature_artifact_ids"] == base.attrs["feature_artifact_ids"] + [
            "e5-artifact"
        ]
        assert kw["representation_recipe"]["provider"] == arm["provider"]
        wrapped = factory().fit(features.iloc[:3], data.frame.y.iloc[:3])
        np.testing.assert_allclose(wrapped.pcas[0].mean_, [1, 3])
        assert len([c for c in received[0] if c.startswith("e5_pca")]) == 32
        assert received[0]["text_presence__000__missing"].tolist() == [0, 0, 1]
        assert not any(c.startswith("e0") or c.startswith("e1") for c in received[0])
        if mode:
            assert received[0].decision.equals(base.decision.iloc[:3])
        wrapped.predict(features.iloc[3:])
        return []

    monkeypatch.setattr(campaign, "evaluate_cv", evaluate)
    campaign.run_arm(arm, allow_model_runs=True, profile=True)
    assert selected == expected_modes
