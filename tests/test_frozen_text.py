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

    train = pd.DataFrame({"x": [1, 2, 3], "e0": [0.0, 2.0, np.nan], "e1": [2.0, 4.0, np.nan]})
    test = pd.DataFrame({"x": [4], "e0": [10000.0], "e1": [20000.0]})
    wrapped = EmbeddingControl(Model(), [["e0", "e1"]], components=2).fit(train, [0, 1, 2])
    np.testing.assert_allclose(wrapped.pcas[0].mean_, [1, 3])
    np.testing.assert_array_equal(wrapped.predict(test), [4])
    assert wrapped.estimator.seen.x.equals(train.x)
    assert wrapped.estimator.seen.iloc[2]["e5_pca__000__missing"] == 1
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
