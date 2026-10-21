"""Cross-experiment contracts without weights, API calls or benchmark training."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data import text_embeddings
from src.evaluation.embedding_control import EmbeddingControl
from src.evaluation.features import combine_representations
from src.evaluation.models import KumoBackend
from src.evaluation.widths import feature_width, width_status


def test_joint_pca_fits_only_embedding_coordinates_from_training_rows():
    train = pd.DataFrame(
        {
            "amount": [1, 2, 3, 4],
            "decision": [0.2, 0.3, 0.4, 0.5],
            "a0": [0.0, 2.0, np.nan, np.nan],
            "a1": [2.0, 4.0, np.nan, np.nan],
            "b0": [10.0, np.nan, 20.0, np.nan],
            "b1": [30.0, np.nan, 40.0, np.nan],
        }
    )
    received = []
    model = SimpleNamespace(fit=lambda x, y: received.append(x))
    wrapped = EmbeddingControl(
        model, [["a0", "a1"], ["b0", "b1"]], components=3, scope="joint"
    ).fit(train, [0, 1, 2, 3])
    np.testing.assert_allclose(wrapped.imputation_[0], [1, 3, 15, 35])
    np.testing.assert_allclose(wrapped.pcas[0].mean_, [1, 3, 15, 35])
    assert received[0].amount.equals(train.amount)
    assert received[0].decision.equals(train.decision)
    assert received[0].filter(like="e5_pca").iloc[-1].isna().all()
    heldout = train.copy()
    heldout.loc[0, ["a0", "a1"]] = [1e6, 2e6]
    assert list(wrapped.transform(heldout)) == list(received[0])
    np.testing.assert_allclose(wrapped.imputation_[0], [1, 3, 15, 35])
    broken = train.copy()
    broken.loc[0, "a0"] = np.nan
    with pytest.raises(ValueError, match="Partially"):
        EmbeddingControl(model, [["a0", "a1"], ["b0", "b1"]], scope="joint").fit(broken, [0] * 4)


def test_provider_concatenation_preserves_original_text_and_shared_presence_once():
    base = pd.DataFrame({"number": [1.0, 2.0], "text": ["sample", None]}, index=["csv:0", "csv:1"])
    additions = []
    for name, probability in (("jev", 0.2), ("liquid", 0.8)):
        matrix = base[["number"]].copy()
        matrix["p_positive"] = [probability, np.nan]
        matrix["text_presence__000__missing"] = [0, 1]
        matrix.attrs["feature_artifact_ids"] = [name + "-artifact"]
        additions.append((name, matrix))
    result = combine_representations(base, additions)
    assert result.text.equals(base.text)
    assert result.columns.tolist() == [
        "number",
        "text",
        "jev__p_positive",
        "text_presence__000__missing",
        "liquid__p_positive",
    ]
    assert result.attrs["feature_artifact_ids"] == ["jev-artifact", "liquid-artifact"]
    additions[1][1].index = ["csv:1", "csv:0"]
    with pytest.raises(ValueError, match="row identities"):
        combine_representations(base, additions)


def test_rotten_tomatoes_pca_width_limit_and_joint_alternative():
    metadata = SimpleNamespace(
        text_columns=range(13), non_text_columns=range(2), task_type="regression", classes=[]
    )
    arm = dict(
        feature_set="non_text_plus_frozen_e5",
        model="kumo_tabular",
        provider=None,
        pca_components=64,
        pca_scope="per_column",
    )
    assert feature_width(metadata, arm) == 847
    assert width_status(metadata, arm)["width_status"] == "unsupported_width"
    assert feature_width(metadata, arm | {"pca_scope": "joint"}) == 79
    assert feature_width(metadata, arm | {"pca_components": 32}) == 431
    combined = arm | dict(feature_set="non_text_plus_both", providers=["jev", "liquid"])
    assert feature_width(metadata, combined) == 2 + 13 + 2 * 14


def test_kumo_guard_counts_publisher_category_expansion_before_silent_selection():
    model = object.__new__(KumoBackend)
    model.categorical_columns = {"category"}
    values = pd.DataFrame(np.zeros((51, 499)))
    values["category"] = [f"value_{i}" for i in range(51)]
    with pytest.raises(ValueError, match="category-count"):
        model.fit(values, np.zeros(51))


def test_encoder_namespaces_are_distinct_and_recipes_match_publishers(isolated_output):
    configs = [text_embeddings.encoder_config(i) for i in text_embeddings.MODEL_NAMESPACES.values()]
    locations = [text_embeddings.cache_path("dataset", c) for c in configs]
    assert len(set(locations)) == 3
    assert [c["dimensions"] for c in configs] == [384, 1024, 1024]
    assert [c["pooling"] for c in configs] == ["attention_mask_mean"] * 2 + [
        "last_nonpadding_token"
    ]
    assert [c["prefix"] for c in configs] == ["query: ", "query: ", ""]
    for c in configs:
        assert text_embeddings.encoder_id_for_model(c["model"]) == c["encoder_id"]


def test_all_encoder_caches_transfer_without_namespace_collision(
    isolated_output, tmp_path, monkeypatch
):
    from test_frozen_text import Encoder, fixture_data

    from src.cluster.transfer import export_bundle, import_bundle
    from src.utils import paths

    recipes = [
        text_embeddings.encoder_config(i) | {"dimensions": 3, "batch_size": 2}
        for i in text_embeddings.MODEL_NAMESPACES.values()
    ]
    for cfg in recipes:
        text_embeddings.build(fixture_data(), cfg, Encoder())
    bundle = export_bundle()
    monkeypatch.delenv("VSC_DATA")
    monkeypatch.delenv("JEVPFN_STAGING_ROOT")
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path / "new_checkout")
    report = import_bundle(bundle)
    assert len(report["embedding_caches"]) == 3
    for cfg in recipes:
        assert text_embeddings.load(fixture_data(), cfg).shape == (4, 6)
