"""Synthetic contract tests only: no paid responses, datasets or models are fetched."""

import copy
import json
from dataclasses import replace

import pandas as pd
import pytest

from src.data.loaders import LoadedDataset
from src.data.metadata import extract_metadata
from src.jev.cache import ResponseCache
from src.jev.features import (
    assemble_features,
    block_identity,
    load_block,
    materialize_cached_block,
    save_block,
)
from src.jev.provider import api_preview, response_features
from src.jev.requests import build_requests
from src.utils.config import load_config


@pytest.fixture
def dataset():
    frame = pd.DataFrame(
        {"text": ["alpha", "beta", "alpha"], "context": [10, 20, 10], "target": [0, 1, 0]}
    )
    official = {"target": "target", "task_type": "cls", "num_classes": 2}
    metadata = extract_metadata(
        frame, official, dataset_id="fixture", dataset_name="Synthetic", text_columns=["text"]
    )
    return LoadedDataset(frame, metadata, official, {"data_identity": "sha256:fixture"}, [])


@pytest.fixture
def cfg():
    result = load_config("feature_creation/default")
    result["jev"]["representation"] = "per_column"
    return result


def response(request, value=0.25):
    return {
        "model": json.loads(request.settings_json)["model"],
        "answers": {"semantic": {"type": "noul", "noul": value}},
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }


def populate(dataset, cfg, cache):
    for i in range(len(dataset.frame)):
        request = build_requests(dataset.features, dataset.metadata, i, cfg, source=dataset.source)[
            0
        ]
        cache.put(request, response(request, 0.75 if i == 1 else 0.25), kind="verified_response")


def test_binary_mapping_is_label_free(dataset, cfg):
    request = build_requests(dataset.features, dataset.metadata, 0, cfg)[0]
    body = api_preview(request)
    assert body["model"] == "jev-1.13.0"
    assert body["questions"]["semantic"]["type"] == "noul"
    assert body["state"]["text"] == {"text": "alpha"}
    assert set(body["state"]) == {"task", "text"}
    assert response_features(request, response(request)) == {"p_positive": 0.25}


@pytest.mark.parametrize("bad", [True, None, -0.01, 1.01, float("nan"), float("inf"), "0.5"])
def test_invalid_probability_fails(dataset, cfg, bad):
    request = build_requests(dataset.features, dataset.metadata, 0, cfg)[0]
    with pytest.raises(ValueError):
        response_features(request, response(request, bad))


def test_response_version_must_match(dataset, cfg):
    request = build_requests(dataset.features, dataset.metadata, 0, cfg)[0]
    supplied = response(request)
    supplied["model"] = "jev-latest"
    with pytest.raises(ValueError, match="model"):
        response_features(request, supplied)
    cfg["jev"]["model"] = "jev-latest"
    with pytest.raises(ValueError, match="pin"):
        api_preview(build_requests(dataset.features, dataset.metadata, 0, cfg)[0])


def test_complete_multiclass_vector(dataset, cfg):
    # Stable IDs disambiguate numeric 1 from the string "1" without invented descriptions.
    dataset.metadata = replace(
        dataset.metadata, task_type="multiclass_classification", classes=(1, "1", "other")
    )
    request = build_requests(dataset.features, dataset.metadata, 0, cfg)[0]
    assert api_preview(request)["questions"]["semantic"]["criteria"] == {
        "c000": "1",
        "c001": '"1"',
        "c002": '"other"',
    }
    answer = {
        "type": "choice",
        "choice": "c001",
        "probabilities": {"c000": 0.1, "c001": 0.7, "c002": 0.2},
        "confidence": 0.4,
    }
    supplied = {"model": "jev-1.13.0", "answers": {"semantic": answer}}
    assert response_features(request, supplied) == {
        "p_c000": 0.1,
        "p_c001": 0.7,
        "p_c002": 0.2,
        "confidence": 0.4,
    }
    answer["probabilities"].pop("c002")
    with pytest.raises(ValueError, match="complete"):
        response_features(request, supplied)


def test_regression_preserves_nine_probabilities_and_maps_index(dataset, cfg):
    dataset.metadata = replace(dataset.metadata, task_type="regression", classes=())
    request = build_requests(dataset.features, dataset.metadata, 0, cfg)[0]
    criteria = api_preview(request)["questions"]["semantic"]["criteria"]
    assert len(criteria) == 9
    assert "no meaningful" in criteria[4]
    probabilities = {str(i): 0.0 for i in range(9)}
    probabilities.update({"2": 0.25, "6": 0.75})
    answer = {
        "type": "score",
        "score": 5.0,
        "legend": dict(zip(probabilities, criteria)),
        "probabilities": probabilities,
        "confidence": 0.5,
    }
    supplied = {"model": "jev-1.13.0", "answers": {"semantic": answer}}
    result = response_features(request, supplied)
    assert result["expected_direction"] == 1.0
    assert sum(value for key, value in result.items() if key.startswith("p_")) == 1
    assert len(result) == 11
    answer["score"] = 4
    with pytest.raises(ValueError, match="inconsistent"):
        response_features(request, supplied)


def test_features_require_verified_complete_cache(dataset, cfg, tmp_path):
    cache = ResponseCache(tmp_path / "responses.sqlite3")
    for i in range(len(dataset.frame)):
        request = build_requests(dataset.features, dataset.metadata, i, cfg)[0]
        cache.put(request, response(request), kind="mock")
    with pytest.raises(ValueError, match="Missing verified"):
        materialize_cached_block(dataset, cfg, cache)
    populate(dataset, cfg, cache)
    reopened = ResponseCache(cache.path)
    block = materialize_cached_block(dataset, cfg, reopened)
    assert block.values.columns.tolist() == [
        "row_id",
        "jev__per_column__text_only__text_000__p_positive",
    ]
    assert block.requests.request_key.nunique() == 2
    assert block.values.row_id.tolist() == ["csv:0", "csv:1", "csv:2"]


def test_parquet_roundtrip_idempotence_and_corruption(dataset, cfg, tmp_path):
    cache = ResponseCache(tmp_path / "responses.sqlite3")
    populate(dataset, cfg, cache)
    block = materialize_cached_block(dataset, cfg, cache)
    folder = save_block(block, tmp_path / "feature-block")
    pd.testing.assert_frame_equal(load_block(folder).values, block.values)
    assert save_block(block, folder) == folder
    changed = copy.deepcopy(block)
    changed.values.iloc[0, 1] = 0.99
    with pytest.raises(ValueError, match="overwrite"):
        save_block(changed, folder)
    with (folder / "features.parquet").open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_block(folder)


def test_join_uses_row_ids_and_rejects_wrong_artifacts(dataset, cfg, tmp_path):
    cache = ResponseCache(tmp_path / "responses.sqlite3")
    populate(dataset, cfg, cache)
    block = materialize_cached_block(dataset, cfg, cache)
    block.values = block.values.iloc[[1, 2, 0]]
    result = assemble_features(dataset, [block])
    assert result.iloc[:, -1].tolist() == [0.25, 0.75, 0.25]
    assert result.context.tolist() == [10, 20, 10]
    assert "target" not in result and "text" not in result
    with pytest.raises(ValueError, match="Duplicate"):
        assemble_features(dataset, [block, block])
    block.values = block.values.iloc[:2]
    with pytest.raises(ValueError, match="row IDs"):
        assemble_features(dataset, [block])
    block.manifest["identity"]["data_identity"] = "different-version"
    with pytest.raises(ValueError, match="different dataset"):
        assemble_features(dataset, [block])


def test_feature_identity_excludes_experiments_and_pricing(dataset, cfg):
    before = block_identity(dataset.metadata, "fixture", cfg)
    cfg["experiment"] = 3
    cfg["cost"]["input_price_per_million"] = 200
    assert block_identity(dataset.metadata, "fixture", cfg) == before
    cfg["jev"]["version"] = "future-version"
    assert block_identity(dataset.metadata, "fixture", cfg) != before


def test_missing_text_features_are_nan_and_traceable(dataset, cfg, tmp_path):
    dataset.frame.loc[0, "text"] = " \t "
    dataset.frame.loc[2, "text"] = None
    cache = ResponseCache(tmp_path / "responses.sqlite3")
    request = build_requests(dataset.features, dataset.metadata, 1, cfg)[0]
    cache.put(request, response(request), kind="verified_response")
    block = materialize_cached_block(dataset, cfg, cache)
    assert block.values.iloc[[0, 2], 1:].isna().all().all()
    assert block.values.iloc[1, 1] == 0.25
    assert block.requests.status.tolist() == ["missing_text", "verified_response", "missing_text"]
    assert block.requests.loc[[0, 2], "request_key"].isna().all()
    restored = load_block(save_block(block, tmp_path / "features"))
    assert assemble_features(dataset, [restored]).iloc[[0, 2], -1].isna().all()
