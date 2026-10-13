"""Provider-neutral storage/selection tests; synthetic values, not alternative-model decoding."""

import copy
import json

import pandas as pd
import pytest

from src.data.loaders import LoadedDataset
from src.data.metadata import extract_metadata
from src.evaluation.features import assemble_feature_set, load_feature_selection
from src.jev.cache import ResponseCache
from src.jev.features import feature_inventory, materialize_cached_block, save_block
from src.jev.identity import bind_provider
from src.jev.requests import build_requests
from src.utils import paths
from src.utils.config import load_config, load_yaml
from src.utils.serialization import digest


def test_two_providers_coexist_and_are_selected_explicitly(tmp_path, isolated_output):
    frame = pd.DataFrame({"text": ["a", "b"], "n": [1, 2], "y": [0, 1]})
    meta = extract_metadata(
        frame,
        {"target": "y", "task_type": "cls"},
        dataset_id="fixture",
        dataset_name="Fixture",
        text_columns=["text"],
    )
    dataset = LoadedDataset(frame, meta, {}, {"data_identity": "fixture"}, [])
    cfg = load_config("feature_creation")
    cfg["jev"]["representation"] = "per_column"
    path = paths.dataset_cache_path("fixture")
    cache = ResponseCache(path, dataset_id="fixture")
    requests = [build_requests(dataset.features, meta, i, cfg)[0] for i in range(2)]
    for r in requests:
        cache.put(
            r,
            {"model": "jev-1.13.0", "answers": {"semantic": {"type": "noul", "noul": 0.25}}},
            kind="verified_response",
        )
    jev = materialize_cached_block(dataset, cfg, cache)
    save_block(jev, path)
    # Stand-in for the future Liquid decoder's normalized block. Never claim the
    # synthetic raw response below is a real provider response format.
    liquid = copy.deepcopy(jev)
    bound = [bind_provider(r, "liquid", load_yaml("extractors")) for r in requests]
    identity = liquid.manifest["identity"]
    settings = json.loads(bound[0].settings_json)
    for field in ("provider", "model", "version", "api_mapping_version"):
        identity[field] = settings[field]
    liquid.manifest["artifact_id"] = digest(identity)
    liquid.requests["request_key"] = [r.cache_key for r in bound]
    for r in bound:
        cache.put(r, {"synthetic_fixture": True, "p": 0.8}, kind="verified_response")
    old_columns = liquid.manifest["feature_columns"]
    new_columns = [c.replace("jev__", "liquid__", 1) for c in old_columns]
    liquid.values = liquid.values.rename(columns=dict(zip(old_columns, new_columns)))
    liquid.values[new_columns] = 0.8
    liquid.manifest["feature_columns"] = new_columns
    save_block(liquid, path)
    inventory = feature_inventory(path)
    assert set(inventory.provider) == {"jev", "liquid"}
    chosen = load_feature_selection(path, {"provider": "liquid"}, ("per_column",))
    assert len(chosen) == 1 and chosen[0].manifest["artifact_id"] == liquid.manifest["artifact_id"]
    matrix = assemble_feature_set(
        dataset, [jev, liquid], "non_text_plus_per_column", selector={"provider": "liquid"}
    )
    assert matrix.filter(like="liquid__").eq(0.8).all().all()
    assert not any(c.startswith("jev__") for c in matrix)
    assert matrix.attrs["feature_artifact_ids"] == [liquid.manifest["artifact_id"]]
    with pytest.raises(ValueError, match="Ambiguous"):
        assemble_feature_set(dataset, [jev, liquid], "non_text_plus_per_column")
    with pytest.raises(ValueError, match="exactly one"):
        load_feature_selection(path, {"provider": "microsoft"}, ("per_column",))


def test_inventory_never_creates_empty_dataset_cache(tmp_path):
    path = tmp_path / "absent.sqlite3"
    assert feature_inventory(path).empty
    assert not path.exists()
