"""The expensive mistakes: label leakage, unstable keys, mode semantics and cost undercounting."""

import copy
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.data.audit import audit_dataset, request_lengths
from src.data.loaders import LoadedDataset
from src.data.metadata import extract_metadata
from src.jev.cache import DisabledJevClient, ResponseCache, dry_run
from src.jev.cost import estimate_dataset
from src.jev.requests import IntendedRequest, build_requests
from src.jev.templates import make_question
from src.utils.config import load_config, validate_config
from src.utils.serialization import canonical_json, scalar


@pytest.fixture
def cfg():
    return load_config()


def example(task="cls", *, single=False):
    frame = pd.DataFrame(
        {
            "short": ["same", "same", None],
            "long": ['a "quoted" string\n', "Ω", ""],
            "number": [4.5, np.nan, -2.0],
            "category": ["A", "B", "A"],
            "outcome": ["no", "yes", "no"] if task == "cls" else [1.0, 20.0, 400.0],
        }
    )
    text = ["short"] if single else ["short", "long"]
    official = {"target": "outcome", "task_type": task, "num_classes": 2}
    metadata = extract_metadata(
        frame, official, dataset_id="TEST_TEXT", dataset_name="Example", text_columns=text
    )
    return LoadedDataset(frame, metadata, official, {"data_identity": "fixture"}, [])


def test_metadata_uses_official_task_and_vocab_only():
    dataset = example()
    assert dataset.metadata.classes == ("no", "yes")
    assert dataset.metadata.task_type == "binary_classification"
    assert set(dataset.metadata.task_context()) == {
        "dataset",
        "target",
        "task_type",
        "class_labels",
    }
    regression = example("reg")
    assert regression.metadata.classes == ()
    assert "statistics" not in canonical_json(regression.metadata.task_context())


def test_numeric_classification_is_not_inferred_as_regression():
    frame = pd.DataFrame({"text": ["A", "B", "C"], "y": [0, 1, 2]})
    metadata = extract_metadata(
        frame,
        {"target": "y", "task_type": "cls", "num_classes": 3},
        dataset_id="M",
        dataset_name="M",
        text_columns=["text"],
    )
    assert metadata.task_type == "multiclass_classification"
    assert make_question(metadata)["allowed_answers"] == [0, 1, 2]


@pytest.mark.parametrize(
    "mode,count", [("per_column", 2), ("joint", 1), ("per_column_and_joint", 3)]
)
@pytest.mark.parametrize("structured", [False])
def test_state_modes_are_deterministic_and_lossless(cfg, mode, count, structured):
    dataset = example()
    cfg["jev"].update(representation=mode, include_non_text_features=structured)
    requests = build_requests(dataset.features, dataset.metadata, 0, cfg)
    assert len(requests) == count
    for request in requests:
        state = request.to_dict()["payload"]["state"]
        assert ("structured" in state) is structured
        assert "outcome" not in state["text"]
        if "long" in state["text"]:
            assert state["text"]["long"] == 'a "quoted" string\n'
        assert IntendedRequest.from_json(request.to_json()) == request
    assert requests == build_requests(dataset.features, dataset.metadata, 0, cfg)


def test_regression_scale_and_neutral_label(cfg):
    dataset = example("reg")
    question = make_question(dataset.metadata)
    assert [a["score"] for a in question["allowed_answers"]] == list(range(-4, 5))
    assert (
        question["allowed_answers"][4]["meaning"] == "provides no meaningful directional evidence"
    )
    assert "relatively lower or higher" in question["text"]
    with pytest.raises(ValueError):
        make_question(dataset.metadata, scores=[-1, 0, 1])


def test_target_values_and_frequencies_cannot_change_requests(cfg):
    dataset = example()
    before = build_requests(dataset.features, dataset.metadata, 0, cfg)
    dataset.frame["outcome"] = ["yes", "yes", "no"]
    new_meta = extract_metadata(
        dataset.frame,
        dataset.official,
        dataset_id="TEST_TEXT",
        dataset_name="Example",
        text_columns=["short", "long"],
    )
    after = build_requests(dataset.features, new_meta, 0, cfg)
    assert before == after
    with pytest.raises(ValueError, match="never the target"):
        build_requests(dataset.frame, dataset.metadata, 0, cfg)
    with pytest.raises(ValueError):
        build_requests(dataset.features.assign(target_mean=0.5), dataset.metadata, 0, cfg)


def test_regression_targets_do_not_affect_metadata_or_keys(cfg):
    dataset = example("reg")
    before = build_requests(dataset.features, dataset.metadata, 0, cfg)
    dataset.frame["outcome"] = [2000.0, -99.0, 0.002]
    new_meta = extract_metadata(
        dataset.frame,
        dataset.official,
        dataset_id="TEST_TEXT",
        dataset_name="Example",
        text_columns=["short", "long"],
    )
    assert before == build_requests(dataset.features, new_meta, 0, cfg)


def test_question_key_tracks_version_configuration_and_content(cfg):
    dataset = example()
    original = build_requests(dataset.features, dataset.metadata, 0, cfg)[0]
    for field, value in [
        ("version", "verified-test-v1"),
        ("model", "test-model"),
        ("request_configuration", {"future_parameter": 1}),
    ]:
        variant = copy.deepcopy(cfg)
        variant["jev"][field] = value
        assert (
            build_requests(dataset.features, dataset.metadata, 0, variant)[0].cache_key
            != original.cache_key
        )
    reordered = dataset.features[dataset.features.columns[::-1]]
    assert build_requests(reordered, dataset.metadata, 0, cfg)[0].cache_key == original.cache_key
    other_trace = replace(original, provenance_json='{"row_id":"different"}')
    assert other_trace.cache_key == original.cache_key
    tampered = original.to_dict()
    tampered["payload"]["question"]["text"] = "changed"
    with pytest.raises(ValueError, match="checksum"):
        IntendedRequest.from_json(canonical_json(tampered))


def test_single_text_combined_has_two_slots_but_one_content_key(cfg):
    dataset = example(single=True)
    requests = build_requests(dataset.features, dataset.metadata, 0, cfg)
    assert len(requests) == 2
    assert requests[0].cache_key == requests[1].cache_key
    assert requests[0].provenance_json != requests[1].provenance_json


def test_cache_resume_and_namespace(tmp_path, cfg):
    dataset = example()
    request = build_requests(dataset.features, dataset.metadata, 0, cfg)[0]
    path = tmp_path / "responses.sqlite3"
    cache = ResponseCache(path)
    assert dry_run(request, cache)["cache_hit"] is False
    assert dry_run(request, ResponseCache(path))["cache_hit"] is True
    assert cache.get(request, kind="verified_response") is None
    cfg["jev"]["version"] = None
    unpinned_request = build_requests(dataset.features, dataset.metadata, 0, cfg)[0]
    with pytest.raises(ValueError, match="version pins"):
        cache.get(unpinned_request, kind="verified_response")
    with pytest.raises(ValueError, match="replace"):
        cache.put(request, {"bad": True})
    with pytest.raises(NotImplementedError):
        DisabledJevClient().evaluate(request)


@pytest.mark.parametrize("task", ["cls", "reg"])
def test_analytical_cost_equals_serialising_every_request(cfg, task):
    dataset = example(task)
    cfg["cost"]["input_price_per_million"] = 2.0
    estimates = estimate_dataset(dataset.metadata, request_lengths(dataset), cfg)
    for estimate in estimates:
        variant = copy.deepcopy(cfg)
        variant["jev"].update(
            representation=estimate["mode"],
            include_non_text_features=estimate["include_non_text_features"],
        )
        requests = [
            request
            for i in range(len(dataset.frame))
            for request in build_requests(dataset.features, dataset.metadata, i, variant)
        ]
        assert estimate["requests"] == len(requests)
        exact = sum(int(np.ceil(len(r.payload_json) / 4)) for r in requests)
        assert estimate["input_tokens_approx"] == exact
        assert estimate["input_cost_estimate"] == pytest.approx(exact * 2 / 1e6)


def test_audit_statistics_and_missingness(cfg):
    dataset = example()
    table, details = audit_dataset(dataset, cfg)
    short = next(t for t in table["text"] if t["feature"] == "short")
    assert short["missing_pct"] == pytest.approx(100 / 3)
    assert short["n_unique"] == 1
    assert short["unique_ratio"] == 0.5
    assert short["characters_mean"] == pytest.approx(8 / 3)
    assert table["summary"]["total_text_tokens"] >= 2
    assert sum(x["count"] for x in details["target_audit_only"]["distribution"]) == 3
    assert "target_audit_only" not in details["task_metadata"]


@pytest.mark.parametrize("field", ["jev", "downstream"])
def test_real_work_cannot_be_enabled(cfg, field):
    cfg[field]["enabled"] = True
    with pytest.raises(ValueError, match="disabled"):
        validate_config(cfg)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, pd.NA, pd.NaT, None])
def test_nonfinite_values_serialize_to_null(value):
    assert json.dumps(scalar(value), allow_nan=False) == "null"


def test_non_text_changes_never_change_semantic_requests(cfg):
    dataset = example()
    before = [r.cache_key for r in build_requests(dataset.features, dataset.metadata, 0, cfg)]
    dataset.frame.loc[0, "number"] = 999
    dataset.frame.loc[0, "category"] = "changed"
    assert before == [
        r.cache_key for r in build_requests(dataset.features, dataset.metadata, 0, cfg)
    ]
    cfg["jev"]["include_non_text_features"] = True
    with pytest.raises(ValueError, match="Non-text"):
        build_requests(dataset.features, dataset.metadata, 0, cfg)


def test_missing_groups_skip_without_renumbering_slots(cfg):
    dataset = example()
    dataset.frame.loc[0, "short"] = "  "
    requests = build_requests(dataset.features, dataset.metadata, 0, cfg)
    assert len(requests) == 2
    assert [r.to_dict()["provenance"]["representation_slot"] for r in requests] == [1, 2]
    assert requests[0].cache_key == requests[1].cache_key
    assert all(set(r.to_dict()["payload"]["state"]["text"]) == {"long"} for r in requests)
    assert build_requests(dataset.features, dataset.metadata, 2, cfg) == []
