"""Offline checks for feature widths, reviewed wording, context screens and pilot identities."""

import copy
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from src.data.loaders import LoadedDataset
from src.data.metadata import extract_metadata
from src.jev.design import proposed_request, reviewed_definition
from src.jev.dimensions import numeric_feature_dimensions
from src.jev.pilot import dataset_pilot_requests, prepare_pilot, save_pilot_manifest
from src.jev.requests import build_requests
from src.jev.review import context_screen_examples, review_dataset
from src.utils import paths
from src.utils.config import load_config, load_yaml


def fixture_dataset(task="cls", texts=2, classes=3):
    frame = pd.DataFrame(
        {
            "a": ["a" * 1000, "repeat", "repeat", None, "", "other"],
            "n": range(6),
            "y": [i % classes for i in range(6)],
        }
    )
    if texts == 2:
        frame["b"] = [None, "b" * 1300, "b" * 1300, None, "b", "short"]
    if task == "reg":
        frame["y"] = frame["y"] * 10.5
    columns = ["a", "b"] if texts == 2 else ["a"]
    official = {"target": "y", "task_type": task}
    meta = extract_metadata(
        frame, official, dataset_id="fake", dataset_name="Fake", text_columns=columns
    )
    return LoadedDataset(frame, meta, official, {"data_identity": "fixed"}, [])


@pytest.mark.parametrize("task,classes,width", [("cls", 2, 1), ("cls", 3, 3), ("reg", 3, 1)])
@pytest.mark.parametrize("texts", [1, 2])
def test_feature_widths_are_independent_of_row_counts_and_duplicates(task, classes, width, texts):
    dataset = fixture_dataset(task, texts, classes)
    audit = SimpleNamespace(
        summary=pd.DataFrame([{"dataset_id": "fake", "rows": 6}]),
        metadata=lambda _: dataset.metadata,
    )
    table = numeric_feature_dimensions(audit, excluded=["fake"]).iloc[0]
    assert table.status == "REMOVED"
    assert table.per_column_features == texts * width
    assert table.joint_features == width
    assert table.combined_features == (texts + (texts > 1)) * width
    assert table.predictor_columns_with_combined == table.combined_features + 1
    assert table.combined_numeric_values == 6 * table.combined_features
    assert table.combined_float64_mib == pytest.approx(8 * table.combined_numeric_values / 1024**2)


@pytest.mark.parametrize("task,classes", [("cls", 2), ("cls", 3), ("reg", 3)])
def test_proposed_wording_preserves_text_options_and_accepted_templates(task, classes):
    dataset = fixture_dataset(task=task, classes=classes)
    accepted = build_requests(dataset.features, dataset.metadata, 0, load_config())[0]
    before = accepted.to_json()
    candidate = proposed_request(accepted, dataset.metadata)
    original, changed = map(lambda r: json.loads(r.payload_json), [accepted, candidate])
    assert original["state"] == changed["state"]
    assert original["question"]["kind"] == changed["question"]["kind"]
    for key in set(original["question"]) - {"text"}:
        assert original["question"][key] == changed["question"][key]
    assert accepted.to_json() == before and accepted.cache_key != candidate.cache_key
    assert json.loads(candidate.provenance_json)["design_status"] == "proposed_not_accepted"


def test_documentary_context_rejects_label_statistics_and_unknown_columns():
    meta = fixture_dataset().metadata
    for definition in (
        {"class_frequencies": {}},
        {"source_reference": "source", "target_mean": 1},
        {"target_definition": "meaning"},
        {"source_reference": "source", "text_column_definitions": {"n": "number"}},
        {"source_reference": "source", "class_definitions": {"c999": "invented"}},
    ):
        with pytest.raises(ValueError):
            reviewed_definition(meta, definition)
    definition = {
        "source_reference": "official schema",
        "observation_unit": "case",
        "target_definition": "a declared category",
        "class_definitions": {"c000": "declared meaning"},
    }
    assert reviewed_definition(meta, definition) == definition


def test_context_screen_counts_distinct_text_groups_not_rows_or_classes():
    dataset, cfg = fixture_dataset(), load_config("feature_creation")
    limit = 500
    details = context_screen_examples(dataset, cfg, limit=limit)
    records, _ = review_dataset(dataset, cfg)
    expected = sum(
        r["requests_for_combined_build"] for r in records if r["characters_per_token"] == 3
    )
    assert 0 < len(details) <= expected
    assert details.request_key.is_unique
    assert details.input_tokens_proxy.gt(limit).all()
    assert not details.columns.isin(["y", "text", "target_value"]).any()
    # The long b value is repeated on two rows and needs only one per-column request.
    assert (
        len(
            details.loc[
                details["mode"].eq("per_column") & details.text_columns.map(lambda x: x == ["b"])
            ]
        )
        <= 1
    )


def test_offline_pilot_has_distinct_provider_identities_and_no_raw_text(monkeypatch):
    dataset = fixture_dataset()
    monkeypatch.setattr("src.jev.pilot.load_dataset", lambda *a, **k: dataset)
    monkeypatch.setattr("src.jev.pilot.code_identity", lambda: "test-code")
    cfg, pilot_cfg = load_config("feature_creation"), load_yaml("decision_pilot")
    pilot_cfg["question_variants"] = [
        "accepted",
        "proposed_direct",
    ]  # Synthetic task has no source dictionary.
    catalog = load_yaml("extractors")
    manifest = prepare_pilot(cfg, pilot_cfg, catalog, entries=[{"id": "fake"}])
    assert manifest["network_calls"] == manifest["models_loaded"] == 0
    assert manifest["question_bank_id"]
    assert all(
        r["question_bank_id"] == manifest["question_bank_id"] and r["request_key"]
        for r in manifest["records"]
    )
    assert len(manifest["providers"]) == 11
    assert not any("a" * 1000 in json.dumps(record) for record in manifest["records"])
    keys = {
        provider: {
            r["provider_request_identity"] for r in manifest["records"] if r["provider"] == provider
        }
        for provider in manifest["providers"]
    }
    assert not keys["jev"] & keys["intern"]
    assert (
        manifest["providers"]["jev"]["unique_planned_requests"]
        == manifest["providers"]["liquid"]["unique_planned_requests"]
    )
    changed = copy.deepcopy(dataset)
    changed.frame["y"] = changed.frame["y"].iloc[::-1].to_numpy()
    first = dataset_pilot_requests(dataset, cfg)
    second = dataset_pilot_requests(changed, cfg)
    assert first == second  # Same vocabulary, new row targets: no pilot-input change.
    pilot_cfg["enabled"] = True
    with pytest.raises(ValueError, match="offline"):
        prepare_pilot(cfg, pilot_cfg, catalog, entries=[])


def test_large_pilot_trace_and_small_summary_use_separate_vsc_tiers(isolated_output):
    manifest = {"dataset_ids": ["fixture"], "network_calls": 0, "records": [{"row_id": "csv:0"}]}
    full = save_pilot_manifest(manifest)
    assert full.is_relative_to(paths.large_outputs_dir())
    compact = next((paths.manifests_dir("feature_creation") / "pilots").glob("*.summary.json"))
    assert not compact.is_relative_to(paths.large_outputs_dir())
    record = json.loads(compact.read_text(encoding="utf-8"))
    assert "records" not in record and record["source_trace_records"] == 1
    assert record["full_manifest_path"] == str(full)
