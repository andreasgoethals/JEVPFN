"""Shared immutable questions, truthful exact previews and separate provider caches."""

import copy
import json
import sqlite3
from contextlib import closing
from dataclasses import replace

import pandas as pd
import pytest

from src.data.loaders import LoadedDataset
from src.data.metadata import extract_metadata
from src.jev.cache import ResponseCache
from src.jev.catalog_inspection import dataset_example
from src.jev.identity import bind_provider
from src.jev.provider import api_preview
from src.jev.question_bank import (
    apply_frozen_question,
    build_question_bank,
    load_question_bank,
    save_question_bank,
)
from src.jev.requests import build_requests
from src.utils import paths
from src.utils.config import load_config, load_yaml


@pytest.fixture
def dataset():
    frame = pd.DataFrame(
        {
            "a": [None, "preserve  spaces\nand newlines", "short"],
            "b": ["other", "second text", ""],
            "numeric": [123456789, 123456789, 123456789],
            "target": [0, 1, 0],
        }
    )
    official = {"target": "target", "task_type": "cls"}
    metadata = extract_metadata(
        frame, official, dataset_id="fixture", dataset_name="Fixture", text_columns=["a", "b"]
    )
    return LoadedDataset(frame, metadata, official, {"data_identity": "fixed"}, [])


def test_question_bank_round_trip_stability_and_tamper_rejection(dataset, isolated_output):
    bank = build_question_bank([dataset.metadata], load_yaml("question_design"))
    destination = save_question_bank(bank)
    dataset_file = paths.dataset_question_path("fixture")
    before = dataset_file.stat().st_mtime_ns
    assert save_question_bank(bank) == destination
    assert dataset_file.stat().st_mtime_ns == before
    assert load_question_bank(destination) == bank
    assert destination.is_relative_to(paths.staging_root())
    with closing(sqlite3.connect(dataset_file)) as db:
        stored = db.execute("SELECT entry_json FROM question_banks").fetchone()[0]
    assert "123456789" not in stored
    broken = copy.deepcopy(bank)
    broken["datasets"]["fixture"]["variants"]["accepted"]["question"]["text"] = "tampered"
    with closing(sqlite3.connect(dataset_file)) as db, db:
        db.execute(
            "UPDATE question_banks SET entry_json=?", (json.dumps(broken["datasets"]["fixture"]),)
        )
    with pytest.raises(ValueError, match="checksum"):
        load_question_bank(destination)


@pytest.mark.parametrize(
    "task,classes",
    [
        ("binary_classification", (0, 1)),
        ("multiclass_classification", (0, 1, 2)),
        ("regression", ()),
    ],
)
def test_frozen_questions_preserve_options_and_identify_row(dataset, task, classes):
    dataset.metadata = replace(dataset.metadata, task_type=task, classes=classes)
    bank = build_question_bank([dataset.metadata], load_yaml("question_design"))
    original = build_requests(dataset.features, dataset.metadata, 1, load_config())[0]
    accepted = apply_frozen_question(original, dataset.metadata, bank)
    candidate = apply_frozen_question(original, dataset.metadata, bank, "proposed_direct")
    assert accepted.cache_key == original.cache_key  # Trace does not invalidate paid reuse.
    assert candidate.cache_key != original.cache_key
    body = json.loads(candidate.payload_json)
    assert body["state"] == json.loads(original.payload_json)["state"]
    assert "dataset row described by the supplied text" in body["question"]["text"]
    assert "For this case" not in body["question"]["text"]
    assert {k: v for k, v in body["question"].items() if k != "text"} == {
        k: v for k, v in json.loads(original.payload_json)["question"].items() if k != "text"
    }
    with pytest.raises(ValueError, match="schema"):
        apply_frozen_question(original, replace(dataset.metadata, target="changed"), bank)


def test_all_six_share_saved_semantics_but_not_cache_entries(dataset, tmp_path):
    catalog = load_yaml("extractors")
    request = build_requests(dataset.features, dataset.metadata, 1, load_config())[0]
    store = ResponseCache(tmp_path / "fixture.sqlite3", dataset_id="fixture")
    bound = {name: bind_provider(request, name, catalog) for name in catalog["candidates"]}
    assert len({r.cache_key for r in bound.values()}) == 7
    assert len({r.payload_json for r in bound.values()}) == 1
    for name, candidate in bound.items():
        store.put(candidate, {"mock_provider": name}, kind="mock")
    for name, candidate in bound.items():
        assert store.get(candidate, kind="mock") == {"mock_provider": name}
        if name != "jev":
            with pytest.raises(ValueError, match="Jev only"):
                api_preview(candidate)
    # Clef now has an immutable open checkpoint, separate from its former API identity.
    assert (
        json.loads(bound["cloudflare"].settings_json)["version"]
        == catalog["candidates"]["cloudflare"]["revision"]
    )


def test_gallery_uses_exact_row_text_all_groups_and_no_true_target(dataset, monkeypatch):
    monkeypatch.setattr("src.jev.catalog_inspection.code_identity", lambda: "fixture")
    bank = build_question_bank([dataset.metadata], load_yaml("question_design"))
    example = dataset_example(dataset, load_config(), bank, load_yaml("extractors"))
    assert example["row_id"] == "csv:1"
    assert len(example["groups"]) == 3
    assert [g["mode"] for g in example["groups"]] == ["per_column", "per_column", "joint"]
    assert (
        example["groups"][0]["variants"]["proposed_direct"]["semantic_payload"]["state"]["text"][
            "a"
        ]
        == dataset.frame.a.iloc[1]
    )
    assert "123456789" not in json.dumps(example)
    for group in example["groups"]:
        for variant in group["variants"].values():
            assert len(variant["provider_previews"]) == 7
            assert "row_id" not in variant["semantic_payload"]["state"]
            assert set(variant["semantic_payload"]["state"]) == {"task", "text"}
    changed = copy.deepcopy(dataset)
    changed.frame["target"] = [1, 0, 1]
    assert dataset_example(changed, load_config(), bank, load_yaml("extractors")) == example
