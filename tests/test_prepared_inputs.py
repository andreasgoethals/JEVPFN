"""Offline preparation must agree with the request builder and never lose row alignment."""

import copy
import json
import sqlite3
from contextlib import closing

import pandas as pd
import pytest

from src.data.loaders import LoadedDataset
from src.data.metadata import extract_metadata
from src.jev.inputs import input_origins, iter_prepared_inputs, prepare_dataset_inputs, prepared_row
from src.jev.question_bank import apply_frozen_question, build_question_bank, save_question_bank
from src.jev.requests import build_requests
from src.utils.config import load_config, load_yaml
from src.utils.serialization import digest


@pytest.fixture
def preparation(isolated_output):
    frame = pd.DataFrame(
        {
            "a": ["same", "same", "same ", None, " "],
            "b": ["same", "same", None, "only", None],
            "nontext": [999987654] * 5,
            "y": [0, 1, 0, 1, 1],
        }
    )
    meta = extract_metadata(
        frame,
        {"task_type": "cls", "target": "y"},
        dataset_id="fixture",
        dataset_name="Fixture",
        text_columns=["a", "b"],
    )
    dataset = LoadedDataset(frame, meta, {}, {"data_identity": "fixed"}, [])
    bank = build_question_bank([meta], load_yaml("question_design"))
    save_question_bank(bank)
    return dataset, bank


def test_grouping_preserves_exact_values_field_names_modes_and_missing(preparation):
    dataset, bank = preparation
    saved = prepare_dataset_inputs(dataset, bank, variant="accepted")
    assert saved["unique_inputs"] == 5
    assert saved["nonempty_slots"] == 10 and saved["empty_slots"] == 5
    assert saved["reused_slots"] == 5
    inputs = list(iter_prepared_inputs(saved["path"]))
    assert len(inputs) == 5
    assert "999987654" not in json.dumps(inputs)
    cfg = load_config()
    cfg["jev"]["representation"] = "per_column_and_joint"
    expected = {}
    for pos in range(len(dataset.frame)):
        for r in build_requests(dataset.features, dataset.metadata, pos, cfg):
            frozen = apply_frozen_question(r, dataset.metadata, bank)
            body, trace = json.loads(frozen.payload_json), json.loads(frozen.provenance_json)
            expected.setdefault(digest(body), []).append(
                (trace["row_id"], trace["representation_slot"])
            )
    assert {i["payload_sha256"] for i in inputs} == set(expected)
    for record in inputs:
        origins = list(input_origins(saved["path"], record["input_no"]))
        assert [(r["row_id"], r["slot"]) for r in origins] == expected[record["payload_sha256"]]
    assert all(r["status"] == "missing_no_request" for r in prepared_row(saved["path"], 4))
    with pytest.raises(IndexError):
        prepared_row(saved["path"], -1)


def test_rerun_reuses_completed_set_and_label_changes_cannot_change_inputs(preparation):
    dataset, bank = preparation
    saved = prepare_dataset_inputs(dataset, bank, variant="accepted")
    dataset.frame["y"] = [1, 0, 1, 0, 0]
    again = prepare_dataset_inputs(dataset, bank, variant="accepted")
    assert again == {**saved, "reused": True}
    changed = copy.deepcopy(dataset)
    changed.source["data_identity"] = "different-artifact"
    another = prepare_dataset_inputs(changed, bank, variant="accepted")
    assert another["input_set_id"] != saved["input_set_id"]
    with pytest.raises(ValueError, match="exactly one"):
        list(iter_prepared_inputs(saved["path"]))
    assert len(list(iter_prepared_inputs(saved["path"], input_set_id=saved["input_set_id"]))) == 5


def test_interruption_rolls_back_and_tampering_is_rejected(preparation, monkeypatch):
    import src.jev.inputs as module

    dataset, bank = preparation
    original = module._flush

    def broken(*args):
        original(*args)
        raise RuntimeError("interrupted")

    monkeypatch.setattr(module, "_flush", broken)
    with pytest.raises(RuntimeError, match="interrupted"):
        prepare_dataset_inputs(dataset, bank, variant="accepted")
    monkeypatch.setattr(module, "_flush", original)
    saved = prepare_dataset_inputs(dataset, bank, variant="accepted")
    assert not saved["reused"]
    with closing(sqlite3.connect(saved["path"])) as db, db:
        db.execute("UPDATE unique_inputs SET text_json=? WHERE input_no=0", ('{"a":"tampered"}',))
    with pytest.raises(ValueError, match="checksum"):
        list(iter_prepared_inputs(saved["path"]))


def test_prepared_documented_payload_matches_frozen_builder(preparation):
    dataset, bank = preparation
    # A documentary field definition should be included only when its text is supplied.
    variant = bank["datasets"]["fixture"]["variants"]["proposed_direct"]
    variant["documentary_definition"] = {
        "source_reference": "https://example.org/source",
        "text_column_definitions": {"a": "First field", "b": "Second field"},
    }
    bank["bank_id"] = digest({k: v for k, v in bank.items() if k != "bank_id"})
    save_question_bank(bank)
    saved = prepare_dataset_inputs(dataset, bank, variant="proposed_direct")
    cfg = load_config()
    cfg["jev"]["representation"] = "per_column_and_joint"
    request = build_requests(dataset.features, dataset.metadata, 3, cfg)[0]
    frozen = apply_frozen_question(request, dataset.metadata, bank, "proposed_direct")
    row = prepared_row(saved["path"], 3)
    assert row[1]["payload"] == json.loads(frozen.payload_json)
    assert row[1]["input_no"] == row[2]["input_no"]


def test_deleted_row_links_cannot_be_silently_reused(preparation):
    dataset, bank = preparation
    saved = prepare_dataset_inputs(dataset, bank, variant="accepted")
    with closing(sqlite3.connect(saved["path"])) as db, db:
        db.execute("DELETE FROM row_map WHERE row_position=0 AND slot=0")
    with pytest.raises(ValueError, match="incomplete"):
        prepare_dataset_inputs(dataset, bank, variant="accepted")
    with pytest.raises(ValueError, match="incomplete"):
        prepared_row(saved["path"], 0)
