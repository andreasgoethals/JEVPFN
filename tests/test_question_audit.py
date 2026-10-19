"""The full input audit must detect valid-but-misaligned data, not merely corrupt files."""

import copy
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd
import pytest

from src.data.loaders import LoadedDataset
from src.data.metadata import extract_metadata
from src.jev.inputs import activate_input_set, prepare_dataset_inputs
from src.jev.question_audit import audit_dataset
from src.jev.question_bank import build_question_bank, save_question_bank
from src.utils.config import load_yaml
from src.utils.serialization import file_sha256


@pytest.fixture
def prepared(isolated_output):
    frame = pd.DataFrame(
        {
            "a": ['cafÃ©\nquoted "text"', "same", "same", None, " "],
            "b": ["other", "second", "second", 0, None],
            "nontext": [123] * 5,
            "target": [0, 1, 0, 1, 0],
        }
    )
    meta = extract_metadata(
        frame,
        {"task_type": "cls", "target": "target"},
        dataset_id="fixture",
        dataset_name="Fixture",
        text_columns=["a", "b"],
    )
    dataset = LoadedDataset(frame, meta, {}, {"data_identity": "fixed"}, [])
    bank = build_question_bank([meta], load_yaml("question_design"))
    save_question_bank(bank)
    summary = prepare_dataset_inputs(dataset, bank, variant="accepted")
    activate_input_set(summary["path"], summary["input_set_id"])
    return dataset, Path(summary["path"]), bank["datasets"]["fixture"]


def audit(prepared):
    return audit_dataset(*prepared, variant="accepted")


def mutate(path, query):
    with closing(sqlite3.connect(path)) as db, db:
        db.execute(query)


def test_exhaustive_read_only_audit_checks_unicode_duplicates_zero_and_empty(prepared):
    before = file_sha256(prepared[1])
    report = audit(prepared)
    assert report["status"] == "mechanical_checks_passed"
    assert report["rows"] == 5 and report["slots_total"] == 15
    assert report["empty_slots"] == 4
    assert report["unique_inputs"] == 7 and report["reused_slots"] == 4
    assert report["linguistic_correctness_proven"] is False
    assert file_sha256(prepared[1]) == before == report["file_sha256"]


def test_explicit_rebuild_preserves_question_identity_and_all_row_links(prepared):
    dataset, path, _ = prepared
    bank = build_question_bank([dataset.metadata], load_yaml("question_design"))
    before = audit(prepared)
    result = prepare_dataset_inputs(dataset, bank, variant="accepted", rebuild=True)
    after = audit(prepared)
    assert result["reused"] is False
    assert before["input_set_id"] == after["input_set_id"]
    assert before["unique_inputs"] == after["unique_inputs"]
    assert before["slots_total"] == after["slots_total"]


@pytest.mark.parametrize(
    "sql,match",
    [
        ("UPDATE row_map SET input_no=3 WHERE row_position=0 AND slot=0", "Wrong row text"),
        ("UPDATE row_map SET input_no=NULL WHERE row_position=0 AND slot=0", "Wrong row text"),
        ("UPDATE row_map SET input_no=0 WHERE row_position=4 AND slot=0", "Empty text linked"),
        ("DELETE FROM row_map WHERE row_position=2 AND slot=0", "Missing/out-of-order"),
        ("UPDATE row_map SET row_position=55 WHERE row_position=4", "Missing/out-of-order"),
        ("UPDATE row_map SET input_no=999 WHERE row_position=0 AND slot=0", "broken row/input"),
        ('UPDATE unique_inputs SET text_json=\'{"a":"tampered"}\' WHERE input_no=0', "checksum"),
    ],
)
def test_audit_rejects_corruption_and_valid_inputs_linked_to_wrong_rows(prepared, sql, match):
    mutate(prepared[1], sql)
    with pytest.raises(ValueError, match=match):
        audit(prepared)


def test_audit_rejects_template_drift_without_modifying_frozen_questions(prepared):
    dataset, path, expected = prepared
    changed = copy.deepcopy(expected)
    changed["variants"]["accepted"]["question"]["text"] += " Changed."
    with pytest.raises(ValueError, match="differ from current source"):
        audit_dataset(dataset, path, changed, variant="accepted")


def test_target_and_non_text_row_changes_cannot_affect_question_audit(prepared):
    before = audit(prepared)
    prepared[0].frame["target"] = [1, 0, 1, 0, 1]
    prepared[0].frame["nontext"] = [999] * 5
    assert audit(prepared) == before


def test_audit_rejects_changed_raw_text_even_if_input_database_itself_is_valid(prepared):
    prepared[0].frame.loc[0, "a"] = "different original row"
    with pytest.raises(ValueError, match="Wrong row text"):
        audit(prepared)


def test_failed_start_replaces_prior_success_receipt(isolated_output):
    from src.jev.question_audit import run
    from src.utils import paths
    from src.utils.serialization import write_json

    path = paths.manifest_path("question_audit", phase="question_preparation")
    write_json(path, {"status": "old_pass"})
    with pytest.raises(ValueError, match="exactly the selected"):
        run(progress=None)
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["status"] == "failed" and report["bank_id"] is None
    assert report["inference_performed"] is False


def test_interrupted_rebuild_rolls_back_existing_input_set(prepared, monkeypatch):
    from src.jev import inputs
    dataset, _, _ = prepared
    bank = build_question_bank([dataset.metadata], load_yaml("question_design"))
    before = audit(prepared)
    def interrupted(*args):
        raise RuntimeError("interrupted rebuild")
    monkeypatch.setattr(inputs, "_flush", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        prepare_dataset_inputs(dataset, bank, variant="accepted", rebuild=True)
    assert audit(prepared) == before
