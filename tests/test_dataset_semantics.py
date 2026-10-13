"""Source definitions and shared questions must preserve raw identities and label isolation."""

import copy
import json

import pandas as pd
import pytest

from src.data.loaders import LoadedDataset, load_catalog
from src.data.metadata import TaskMetadata, extract_metadata
from src.jev.preparation import preparation_tables
from src.jev.provider import api_preview
from src.jev.question_bank import (
    apply_frozen_question,
    build_question_bank,
    export_dataset_questions,
    save_question_bank,
)
from src.jev.requests import build_requests
from src.jev.semantics import dataset_semantics
from src.utils import paths
from src.utils.config import load_config, load_yaml


def metadata_catalog():
    semantics = load_yaml("dataset_semantics")
    entries = {e["id"]: e for e in load_catalog()["datasets"]}
    return [
        TaskMetadata(
            key,
            entries[key]["name"],
            item["raw_target"],
            item["task_type"],
            tuple(c["value"] for c in item["classes"]),
            tuple(entries[key]["text_columns"]),
            (),
        )
        for key, item in semantics["datasets"].items()
    ]


def test_every_selected_dataset_has_typed_source_backed_definitions():
    metadata = metadata_catalog()
    bank = build_question_bank(metadata, load_yaml("question_design"))
    assert len(bank["datasets"]) == 18
    tables = preparation_tables(bank)
    assert len(tables["text_fields"]) == 80
    assert len(tables["classes"]) == 60  # 6 binary labels + 54 multiclass labels.
    for meta in metadata:
        entry = bank["datasets"][meta.dataset_id]
        assert (
            entry["semantics"]["classes"]
            == load_yaml("dataset_semantics")["datasets"][meta.dataset_id]["classes"]
        )
        assert (
            entry["variants"]["documented"]["question"]["kind"]
            == entry["variants"]["accepted"]["question"]["kind"]
        )
    assert "ZOMATO" not in json.dumps(bank) and "SPOTIFY" not in json.dumps(bank)


def test_code_mappings_reject_changed_raw_values_or_text_columns():
    catalog = load_yaml("dataset_semantics")
    meta = metadata_catalog()[0]
    broken = copy.deepcopy(catalog)
    broken["datasets"][meta.dataset_id]["classes"][0]["value"] = False
    with pytest.raises(ValueError, match="vocabulary"):
        dataset_semantics(meta, broken)
    broken = copy.deepcopy(catalog)
    broken["datasets"][meta.dataset_id]["text_column_definitions"]["target"] = "Not permitted"
    with pytest.raises(ValueError, match="text columns"):
        dataset_semantics(meta, broken)


def test_translations_enter_wire_options_without_changing_original_classes():
    meta = next(m for m in metadata_catalog() if "DATA_SCIENTIST" in m.dataset_id)
    features = pd.DataFrame([{col: "same source text" for col in meta.text_columns}])
    bank = build_question_bank([meta], load_yaml("question_design"))
    request = build_requests(features, meta, 0, load_config())[0]
    mapped = apply_frozen_question(request, meta, bank, "documented")
    assert json.loads(mapped.payload_json)["question"]["allowed_answers"] == list(meta.classes)
    wire = api_preview(mapped)
    assert '"0-3"' in wire["questions"]["semantic"]["criteria"]["c000"]
    assert "lakh Indian rupees" in wire["questions"]["semantic"]["criteria"]["c000"]
    facts = wire["state"]["task"]["documentary_definition"]
    assert set(facts["text_column_definitions"]) == set(wire["state"]["text"])
    assert mapped.cache_key != request.cache_key


def test_questions_are_saved_per_dataset_and_exports_are_not_the_only_copy(isolated_output):
    bank = build_question_bank(metadata_catalog(), load_yaml("question_design"))
    path = save_question_bank(bank)
    exports = export_dataset_questions(bank)
    assert path.exists() and len(exports) == 18
    for file in exports:
        exported = json.loads(file.read_text(encoding="utf-8"))
        canonical = paths.dataset_question_path(bank["bank_id"], exported["dataset_id"])
        assert canonical.exists() and canonical.read_bytes() == file.read_bytes()
    before = exports[0].stat().st_mtime_ns
    export_dataset_questions(bank)
    assert exports[0].stat().st_mtime_ns == before


def test_source_corrections_are_explicit_and_not_inferred_from_targets():
    entries = load_yaml("dataset_semantics")["datasets"]
    assert "log10" in entries["REG_TEXT_BOOK_PRICE"]["target_unit"]
    assert "not" in " ".join(entries["REG_TEXT_BOOK_PRICE"]["review_notes"])
    assert "ln(" in entries["REG_TEXT_MERCARI_MARKETPLACE"]["target_unit"]
    assert all(
        "traffic-impact" in c["description"] for c in entries["MUL_TEXT_US_ACCIDENTS"]["classes"]
    )
    assert (
        entries["MUL_TEXT_PRODUCT_SENTIMENT"]["classes"][0]["description"]
        != entries["MUL_TEXT_PRODUCT_SENTIMENT"]["classes"][2]["description"]
    )


def test_frozen_question_materialization_matches_cached_request(isolated_output, tmp_path):
    from src.jev.cache import ResponseCache
    from src.jev.features import load_block, materialize_cached_block, save_block

    frame = pd.DataFrame(
        {
            "title": ["Example", "Example"],
            "salary_range": [None, None],
            "description": ["Description", "Description"],
            "fraudulent": [0, 1],
        }
    )
    meta = extract_metadata(
        frame,
        {"target": "fraudulent", "task_type": "cls"},
        dataset_id="BIN_TEXT_FAKE_JOB_POSTING",
        dataset_name="Fake Job Postings",
        text_columns=["title", "salary_range", "description"],
    )
    dataset = LoadedDataset(frame, meta, {}, {"data_identity": "synthetic"}, [])
    cfg = load_config("feature_creation")
    cfg["jev"]["representation"] = "per_column"
    bank = build_question_bank([meta], load_yaml("question_design"))
    cache = ResponseCache(tmp_path / "fixture.sqlite3", dataset_id=meta.dataset_id)
    for pos in range(len(frame)):
        for request in build_requests(dataset.features, meta, pos, cfg):
            frozen = apply_frozen_question(request, meta, bank, "documented")
            cache.put(
                frozen,
                {"model": "jev-1.13.0", "answers": {"semantic": {"type": "noul", "noul": 0.4}}},
                kind="verified_response",
            )
    block = materialize_cached_block(
        dataset, cfg, cache, question_bank=bank, question_variant="documented"
    )
    assert block.manifest["identity"]["question_bank_id"] == bank["bank_id"]
    assert block.values.filter(like="text_001").isna().all().all()
    pd.testing.assert_frame_equal(block.values, load_block(save_block(block)).values)
    with pytest.raises(ValueError, match="Missing verified response"):
        materialize_cached_block(dataset, cfg, cache)
