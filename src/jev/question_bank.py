"""Immutable shared questions, separate from row text, provider calls and paid responses.

Question generation is cheap and deterministic. Saving it freezes what was reviewed,
including answer ordering and source-backed definitions, for reuse by every extractor.
"""

import copy
import json
from pathlib import Path

from src.jev.design import DESIGN_VERSION, proposed_question, reviewed_definition
from src.jev.requests import IntendedRequest
from src.jev.semantics import DOCUMENTED_VERSION, dataset_semantics, documented_question
from src.jev.templates import TEMPLATE_VERSION, make_question
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.serialization import canonical_json, digest, write_json


def build_question_bank(metadata, design_cfg):
    """Use task schemas only. No row data, target statistics or provider settings."""
    if design_cfg["candidate_version"] != DESIGN_VERSION:
        raise ValueError("Question design version differs from implementation")
    if design_cfg["accepted_version"] != TEMPLATE_VERSION:
        raise ValueError("Accepted template version differs from implementation")
    records = {}
    semantics = (
        load_yaml(design_cfg["semantics_catalog"]) if design_cfg.get("semantics_catalog") else None
    )
    if semantics and design_cfg.get("documented_version") != DOCUMENTED_VERSION:
        raise ValueError("Documented question version differs from implementation")
    for meta in metadata:
        if meta.dataset_id in records:
            raise ValueError("Duplicate dataset in question bank")
        definition = reviewed_definition(
            meta, design_cfg["documentary_definitions"].get(meta.dataset_id, {})
        )
        records[meta.dataset_id] = {
            "task_schema_sha256": meta.fingerprint,
            "task": meta.task_context(),
            "text_columns": list(meta.text_columns),
            "variants": {
                "accepted": {
                    "version": TEMPLATE_VERSION,
                    "status": "accepted_template_pilot_still_required",
                    "question": make_question(meta),
                    "documentary_definition": {},
                },
                "proposed_direct": {
                    "version": DESIGN_VERSION,
                    "status": "proposed_not_accepted",
                    "question": proposed_question(meta),
                    "documentary_definition": definition,
                },
            },
        }
        if semantics and meta.dataset_id in semantics["datasets"]:
            entry = dataset_semantics(meta, semantics)
            records[meta.dataset_id]["semantics"] = entry
            records[meta.dataset_id]["variants"]["documented"] = {
                "version": DOCUMENTED_VERSION,
                "status": "source_backed_wording_for_review_not_live_approved",
                "question": documented_question(meta, entry),
                "documentary_definition": entry["documentary_definition"],
            }
    content = {"schema": "jevpfn.question_bank.v1", "datasets": records}
    return {"bank_id": digest(content), **content}


def validate_bank(bank):
    content = {k: v for k, v in bank.items() if k != "bank_id"}
    if content.get("schema") != "jevpfn.question_bank.v1" or digest(content) != bank["bank_id"]:
        raise ValueError("Question bank checksum/schema mismatch")


def save_question_bank(bank):
    """Content-addressed immutable file; changed wording creates a new file."""
    validate_bank(bank)
    destination = paths.question_bank_path(bank["bank_id"])
    if destination.exists():
        if load_question_bank(destination) != bank:
            raise ValueError("Refusing to replace a different question bank")
    else:
        write_json(destination, bank)
    # Durable dataset inputs live outside output cleanup. The complete bank file
    # remains the index/replay entry point and preserves the previous reader API.
    for dataset_id, entry in bank["datasets"].items():
        _save_dataset_question(
            bank, dataset_id, entry, paths.dataset_question_path(bank["bank_id"], dataset_id)
        )
    return destination


def _save_dataset_question(bank, dataset_id, entry, path):
    content = {
        "schema": "jevpfn.dataset_questions.v1",
        "bank_id": bank["bank_id"],
        "dataset_id": dataset_id,
        "entry_sha256": digest(entry),
        "entry": entry,
    }
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != content:
            raise ValueError("Refusing to replace different dataset questions")
    else:
        write_json(path, content)
    return path


def export_dataset_questions(bank):
    """Readable local outputs, one per dataset; durable originals remain under data/."""
    validate_bank(bank)
    folder = paths.notebook_outputs_dir() / "question_preparation" / "questions" / bank["bank_id"]
    return [
        _save_dataset_question(
            bank, dataset_id, entry, folder / f"{paths.output_component(dataset_id)}.json"
        )
        for dataset_id, entry in bank["datasets"].items()
    ]


def load_question_bank(path):
    bank = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_bank(bank)
    return bank


def apply_frozen_question(request, metadata, bank, variant="accepted"):
    """Reuse stored wording/options verbatim; changing row IDs never changes the question."""
    return compile_frozen_question(metadata, bank, variant)(request)


def compile_frozen_question(metadata, bank, variant="accepted"):
    """Validate the bank once, then reuse an isolated record for many row requests."""
    validate_bank(bank)
    entry = bank["datasets"][metadata.dataset_id]
    if entry["task_schema_sha256"] != metadata.fingerprint:
        raise ValueError("Question bank belongs to a different dataset/task schema")
    value = copy.deepcopy(entry["variants"][variant])
    bank_id = bank["bank_id"]
    reviewed_definition(metadata, value["documentary_definition"])

    def apply(request):
        return _apply_question(request, metadata, bank_id, variant, value)

    return apply


def _apply_question(request, metadata, bank_id, variant, value):
    payload = json.loads(request.payload_json)
    if payload["state"]["task"] != metadata.task_context():
        raise ValueError("Request task and question bank disagree")
    payload["question"] = copy.deepcopy(value["question"])
    facts = reviewed_definition(metadata, value["documentary_definition"])
    if facts:
        # Per-column calls need definitions of supplied fields only. This also
        # preserves separate/joint cache reuse when their available text is equal.
        if "text_column_definitions" in facts:
            facts["text_column_definitions"] = {
                k: v
                for k, v in facts["text_column_definitions"].items()
                if k in payload["state"]["text"]
            }
        payload["state"]["task"]["documentary_definition"] = facts
    trace = json.loads(request.provenance_json)
    trace.update(
        question_bank_id=bank_id,
        question_variant=variant,
        question_design_version=value["version"],
        design_status=value["status"],
        question_sha256=digest(value["question"]),
    )
    return IntendedRequest(canonical_json(payload), request.settings_json, canonical_json(trace))
