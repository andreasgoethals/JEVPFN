"""Immutable shared questions, separate from row text, provider calls and paid responses.

Question generation is cheap and deterministic. Saving it freezes what was reviewed,
including answer ordering and source-backed definitions, for reuse by every extractor.
"""

import copy
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from src.jev.design import DESIGN_VERSION, proposed_question, reviewed_definition
from src.jev.requests import IntendedRequest
from src.jev.semantics import DOCUMENTED_VERSION, dataset_semantics, documented_question
from src.jev.templates import TEMPLATE_VERSION, make_question
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.serialization import canonical_json, digest


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
    """Save each entry once, inside its dataset file; older bank versions coexist."""
    validate_bank(bank)
    for dataset_id, entry in bank["datasets"].items():
        destination = paths.ensure(paths.dataset_question_path(dataset_id))
        with closing(sqlite3.connect(destination, timeout=120)) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS question_banks (
                bank_id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL,
                members_json TEXT NOT NULL, entry_json TEXT NOT NULL, entry_sha256 TEXT NOT NULL
            )""")
            if db.execute(
                "SELECT 1 FROM question_banks WHERE dataset_id != ?", (dataset_id,)
            ).fetchone():
                raise ValueError("Question file belongs to a different dataset")
            values = (
                bank["bank_id"],
                dataset_id,
                canonical_json(sorted(bank["datasets"])),
                canonical_json(entry),
                digest(entry),
            )
            existing = db.execute(
                "SELECT * FROM question_banks WHERE bank_id=?", (bank["bank_id"],)
            ).fetchone()
            if existing is not None and existing != values:
                raise ValueError("Question entry checksum/content mismatch")
            if existing is None:
                db.execute("INSERT INTO question_banks VALUES (?,?,?,?,?)", values)
    return paths.decision_inputs_dir()


def load_question_bank(path, bank_id=None):
    """Reassemble a bank from its 18 dataset files; no second catalog copy is needed.

    A legacy JSON may still be read for provenance migration. Multiple versions require
    an explicit bank ID, so a later extraction cannot silently select different questions.
    """
    path = Path(path)
    if path.suffix == ".json":
        bank = json.loads(path.read_text(encoding="utf-8"))
        validate_bank(bank)
        return bank
    records, members = {}, None
    candidates = sorted(path.glob("*.sqlite3"))
    for candidate in candidates:
        with closing(sqlite3.connect(candidate.as_uri() + "?mode=ro", uri=True)) as db:
            rows = db.execute("SELECT * FROM question_banks").fetchall()
        if bank_id is None:
            available = {row[0] for row in rows}
            if len(available) != 1:
                raise ValueError("Specify bank_id when multiple question versions exist")
            bank_id = next(iter(available))
        for identity, dataset_id, membership, entry_json, sha in rows:
            if identity != bank_id:
                continue
            entry = json.loads(entry_json)
            expected = json.loads(membership)
            if digest(entry) != sha or (members is not None and members != expected):
                raise ValueError("Question entry checksum/membership mismatch")
            if candidate.stem != dataset_id or dataset_id in records:
                raise ValueError("Question filename/dataset identity mismatch")
            records[dataset_id], members = entry, expected
    if not members or sorted(records) != members:
        raise ValueError("Question bank is incomplete")
    bank = {"bank_id": bank_id, "schema": "jevpfn.question_bank.v1", "datasets": records}
    validate_bank(bank)
    return bank


def payload_from_text(entry, variant, text):
    """Reconstruct a provider-neutral payload from shared metadata and one unique text set."""
    if not text or set(text) - set(entry["text_columns"]):
        raise ValueError("Use a nonempty subset of the official text columns")
    task = copy.deepcopy(entry["task"])
    value = entry["variants"][variant]
    facts = copy.deepcopy(value["documentary_definition"])
    if facts:
        if "text_column_definitions" in facts:
            facts["text_column_definitions"] = {
                k: v for k, v in facts["text_column_definitions"].items() if k in text
            }
        task["documentary_definition"] = facts
    return {"state": {"task": task, "text": text}, "question": copy.deepcopy(value["question"])}


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
