"""Read-only, exhaustive checks of frozen questions, unique inputs and original-row links.

This validates construction, not linguistic truth or model interpretation. It reads every
active input and every row/slot against the pinned raw data, without inference or downloads.
Only one compact audit receipt is written; prepared databases remain byte-identical.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from src.data.loaders import load_catalog, load_dataset
from src.data.selection import selected_entries
from src.jev.inputs import INPUT_SCHEMA, _read_set
from src.jev.question_bank import build_question_bank, load_question_bank, payload_from_text
from src.jev.requests import require_nonempty_text, text_groups
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.serialization import canonical_json, digest, file_sha256, text_value, write_json


def audit_dataset(dataset, path, expected_entry, *, variant="documented", progress=None):
    """Check every active input and row link; no sampling or database writes."""
    path = Path(path).resolve(strict=True)
    before = file_sha256(path)
    meta = dataset.metadata
    groups = text_groups(meta, "per_column_and_joint")
    slots = [
        {
            "slot": i,
            "representation": "per_column" if i < len(meta.text_columns) else "joint",
            "text_columns": list(group),
        }
        for i, group in enumerate(groups)
    ]
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("PRAGMA trusted_schema=OFF")
        db.execute("BEGIN")
        if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise ValueError("Prepared database integrity failed")
        if db.execute("PRAGMA foreign_key_check").fetchone():
            raise ValueError("Prepared database has broken row/input links")
        identity, recipe, summary, entry = _read_set(db, None)
        expected = {
            "schema": INPUT_SCHEMA,
            "dataset_id": meta.dataset_id,
            "data_identity": dataset.source["data_identity"],
            "source_sha256": dataset.source.get("sha256", {}),
            "task_schema_sha256": meta.fingerprint,
            "row_count": len(dataset.frame),
            "row_id_format": "csv:<zero-based-original-position>",
            "question_variant": variant,
            "missing_text_policy": "skip_empty_store_nan",
            "text_policy": "verbatim_no_truncation",
            "slots": slots,
        }
        if path.stem != meta.dataset_id or any(recipe.get(k) != v for k, v in expected.items()):
            raise ValueError("Prepared recipe differs from pinned data/current preparation policy")
        if canonical_json(entry) != canonical_json(expected_entry):
            raise ValueError(
                "Saved wording/options/definitions differ from current source templates"
            )
        question = entry["variants"][variant]["question"]
        wording = question["text"]
        if (
            not wording.strip()
            or wording != wording.strip()
            or "\ufffd" in wording
            or re.search(r"\{[^{}]+\}|[\x00-\x1f]", wording)
        ):
            raise ValueError("Question contains a placeholder, control character or broken Unicode")

        unique = 0
        for number, sha, raw in db.execute(
            "SELECT input_no,payload_sha256,text_json FROM unique_inputs "
            "WHERE input_set_id=? ORDER BY input_no",
            (identity,),
        ):
            if number != unique:
                raise ValueError("Unique input numbering has a gap")
            text = json.loads(raw)
            payload = payload_from_text(entry, variant, text)
            require_nonempty_text(payload)
            if canonical_json(text) != raw or digest(payload) != sha:
                raise ValueError(f"Input {number}: text/payload checksum mismatch")
            if set(payload) != {"state", "question"} or set(payload["state"]) != {"task", "text"}:
                raise ValueError("Unapproved row context in prepared input")
            if meta.target in text or set(text) - set(meta.text_columns):
                raise ValueError("Target or non-text column in prepared input")
            facts = payload["state"]["task"].get("documentary_definition", {})
            if "text_column_definitions" in facts and set(facts["text_column_definitions"]) != set(
                text
            ):
                raise ValueError("Definitions do not match the supplied text columns")
            if variant == "documented" and "source_reference" in facts:
                raise ValueError("Provenance reference entered the model context")
            unique += 1
            if progress and unique % 25000 == 0:
                progress(f"  checked {unique:,} unique input hashes")

        # Independent comparison with the raw text catches a valid hash assigned to
        # the WRONG row or column; hash/integrity checks alone cannot find that mistake.
        mapping = iter(
            db.execute(
                "SELECT m.row_position,m.slot,m.input_no,u.text_json FROM row_map m "
                "LEFT JOIN unique_inputs u ON m.input_set_id=u.input_set_id AND m.input_no=u.input_no "
                "WHERE m.input_set_id=? ORDER BY m.row_position,m.slot",
                (identity,),
            )
        )
        empty, nonempty = [0] * len(groups), [0] * len(groups)
        for position, row in enumerate(
            dataset.frame.loc[:, list(meta.text_columns)].itertuples(
                index=False,
                name=None,
            )
        ):
            available = {
                col: value
                for col, raw in zip(meta.text_columns, row)
                if (value := text_value(raw)) is not None
            }
            for slot, group in enumerate(groups):
                record = next(mapping, None)
                if record is None or record[:2] != (position, slot):
                    raise ValueError(
                        f"Missing/out-of-order row link at row {position}, slot {slot}"
                    )
                _, _, number, raw = record
                text = {col: available[col] for col in group if col in available}
                if not text:
                    if number is not None or raw is not None:
                        raise ValueError(
                            f"Empty text linked to a request at row {position}, slot {slot}"
                        )
                    empty[slot] += 1
                else:
                    if number is None or raw != canonical_json(text):
                        raise ValueError(f"Wrong row text at row {position}, slot {slot}")
                    nonempty[slot] += 1
            if progress and (position + 1) % 25000 == 0:
                progress(f"  checked all text slots for {position + 1:,} original rows")
        if next(mapping, None) is not None:
            raise ValueError("Unexpected extra row/slot links")
        used = db.execute(
            "SELECT COUNT(DISTINCT input_no) FROM row_map WHERE input_set_id=?",
            (identity,),
        ).fetchone()[0]
        if used != unique:
            raise ValueError("A stored input has no original-row link")
        observed = {
            "dataset_id": meta.dataset_id,
            "dataset": meta.dataset_name,
            "input_set_id": identity,
            "bank_id": recipe["bank_id"],
            "variant": variant,
            "rows": len(dataset.frame),
            "text_columns": len(meta.text_columns),
            "slots_total": len(dataset.frame) * len(groups),
            "empty_slots": sum(empty),
            "nonempty_slots": sum(nonempty),
            "unique_inputs": unique,
            "reused_slots": sum(nonempty) - unique,
            "avoided_percent": 100 * (1 - unique / sum(nonempty)) if sum(nonempty) else 0.0,
            "per_slot": [
                {**s, "empty_slots": empty[i], "nonempty_slots": nonempty[i]}
                for i, s in enumerate(slots)
            ],
        }
        if canonical_json(summary) != canonical_json(observed):
            raise ValueError("Saved preparation summary differs from exhaustive counts")
    if file_sha256(path) != before:
        raise ValueError("Question file changed during read-only audit; stop writers and retry")
    return {
        **observed,
        "status": "mechanical_checks_passed",
        "file_sha256": before,
        "question_version": entry["variants"][variant]["version"],
        "question_sha256": digest(question),
        "source_review_notes": entry.get("semantics", {}).get("review_notes", []),
        "linguistic_correctness_proven": False,
    }


def run(*, phase="question_preparation", progress=print):
    """Audit exactly the configured datasets and save one current, compact receipt."""
    report = {
        "schema": "jevpfn.question_audit.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "failed",
        "scope": "every active unique input and every original row/text slot",
        "bank_id": None,
        "datasets": [],
        "inference_performed": False,
        "limitation": "Mechanical checks cannot prove linguistic truth or correct model interpretation.",
    }
    destination = paths.manifest_path("question_audit", phase=phase)
    try:
        config = load_yaml("question_preparation")
        entries = selected_entries(load_catalog(config["catalog"]), config)
        root = paths.decision_inputs_dir()
        if {p.stem for p in root.glob("*.sqlite3")} != {e["id"] for e in entries}:
            raise ValueError("Expected exactly the selected dataset question files")
        bank = load_question_bank(root)
        report["bank_id"] = bank["bank_id"]
        design = load_yaml("question_design")
        for i, entry in enumerate(entries, 1):
            if progress:
                progress(f"[{i}/{len(entries)}] {entry['name']}: exhaustive question/input audit")
            dataset = load_dataset(entry, allow_download=False)
            expected = build_question_bank([dataset.metadata], design)["datasets"][entry["id"]]
            result = audit_dataset(
                dataset,
                paths.dataset_question_path(entry["id"]),
                expected,
                variant=config["question_preparation"]["variant"],
                progress=progress,
            )
            if result["bank_id"] != bank["bank_id"]:
                raise ValueError("Active question bank changed during audit")
            report["datasets"].append(result)
        report["totals"] = {
            key: sum(r[key] for r in report["datasets"])
            for key in (
                "rows",
                "text_columns",
                "slots_total",
                "empty_slots",
                "nonempty_slots",
                "unique_inputs",
                "reused_slots",
            )
        }
        report["status"] = "mechanical_checks_passed_requires_semantic_review"
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        write_json(destination, report)
    if progress:
        progress(
            f"PASS: {len(entries)} datasets; {report['totals']['unique_inputs']:,} unique inputs; "
            f"{report['totals']['slots_total']:,} row/slot links. Receipt: {destination}"
        )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("question_preparation", "experiment_0"), default="question_preparation"
    )
    args = parser.parse_args(argv)
    run(phase=args.phase, progress=lambda message: print(message, flush=True))


if __name__ == "__main__":
    main()
