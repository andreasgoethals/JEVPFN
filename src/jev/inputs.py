"""One provider-neutral SQLite input file per dataset, with exact reuse and row alignment.

Only official text columns are read from the feature table. Questions are stored once;
unique text groups and their row/slot links are separate tables. No inference happens here.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from src.jev.question_bank import payload_from_text, validate_bank
from src.jev.requests import IntendedRequest, require_nonempty_text, text_groups
from src.utils import paths
from src.utils.serialization import canonical_json, digest, text_value

INPUT_SCHEMA = "jevpfn.prepared_inputs.v1"


def _schema(db):
    db.executescript("""
        CREATE TABLE IF NOT EXISTS input_sets (
            input_set_id TEXT PRIMARY KEY, recipe_json TEXT NOT NULL,
            summary_json TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS unique_inputs (
            input_set_id TEXT NOT NULL, input_no INTEGER NOT NULL,
            payload_sha256 TEXT NOT NULL, text_json TEXT NOT NULL,
            PRIMARY KEY (input_set_id, input_no),
            UNIQUE (input_set_id, payload_sha256),
            FOREIGN KEY (input_set_id) REFERENCES input_sets(input_set_id)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS row_map (
            input_set_id TEXT NOT NULL, row_position INTEGER NOT NULL,
            slot INTEGER NOT NULL, input_no INTEGER,
            PRIMARY KEY (input_set_id, row_position, slot),
            FOREIGN KEY (input_set_id, input_no) REFERENCES unique_inputs(input_set_id, input_no)
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS row_map_input ON row_map(input_set_id, input_no);
    """)


def prepare_dataset_inputs(dataset, bank, *, variant="documented", progress=None, rebuild=False):
    """Stream all rows into a transactional, deduplicated input set; reuse completed sets.

    Exact equality includes task, question/options, field names and untouched values.
    Neither source-row identity nor separate/joint slot is part of the semantic hash.
    A blank slot is recorded as NULL, not silently lost or sent to a provider.
    """
    validate_bank(bank)
    meta = dataset.metadata
    entry = bank["datasets"][meta.dataset_id]
    if variant not in entry["variants"]:
        raise ValueError("Unknown prepared question variant")
    if entry["task_schema_sha256"] != meta.fingerprint:
        raise ValueError("Question bank differs from dataset/task schema")
    if not dataset.source.get("data_identity"):
        raise ValueError("Prepared inputs require a pinned source-data identity")
    groups = text_groups(meta, "per_column_and_joint")
    slots = [
        {
            "slot": i,
            "representation": "per_column" if i < len(meta.text_columns) else "joint",
            "text_columns": list(group),
        }
        for i, group in enumerate(groups)
    ]
    recipe = {
        "schema": INPUT_SCHEMA,
        "dataset_id": meta.dataset_id,
        "data_identity": dataset.source["data_identity"],
        "source_sha256": dataset.source.get("sha256", {}),
        "task_schema_sha256": meta.fingerprint,
        "bank_id": bank["bank_id"],
        "question_variant": variant,
        "entry_sha256": digest(entry),
        "row_count": len(dataset.frame),
        "row_id_format": "csv:<zero-based-original-position>",
        "missing_text_policy": "skip_empty_store_nan",
        "text_policy": "verbatim_no_truncation",
        "slots": slots,
    }
    identity = digest(recipe)
    path = paths.dataset_question_path(meta.dataset_id)
    if not path.is_file():
        raise ValueError("Save the question bank before preparing dataset inputs")
    with closing(sqlite3.connect(path, timeout=120)) as db:
        db.execute("PRAGMA foreign_keys=ON")
        _schema(db)
        # One writer, crash rollback, no incomplete input set masquerading as complete.
        db.execute("BEGIN IMMEDIATE")
        try:
            existing = db.execute(
                "SELECT recipe_json,summary_json FROM input_sets WHERE input_set_id=?", (identity,)
            ).fetchone()
            if existing and rebuild:
                # Rebuild only this immutable recipe, transactionally. Historical
                # versions and paid response files are untouched; failure rolls back.
                for table in ("row_map", "unique_inputs", "input_sets"):
                    db.execute(f"DELETE FROM {table} WHERE input_set_id=?", (identity,))
            if existing and not rebuild:
                if json.loads(existing[0]) != recipe:
                    raise ValueError("Input recipe checksum mismatch")
                _, _, summary, _ = _read_set(db, identity)
                observed = tuple(
                    db.execute(query, (identity,)).fetchone()[0]
                    for query in (
                        "SELECT COUNT(*) FROM unique_inputs WHERE input_set_id=?",
                        "SELECT COUNT(*) FROM row_map WHERE input_set_id=?",
                    )
                )
                if observed != (summary["unique_inputs"], recipe["row_count"] * len(slots)):
                    raise ValueError("Prepared input set is incomplete; restore it before reuse")
                db.rollback()
                return {**summary, "path": str(path), "reused": True}
            db.execute(
                "INSERT INTO input_sets VALUES (?,?,?,?)",
                (identity, canonical_json(recipe), "{}", datetime.now(UTC).isoformat()),
            )
            seen, prefixes = {}, {}
            input_batch, map_batch = [], []
            empty, nonempty = [0] * len(groups), [0] * len(groups)
            # Restrict before iteration: true targets and non-text values never enter the loop.
            rows = dataset.frame.loc[:, list(meta.text_columns)].itertuples(index=False, name=None)
            for pos, row in enumerate(rows):
                available = {
                    col: value
                    for col, raw in zip(meta.text_columns, row)
                    if (value := text_value(raw)) is not None
                }
                for slot, group in enumerate(groups):
                    text = {col: available[col] for col in group if col in available}
                    if not text:
                        empty[slot] += 1
                        map_batch.append((identity, pos, slot, None))
                        continue
                    nonempty[slot] += 1
                    names = tuple(text)
                    if names not in prefixes:
                        payload = payload_from_text(entry, variant, text)
                        prefixes[names] = (
                            '{"question":'
                            + canonical_json(payload["question"])
                            + ',"state":{"task":'
                            + canonical_json(payload["state"]["task"])
                            + ',"text":'
                        )
                    text_json = canonical_json(text)
                    sha = hashlib.sha256(
                        (prefixes[names] + text_json + "}}").encode("utf-8")
                    ).hexdigest()
                    number = seen.get(sha)
                    if number is None:
                        number = len(seen)
                        seen[sha] = number
                        input_batch.append((identity, number, sha, text_json))
                    map_batch.append((identity, pos, slot, number))
                if len(map_batch) >= 10000:
                    _flush(db, input_batch, map_batch)
                if progress and (pos + 1) % 25000 == 0:
                    progress(
                        f"  {pos + 1:,}/{len(dataset.frame):,} rows; {len(seen):,} unique inputs"
                    )
            _flush(db, input_batch, map_batch)
            summary = {
                "dataset_id": meta.dataset_id,
                "dataset": meta.dataset_name,
                "input_set_id": identity,
                "bank_id": bank["bank_id"],
                "variant": variant,
                "rows": len(dataset.frame),
                "text_columns": len(meta.text_columns),
                "slots_total": len(dataset.frame) * len(groups),
                "empty_slots": sum(empty),
                "nonempty_slots": sum(nonempty),
                "unique_inputs": len(seen),
                "reused_slots": sum(nonempty) - len(seen),
                "avoided_percent": 100 * (1 - len(seen) / sum(nonempty)) if sum(nonempty) else 0.0,
                "per_slot": [
                    {**s, "empty_slots": empty[i], "nonempty_slots": nonempty[i]}
                    for i, s in enumerate(slots)
                ],
            }
            db.execute(
                "UPDATE input_sets SET summary_json=? WHERE input_set_id=?",
                (canonical_json(summary), identity),
            )
            db.commit()
        except BaseException:
            db.rollback()
            raise
    return {**summary, "path": str(path), "reused": False}


def _flush(db, inputs, mapping):
    db.executemany("INSERT INTO unique_inputs VALUES (?,?,?,?)", inputs)
    db.executemany("INSERT INTO row_map VALUES (?,?,?,?)", mapping)
    inputs.clear()
    mapping.clear()


def _read_set(db, input_set_id):
    if (
        input_set_id is None
        and db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='active_input_set'"
        ).fetchone()
    ):
        selected = db.execute("SELECT input_set_id FROM active_input_set WHERE id=1").fetchone()
        if selected:
            input_set_id = selected[0]
    rows = db.execute(
        "SELECT input_set_id,recipe_json,summary_json FROM input_sets"
        + (" WHERE input_set_id=?" if input_set_id else ""),
        (input_set_id,) if input_set_id else (),
    ).fetchall()
    if len(rows) != 1:
        raise ValueError("Select exactly one complete input_set_id")
    identity, recipe_json, summary_json = rows[0]
    recipe, summary = json.loads(recipe_json), json.loads(summary_json)
    if digest(recipe) != identity or summary.get("input_set_id") != identity:
        raise ValueError("Input recipe/summary checksum mismatch")
    stored = db.execute(
        "SELECT entry_json,entry_sha256 FROM question_banks WHERE bank_id=?", (recipe["bank_id"],)
    ).fetchone()
    if not stored:
        raise ValueError("Input set question version is missing")
    entry = json.loads(stored[0])
    if digest(entry) != stored[1] or stored[1] != recipe["entry_sha256"]:
        raise ValueError("Prepared question checksum mismatch")
    return identity, recipe, summary, entry


def activate_input_set(path, input_set_id):
    """Explicit preparation selects a verified version; retain all historical sets."""
    with closing(sqlite3.connect(path, timeout=120)) as db, db:
        _read_set(db, input_set_id)
        if db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='active_input_set'"
        ).fetchone() and db.execute(
            "SELECT input_set_id FROM active_input_set WHERE id=1"
        ).fetchone() == (input_set_id,):
            return
        db.execute(
            "CREATE TABLE IF NOT EXISTS active_input_set (id INTEGER PRIMARY KEY CHECK(id=1), input_set_id TEXT NOT NULL)"
        )
        db.execute("INSERT OR REPLACE INTO active_input_set VALUES (1,?)", (input_set_id,))


def iter_prepared_inputs(path, *, input_set_id=None, after_input_no=-1, input_numbers=None):
    """Yield each unique provider-neutral payload once, without a full in-memory table."""
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        identity, recipe, _, entry = _read_set(db, input_set_id)
        parameters = [identity, after_input_no]
        subset = ""
        if input_numbers is not None:
            numbers = sorted(set(input_numbers))
            if not numbers:
                return
            if len(numbers) > 1000 or any(
                isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in numbers
            ):
                raise ValueError("Pilot selection must contain at most 1000 nonnegative input IDs")
            subset = " AND input_no IN (" + ",".join("?" for _ in numbers) + ")"
            parameters.extend(numbers)
        for number, sha, text_json in db.execute(
            "SELECT input_no,payload_sha256,text_json FROM unique_inputs WHERE input_set_id=? AND input_no>?"
            + subset
            + " ORDER BY input_no",
            parameters,
        ):
            payload = payload_from_text(entry, recipe["question_variant"], json.loads(text_json))
            if digest(payload) != sha:
                raise ValueError("Prepared input payload checksum mismatch")
            yield {
                "input_set_id": identity,
                "input_no": number,
                "payload_sha256": sha,
                "payload": payload,
            }


def prepared_request(record, *, settings, provenance):
    """Bind explicit provider settings later; extraction still needs its normal gates."""
    require_nonempty_text(record["payload"])
    if digest(record["payload"]) != record["payload_sha256"]:
        raise ValueError("Prepared input payload checksum mismatch")
    return IntendedRequest(
        canonical_json(record["payload"]),
        canonical_json(settings),
        canonical_json(
            {
                **provenance,
                "input_set_id": record["input_set_id"],
                "input_no": record["input_no"],
                "payload_sha256": record["payload_sha256"],
            }
        ),
    )


def prepared_row(path, row_position=0, *, input_set_id=None):
    """Exact row preview, including every per-column/joint slot and explicit empty slots."""
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        identity, recipe, _, entry = _read_set(db, input_set_id)
        if not 0 <= row_position < recipe["row_count"]:
            raise IndexError(row_position)
        rows = db.execute(
            """SELECT m.slot,m.input_no,u.payload_sha256,u.text_json
            FROM row_map m LEFT JOIN unique_inputs u
            ON m.input_set_id=u.input_set_id AND m.input_no=u.input_no
            WHERE m.input_set_id=? AND m.row_position=? ORDER BY m.slot""",
            (identity, row_position),
        ).fetchall()
        if len(rows) != len(recipe["slots"]):
            raise ValueError("Prepared row mapping is incomplete")
        result = []
        for slot, number, sha, text_json in rows:
            payload = (
                payload_from_text(entry, recipe["question_variant"], json.loads(text_json))
                if text_json
                else None
            )
            if payload is not None and digest(payload) != sha:
                raise ValueError("Prepared input payload checksum mismatch")
            result.append(
                {
                    "row_id": f"csv:{row_position}",
                    **recipe["slots"][slot],
                    "input_set_id": identity,
                    "input_no": number,
                    "payload_sha256": sha,
                    "status": "ready_offline" if payload else "missing_no_request",
                    "payload": payload,
                }
            )
        return result


def input_origins(path, input_no, *, input_set_id=None):
    """Stream all row/representation links sharing one input, for later feature expansion."""
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        identity, recipe, _, _ = _read_set(db, input_set_id)
        for pos, slot in db.execute(
            "SELECT row_position,slot FROM row_map WHERE input_set_id=? AND input_no=? ORDER BY row_position,slot",
            (identity, input_no),
        ):
            yield {"row_id": f"csv:{pos}", **recipe["slots"][slot]}
