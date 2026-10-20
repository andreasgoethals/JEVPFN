"""Compact immutable prediction records and resumable folds, one SQLite file per dataset.

Parquet payloads stay on PROJECT. Transactions are short; fitting never holds a
database transaction. DELETE journalling is deliberate for shared filesystems.
"""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils import paths
from src.utils.serialization import canonical_json, digest, write_json

SCHEMA = "jevpfn.evaluation.v1"


def database(dataset_id, phase):
    return paths.results_dir(
        "evaluation", paths.output_component(dataset_id) + ".sqlite3", phase=phase
    )


@contextmanager
def connect(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=120)
    db.execute("PRAGMA journal_mode=DELETE")
    db.execute("PRAGMA synchronous=FULL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS records (
            id TEXT PRIMARY KEY, body TEXT NOT NULL, predictions BLOB NOT NULL);
        CREATE TABLE IF NOT EXISTS folds (
            id TEXT PRIMARY KEY, recipe TEXT NOT NULL, records TEXT NOT NULL);
    """)
    db.execute("INSERT OR IGNORE INTO metadata VALUES ('schema', ?)", (SCHEMA,))
    db.commit()
    if db.execute("SELECT value FROM metadata WHERE key='schema'").fetchone() != (SCHEMA,):
        db.close()
        raise ValueError("Unknown evaluation database schema")
    try:
        with db:
            yield db
    finally:
        db.close()


@contextmanager
def readonly(path):
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        yield db


@dataclass(frozen=True)
class RecordRef:
    database: Path
    key: str

    def __str__(self):
        return f"{self.database}#{self.key}"

    @property
    def name(self):
        return self.key

    def is_relative_to(self, root):
        return self.database.is_relative_to(root)

    def read_text(self, encoding="utf-8"):
        with readonly(self.database) as db:
            row = db.execute("SELECT body FROM records WHERE id=?", (self.key,)).fetchone()
        if row is None:
            raise ValueError("Missing evaluation record")
        return row[0]


def read_predictions(record):
    if isinstance(record, RecordRef):
        body = json.loads(record.read_text())
        with readonly(record.database) as db:
            payload = db.execute(
                "SELECT predictions FROM records WHERE id=?", (record.key,)
            ).fetchone()[0]
        if hashlib.sha256(payload).hexdigest() != body["predictions"]["sha256"]:
            raise ValueError("Corrupt prediction payload")
        return pd.read_parquet(io.BytesIO(payload))
    body = json.loads(record.read_text(encoding="utf-8"))
    from src.utils.serialization import file_sha256

    path = record.parent / body["predictions"]["file"]
    if (
        path.resolve().parent != record.parent.resolve()
        or file_sha256(path) != body["predictions"]["sha256"]
    ):
        raise ValueError("Corrupt prediction payload")
    return pd.read_parquet(path)


def save(record, payload, run_id, phase):
    path = paths.require_project_output(database(record["metadata"]["dataset_id"], phase))
    record = {**record, "schema_version": 2}
    with connect(path) as db:
        try:
            db.execute(
                "INSERT INTO records VALUES (?, ?, ?)", (run_id, canonical_json(record), payload)
            )
        except sqlite3.IntegrityError as exc:
            raise FileExistsError("Immutable evaluation record already exists") from exc
    # A stable small locator, not thousands of per-fold metadata files on DATA.
    write_json(
        paths.manifests_dir(phase) / "evaluations" / f"{path.stem}.json",
        {
            "schema": SCHEMA,
            "dataset_id": path.stem,
            "project_relative_path": f"results/evaluation/{path.name}",
            "description": "Predictions, metrics, timings and completion records are in SQLite.",
        },
    )
    return RecordRef(path, run_id)


def complete(recipe, records, phase):
    path = database(recipe["dataset_id"], phase)
    if any(not isinstance(r, RecordRef) or r.database != path for r in records):
        raise ValueError("A fold must use records in its own dataset store")
    refs = [{"id": r.key, "sha256": digest(json.loads(r.read_text()))} for r in records]
    with connect(path) as db:
        row = (digest(recipe), canonical_json(recipe), canonical_json(refs))
        old = db.execute("SELECT id, recipe, records FROM folds WHERE id=?", (row[0],)).fetchone()
        if old is not None and old != row:
            raise ValueError("Conflicting completed fold")
        db.execute("INSERT OR IGNORE INTO folds VALUES (?, ?, ?)", row)


def completed(recipe, phase):
    path = database(recipe["dataset_id"], phase)
    if not path.exists():
        return None
    with readonly(path) as db:
        row = db.execute(
            "SELECT recipe, records FROM folds WHERE id=?", (digest(recipe),)
        ).fetchone()
    if row is None:
        return None
    if json.loads(row[0]) != recipe:
        raise ValueError("Fold recipe identity differs")
    result, roles = [], []
    for item in json.loads(row[1]):
        ref = RecordRef(path, item["id"])
        body = json.loads(ref.read_text())
        if digest(body) != item["sha256"] or body["metadata"]["fold_recipe_id"] != digest(recipe):
            raise ValueError("Corrupt fold record")
        # Verify compressed bytes without materializing a whole prediction table.
        with readonly(path) as db:
            payload = db.execute(
                "SELECT predictions FROM records WHERE id=?", (ref.key,)
            ).fetchone()[0]
        if hashlib.sha256(payload).hexdigest() != body["predictions"]["sha256"]:
            raise ValueError("Corrupt prediction payload")
        roles.append(body["metadata"]["prediction_role"])
        result.append(ref)
    expected = {"outer_test"}
    if recipe["task_type"] == "binary_classification":
        expected.add("inner_oof_binary_threshold")
    if len(roles) != len(expected) or set(roles) != expected:
        raise ValueError("Fold prediction roles differ")
    return result


def recipes(phase):
    for path in sorted(paths.results_dir("evaluation", phase=phase).glob("*.sqlite3")):
        with readonly(path) as db:
            for (recipe,) in db.execute("SELECT recipe FROM folds ORDER BY id"):
                yield json.loads(recipe)


def merge(source, destination):
    """Atomically union immutable records; refuse conflicts without changing destination."""
    with readonly(source) as incoming:
        if incoming.execute("SELECT value FROM metadata WHERE key='schema'").fetchone() != (
            SCHEMA,
        ):
            raise ValueError("Unknown incoming evaluation store")
        with connect(destination) as local:
            local.execute("BEGIN IMMEDIATE")
            for row in incoming.execute("SELECT id, body, predictions FROM records"):
                body = json.loads(row[1])
                if body["metadata"]["dataset_id"] != Path(destination).stem:
                    raise ValueError("Incoming evaluation dataset differs")
                if hashlib.sha256(row[2]).hexdigest() != body["predictions"]["sha256"]:
                    raise ValueError("Corrupt incoming predictions")
                prior = local.execute(
                    "SELECT id, body, predictions FROM records WHERE id=?", (row[0],)
                ).fetchone()
                if prior is not None and prior != row:
                    raise ValueError("Conflicting immutable evaluation record")
                local.execute("INSERT OR IGNORE INTO records VALUES (?, ?, ?)", row)
            for row in incoming.execute("SELECT id, recipe, records FROM folds"):
                recipe = json.loads(row[1])
                if digest(recipe) != row[0] or recipe["dataset_id"] != Path(destination).stem:
                    raise ValueError("Invalid incoming fold identity")
                for ref in json.loads(row[2]):
                    record = local.execute(
                        "SELECT body FROM records WHERE id=?", (ref["id"],)
                    ).fetchone()
                    if record is None or digest(json.loads(record[0])) != ref["sha256"]:
                        raise ValueError("Missing or corrupt incoming fold record")
                prior = local.execute(
                    "SELECT id, recipe, records FROM folds WHERE id=?", (row[0],)
                ).fetchone()
                if prior is not None and prior != row:
                    raise ValueError("Conflicting completed evaluation fold")
                local.execute("INSERT OR IGNORE INTO folds VALUES (?, ?, ?)", row)
    return "merged"
