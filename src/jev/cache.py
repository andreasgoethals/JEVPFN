"""Transactional local response store; no network transport.

One SQLite file avoids millions of small files. Semantic keys deduplicate across row IDs;
the separate origins table retains every request provenance. Mock entries cannot be read
as real responses. Guarded in-flight reservations are prepared in src.jev.extraction;
this storage module itself never sends requests.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from src.jev.requests import IntendedRequest
from src.utils.serialization import canonical_json, digest


class FutureJevClient(Protocol):
    """Interface for guarded transports; actual live contracts still require a pilot."""

    def evaluate(self, request: IntendedRequest) -> dict: ...


class DisabledJevClient:
    def evaluate(self, request: IntendedRequest) -> dict:
        raise NotImplementedError("Real Jev calls are disabled in this workflow.")


class ResponseCache:
    def __init__(self, path: Path, dataset_id: str | None = None):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.dataset_id = dataset_id
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS dataset_identity (dataset_id TEXT PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS responses (
                    request_key TEXT NOT NULL, kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL, settings_json TEXT NOT NULL,
                    response_json TEXT NOT NULL, created_at TEXT NOT NULL,
                    PRIMARY KEY (request_key, kind));
                CREATE TABLE IF NOT EXISTS origins (
                    request_key TEXT NOT NULL, kind TEXT NOT NULL,
                    provenance_hash TEXT NOT NULL, provenance_json TEXT NOT NULL,
                    first_seen_at TEXT NOT NULL,
                    PRIMARY KEY (request_key, kind, provenance_hash));
            """)
            bound = db.execute("SELECT dataset_id FROM dataset_identity").fetchall()
            if dataset_id is not None:
                if bound and bound != [(dataset_id,)]:
                    raise ValueError("Cache belongs to a different dataset.")
                db.execute("INSERT OR IGNORE INTO dataset_identity VALUES (?)", (dataset_id,))
            elif bound:
                self.dataset_id = bound[0][0]

    @classmethod
    def for_dataset(cls, dataset_id: str):
        from src.utils.paths import dataset_cache_path

        return cls(dataset_cache_path(dataset_id), dataset_id=dataset_id)

    def _validate_dataset(self, request):
        if (
            self.dataset_id is not None
            and json.loads(request.provenance_json)["dataset_id"] != self.dataset_id
        ):
            raise ValueError("Request belongs to a different dataset.")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30, uri=True)
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _validate_kind(request: IntendedRequest, kind: str) -> None:
        if kind not in {"mock", "verified_response"}:
            raise ValueError("Unknown cache namespace.")
        settings = json.loads(request.settings_json)
        if kind == "verified_response" and (not settings["model"] or not settings["version"]):
            raise ValueError("Real response storage requires explicit model and version pins.")

    def _trace(self, db, request: IntendedRequest, kind: str) -> None:
        db.execute(
            "INSERT OR IGNORE INTO origins VALUES (?,?,?,?,?)",
            (
                request.cache_key,
                kind,
                digest(json.loads(request.provenance_json)),
                request.provenance_json,
                datetime.now(UTC).isoformat(),
            ),
        )

    def get(self, request: IntendedRequest, *, kind: str = "mock") -> dict | None:
        self._validate_dataset(request)
        self._validate_kind(request, kind)
        with self._connect() as db:
            row = db.execute(
                "SELECT response_json FROM responses WHERE request_key=? AND kind=?",
                (request.cache_key, kind),
            ).fetchone()
            if row is None:
                return None
            self._trace(db, request, kind)
            return json.loads(row[0])

    def put(self, request: IntendedRequest, response: dict, *, kind: str = "mock") -> None:
        self._validate_dataset(request)
        self._validate_kind(request, kind)
        encoded = canonical_json(response)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT response_json FROM responses WHERE request_key=? AND kind=?",
                (request.cache_key, kind),
            ).fetchone()
            if existing and existing[0] != encoded:
                raise ValueError("Refusing to replace an existing identical request's response.")
            db.execute(
                "INSERT OR IGNORE INTO responses VALUES (?,?,?,?,?,?)",
                (
                    request.cache_key,
                    kind,
                    request.payload_json,
                    request.settings_json,
                    encoded,
                    datetime.now(UTC).isoformat(),
                ),
            )
            self._trace(db, request, kind)

    def get_many(self, requests, *, kind="verified_response") -> dict:
        """One connection per bounded batch; record every row origin, including duplicates."""
        requests = list(requests)
        for request in requests:
            self._validate_dataset(request)
            self._validate_kind(request, kind)
        keys = list(dict.fromkeys(request.cache_key for request in requests))
        found = {}
        with self._connect() as db:
            for start in range(0, len(keys), 500):
                batch = keys[start : start + 500]
                marks = ",".join("?" for _ in batch)
                found.update(
                    (key, json.loads(value))
                    for key, value in db.execute(
                        f"SELECT request_key,response_json FROM responses WHERE kind=? AND request_key IN ({marks})",
                        [kind, *batch],
                    )
                )
            for request in requests:
                if request.cache_key in found:
                    self._trace(db, request, kind)
        return found


def dry_run(request: IntendedRequest, cache: ResponseCache) -> dict:
    cached = cache.get(request)
    if cached is not None:
        return {"cache_hit": True, "response": cached}
    response = {"status": "mock_only", "sent": False, "probabilities": None}
    cache.put(request, response)
    return {"cache_hit": False, "response": response}
