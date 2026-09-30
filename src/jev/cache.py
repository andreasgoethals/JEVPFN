"""Transactional local response store; no network transport.

One SQLite file avoids millions of small files. Semantic keys deduplicate across row IDs;
the separate origins table retains every request provenance. Mock entries cannot be read
as real responses. Concurrent *paid* in-flight reservation remains a future API concern.
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
    """TODO implement approved transport, auth, reservation/retries and live validation."""

    def evaluate(self, request: IntendedRequest) -> dict: ...


class DisabledJevClient:
    def evaluate(self, request: IntendedRequest) -> dict:
        raise NotImplementedError("Real Jev calls are intentionally not implemented.")


class ResponseCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._connect() as db:
            db.executescript("""
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

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
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
        self._validate_kind(request, kind)
        encoded = canonical_json(response)
        with self._connect() as db:
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


def dry_run(request: IntendedRequest, cache: ResponseCache) -> dict:
    cached = cache.get(request)
    if cached is not None:
        return {"cache_hit": True, "response": cached}
    response = {"status": "mock_only", "sent": False, "probabilities": None}
    cache.put(request, response)
    return {"cache_hit": False, "response": response}
