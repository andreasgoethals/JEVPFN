"""Compact extraction timings, separate from immutable responses and feature identities.

One SQLite file per run goes to PROJECT; its small summary goes to DATA. Calls are
timed with a monotonic clock. Cache reuse is an untimed event, never zero-cost inference.
These client/batch durations include input preparation, not just GPU kernel execution.
"""

from __future__ import annotations

import math
import os
import sqlite3
import time
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from threading import Lock
from uuid import uuid4

from src.utils import paths
from src.utils.serialization import canonical_json, write_json


class TimingJournal:
    def __init__(self, name, *, phase="feature_creation", run_id=None, context=None):
        self.phase = paths.phase_name(phase)
        self.run_id = paths.output_component(run_id or uuid4().hex)
        self.name = paths.output_component(name)
        self.context = {
            "cluster": os.environ.get("SLURM_CLUSTER_NAME"),
            "job_id": os.environ.get("SLURM_JOB_ID"),
            **(context or {}),
        }
        self._pending, self._lock = [], Lock()
        self.path = paths.require_project_output(
            paths.large_outputs_dir() / paths.phase_folder(self.phase) / "results" / "timings"
            / f"{self.name}_{self.run_id}.sqlite3"
        )

    def record(
        self,
        stage,
        *,
        seconds=None,
        dataset_id=None,
        item_keys=(),
        items=1,
        error_type=None,
        cuda_synchronized=False,
    ):
        if seconds is not None and (not math.isfinite(seconds) or seconds < 0):
            raise ValueError("Timing seconds must be finite and nonnegative")
        if type(items) is not int or items < 0:
            raise ValueError("Timing item count must be a nonnegative integer")
        row = (
            datetime.now(UTC).isoformat(),
            stage,
            dataset_id,
            canonical_json(list(item_keys)),
            items,
            seconds,
            error_type,
            int(cuda_synchronized),
        )
        with self._lock:
            self._pending.append(row)
            # Amortize shared-filesystem commits. Summaries flush the final partial batch.
            if len(self._pending) >= 64 or error_type is not None or stage == "model_loading":
                self._flush()

    def _flush(self):
        if not self._pending:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=120)) as db, db:
            # Default DELETE journalling and FULL synchronization work on shared storage.
            db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
            db.execute(
                "INSERT OR IGNORE INTO metadata VALUES ('run', ?)",
                (
                    canonical_json(
                        {
                            "schema": "jevpfn.timing.v1",
                            "name": self.name,
                            "run_id": self.run_id,
                            "phase": self.phase,
                            "context": self.context,
                        }
                    ),
                ),
            )
            db.execute("""CREATE TABLE IF NOT EXISTS measurements (
                id INTEGER PRIMARY KEY, created_utc TEXT NOT NULL, stage TEXT NOT NULL,
                dataset_id TEXT, item_keys_json TEXT NOT NULL, items INTEGER NOT NULL,
                seconds REAL, error_type TEXT, cuda_synchronized INTEGER NOT NULL)""")
            db.executemany(
                "INSERT INTO measurements VALUES (NULL,?,?,?,?,?,?,?,?)",
                self._pending,
            )
        self._pending.clear()

    @contextmanager
    def measure(self, stage, *, synchronize=None, **context):
        if synchronize is not None:
            synchronize()
        started = time.perf_counter()
        error_type, synchronized = None, False
        try:
            yield
            if synchronize is not None:
                synchronize()
                synchronized = True
        except BaseException as exc:
            error_type = type(exc).__name__
            raise
        finally:
            self.record(
                stage,
                seconds=time.perf_counter() - started,
                error_type=error_type,
                cuda_synchronized=synchronized,
                **context,
            )

    def summary(self, *, dataset_id=None):
        with self._lock:
            self._flush()
        rows = []
        if self.path.exists():
            with closing(
                sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)
            ) as db:
                rows = db.execute(
                    """SELECT stage,dataset_id,COUNT(*),SUM(items),COUNT(seconds),
                        SUM(seconds),MIN(seconds),MAX(seconds),
                        SUM(error_type IS NOT NULL),SUM(cuda_synchronized)
                        FROM measurements WHERE (? IS NULL OR dataset_id=?)
                        GROUP BY stage,dataset_id ORDER BY stage,dataset_id""",
                    (dataset_id, dataset_id),
                ).fetchall()
        return {
            "schema": "jevpfn.timing.v1",
            "run_id": self.run_id,
            "name": self.name,
            "context": self.context,
            "database": str(self.path),
            "clock": "monotonic_wall_seconds",
            "latency_scope": "client_or_encoder_batch_including_input_preparation",
            "cache_reuse_is_not_inference": True,
            "stages": [
                dict(
                    stage=stage,
                    dataset_id=dataset,
                    observations=n,
                    items=items,
                    timed_observations=timed,
                    wall_seconds=total,
                    mean_seconds_per_observation=total / timed if timed else None,
                    min_seconds=low,
                    max_seconds=high,
                    failures=failed,
                    cuda_synchronized_observations=synced,
                )
                for stage, dataset, n, items, timed, total, low, high, failed, synced in rows
            ],
        }

    def write_summary(self):
        report = self.summary()
        write_json(
            paths.manifests_dir(self.phase) / "timings" / f"{self.name}_{self.run_id}.json",
            report,
        )
        return report
