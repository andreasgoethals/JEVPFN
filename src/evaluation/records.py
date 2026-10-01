"""Timing and immutable records for supplied predictions; never launches a model."""

from __future__ import annotations

import os
import platform
import tempfile
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.utils.files import atomic_replace
from src.utils.serialization import file_sha256, write_json


class PhaseTimer:
    """Accumulate nonoverlapping wall/CPU timings, including failed attempts.

    For CUDA, pass the adapter's device synchronization callback. Model loading,
    preprocessing, inner fitting, final fitting and prediction must be distinct stages.
    CPU time covers this process (all its threads), not subprocesses. GPU peak memory
    must be supplied separately by a device-aware adapter; it is never invented here.
    """

    def __init__(self, *, device="cpu", synchronize=None):
        if device.startswith("cuda") and synchronize is None:
            raise ValueError("CUDA timings require a synchronization callback")
        self.device = device
        self.synchronize = synchronize
        self.stages = {}
        self._active = False

    @contextmanager
    def measure(self, stage):
        if self._active:
            raise ValueError("Nested timing stages would double-count time")
        if not isinstance(stage, str) or not stage:
            raise ValueError("A timing stage name is required")
        if self.synchronize:
            self.synchronize()
        self._active = True
        start, cpu_start = time.perf_counter(), time.process_time()
        success = False
        try:
            yield
            if self.synchronize:
                self.synchronize()
            success = True
        finally:
            end, cpu_end = time.perf_counter(), time.process_time()
            self._active = False
            item = self.stages.setdefault(
                stage, {"wall_seconds": 0.0, "cpu_seconds": 0.0, "attempts": 0, "failures": 0}
            )
            item["wall_seconds"] += end - start
            item["cpu_seconds"] += cpu_end - cpu_start
            item["attempts"] += 1
            item["failures"] += int(not success)

    def to_dict(self):
        if self._active:
            raise ValueError("Cannot finalize timings while a stage is active")
        return {
            "device": self.device,
            "cuda_synchronized": bool(self.synchronize),
            "stages": {k: dict(v) for k, v in self.stages.items()},
            "measured_wall_seconds": sum(v["wall_seconds"] for v in self.stages.values()),
            "cpu_scope": "current_process_all_threads_excluding_child_processes",
        }


def save_evaluation(
    folder: Path,
    *,
    scores: dict,
    predictions: pd.DataFrame,
    timer: PhaseTimer,
    metadata: dict,
    resources: dict | None = None,
) -> Path:
    """Save raw predictions first and a completion record last, with their checksum.

    The caller supplies the fold/split/checkpoint identities. Labels are evaluation-only;
    this module and its outputs must never be used by Jev request construction.
    """
    required = {
        "dataset_id",
        "model_id",
        "feature_set",
        "outer_fold",
        "split_id",
        "n_train",
        "n_validation",
        "checkpoint_id",
        "target_scale",
    }
    if required - metadata.keys():
        raise ValueError(f"Missing evaluation metadata: {sorted(required - metadata.keys())}")
    if not {"row_id", "y_true"} <= set(predictions.columns):
        raise ValueError("Predictions must include row_id and y_true")
    if predictions.row_id.isna().any() or predictions.row_id.duplicated().any():
        raise ValueError("Prediction row IDs must be nonmissing and unique")
    if len(predictions) != scores["n_observations"]:
        raise ValueError("Prediction row count differs from the metric report")
    timer.to_dict()  # Reject an active/nested stage before writing anything.
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    record_path = folder / "record.json"
    from src.utils.locking import file_lock

    with file_lock(folder / ".record.lock"):
        if record_path.exists():
            raise FileExistsError(
                "Completed evaluation records are immutable; use a new run folder"
            )
        with tempfile.NamedTemporaryFile(dir=folder, suffix=".parquet.pending", delete=False) as f:
            temporary = Path(f.name)
        destination = folder / "predictions.parquet"
        with timer.measure("prediction_serialization"):
            predictions.to_parquet(temporary, index=False)
            atomic_replace(temporary, destination)
        return write_json(
            record_path,
            {
                "schema_version": 1,
                "created_utc": datetime.now(UTC).isoformat(),
                "metadata": metadata,
                "scores": scores,
                "timing": timer.to_dict(),
                "resources": {
                    "peak_rss_bytes": None,
                    "peak_gpu_allocated_bytes": None,
                    "peak_gpu_reserved_bytes": None,
                    **(resources or {}),
                },
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "cpu_count": os.cpu_count(),
                },
                "predictions": {
                    "file": destination.name,
                    "sha256": file_sha256(destination),
                    "columns": list(predictions.columns),
                },
            },
        )
