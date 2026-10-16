"""Timing and immutable records for supplied predictions; never launches a model."""

from __future__ import annotations

import json
import os
import platform
import tempfile
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Thread

import pandas as pd

from src.utils import paths
from src.utils.events import append_event
from src.utils.files import atomic_replace
from src.utils.serialization import digest, file_sha256, write_json


class ResourceMonitor:
    """Sample current-process RSS; CUDA peaks include all fits in this outer fold."""

    def __init__(self, device="cpu"):
        self.device = device
        self.stop = Event()
        self.peak = 0
        self.torch = None

    def __enter__(self):
        import psutil

        self.process = psutil.Process()
        if self.device.startswith("cuda"):
            import torch

            self.torch = torch
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self._sample()
        self.thread = Thread(target=self._watch, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        self.peak = max(self.peak, self.process.memory_info().rss)

    def _watch(self):
        while not self.stop.wait(0.05):
            self._sample()

    def snapshot(self):
        self._sample()
        return {
            "peak_rss_bytes": self.peak,
            "rss_scope": "sampled_current_process_including_preexisting_allocations_excluding_children",
            "rss_sampling_interval_seconds": 0.05,
            "peak_gpu_allocated_bytes": self.torch.cuda.max_memory_allocated(self.device)
            if self.torch
            else None,
            "peak_gpu_reserved_bytes": self.torch.cuda.max_memory_reserved(self.device)
            if self.torch
            else None,
        }

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()


def fold_marker(recipe, phase):
    return paths.results_dir("cv_folds", digest(recipe) + ".json", phase=phase)


def completed_fold(recipe, phase):
    """Reuse a fold only when all role records and prediction checksums still match."""
    marker = fold_marker(recipe, phase)
    if not marker.is_file():
        return None
    manifest = json.loads(marker.read_text(encoding="utf-8"))
    if manifest.get("recipe") != recipe or manifest.get("status") != "complete":
        raise ValueError("Completed CV fold identity differs")
    root = paths.results_dir(phase=phase).resolve()
    result = []
    for item in manifest["records"]:
        record = (root / item["relative_path"]).resolve()
        if not record.is_relative_to(root) or file_sha256(record) != item["sha256"]:
            raise ValueError("Completed CV record is missing or corrupt")
        body = json.loads(record.read_text(encoding="utf-8"))
        if body["metadata"].get("fold_recipe_id") != digest(recipe):
            raise ValueError("Completed CV record belongs to a different fold recipe")
        prediction = record.parent / body["predictions"]["file"]
        if (
            prediction.resolve().parent != record.parent
            or file_sha256(prediction) != body["predictions"]["sha256"]
        ):
            raise ValueError("Completed CV predictions are missing or corrupt")
        result.append(record)
    expected_roles = {"outer_test"}
    if recipe["task_type"] == "binary_classification":
        expected_roles.add("inner_oof_binary_threshold")
    roles = [
        json.loads(p.read_text(encoding="utf-8"))["metadata"]["prediction_role"] for p in result
    ]
    if len(roles) != len(expected_roles) or set(roles) != expected_roles:
        raise ValueError("Completed CV fold does not contain every prediction role")
    return result


def complete_fold(recipe, records, phase):
    root = paths.results_dir(phase=phase).resolve()
    write_json(
        fold_marker(recipe, phase),
        {
            "status": "complete",
            "recipe": recipe,
            "records": [
                {
                    "relative_path": p.resolve().relative_to(root).as_posix(),
                    "sha256": file_sha256(p),
                }
                for p in records
            ],
        },
    )


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
    folder: Path | None = None,
    *,
    scores: dict,
    predictions: pd.DataFrame,
    timer: PhaseTimer,
    metadata: dict,
    resources: dict | None = None,
    warnings: list | None = None,
    run_id: str | None = None,
    phase: str = "experiment_1",
) -> Path:
    """Save raw predictions first and a completion record last, with their checksum.

    The caller supplies the fold/split/checkpoint identities. Labels are evaluation-only;
    this module and its outputs must never be used by Jev request construction.
    """
    required = {
        "dataset_id",
        "data_identity",
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
    paths.phase_name(phase)
    if folder is None and run_id is None:
        raise ValueError("Supply a run_id for automatically routed prediction storage")
    run_id = paths.output_component(run_id) if run_id else digest(metadata)
    folder = paths.require_project_output(folder or paths.prediction_dir(run_id, phase=phase))
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
        record = write_json(
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
                "warnings": warnings or [],
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
        # Full probability arrays, confusion matrices and reliability tables stay on PROJECT.
        compact_scores = {
            "task": scores["task"],
            "n_observations": scores["n_observations"],
            "metrics": scores["metrics"],
            "undefined": scores["undefined"],
            "decisions": {
                name: {
                    key: value
                    for key, value in decision.items()
                    if key in {"metrics", "undefined", "threshold"}
                }
                for name, decision in scores.get("decisions", {}).items()
            },
        }
        if "text_overlap" in scores:
            compact_scores["text_overlap"] = {
                "version": scores["text_overlap"]["version"],
                "cohorts": {
                    name: {
                        k: v
                        for k, v in cohort.items()
                        if k
                        in {
                            "n_observations",
                            "fraction_of_outer_test",
                            "metrics",
                            "undefined",
                            "status",
                            "reason",
                        }
                    }
                    | {
                        "decisions": {
                            d: {
                                k: v
                                for k, v in values.items()
                                if k in {"metrics", "undefined", "threshold"}
                            }
                            for d, values in cohort.get("decisions", {}).items()
                        }
                    }
                    for name, cohort in scores["text_overlap"]["cohorts"].items()
                },
            }
        write_json(
            paths.manifests_dir(phase) / "evaluations" / f"{run_id}.json",
            {
                "status": "complete",
                "metadata": metadata,
                "scores": compact_scores,
                "timing": timer.to_dict(),
                "record_path": str(record),
                "predictions_path": str(destination),
                "predictions_sha256": file_sha256(destination),
            },
        )
        append_event(
            paths.logs_dir(phase) / "evaluations.jsonl",
            "evaluation_saved",
            run_id=run_id,
            rows=len(predictions),
            record_path=str(record),
        )
        return record
