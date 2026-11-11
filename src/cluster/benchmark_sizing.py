"""Size bundled benchmark tasks from measured 0B full-fold profiles.

0B profiles fit the first outer fold at full size for the tallest and widest dataset
of every predictor/phase/recipe. Their peaks are therefore conservative bounds for
the other datasets. Fold time is extrapolated linearly in training rows from the
slowest measured rate. Without a measured profile nothing is invented: callers fall
back to fixed, documented defaults and one worker per GPU.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from src.utils import paths

# Mindwell gpu_b200 ceilings per allocated GPU (vsc-library CLUSTERS.md).
B200_CORES_PER_GPU = 24
B200_HOST_MIB_PER_GPU = 194_400
B200_VRAM_BYTES = 192 * 1024**3
CPU_NODE_CORES = 192
CPU_DEFAULT_MIB_PER_CORE = 8_000
MAX_CPU_WORKERS = 8
TIME_BUCKETS_HOURS = (2, 6, 12, 24, 48, 72)
VRAM_HEADROOM = 0.85  # CUDA contexts, fragmentation and allocator caching.


def profile_records(model, *, phase="experiment_0"):
    """One row per profiled outer fold: training rows, wall seconds and memory peaks."""
    from src.evaluation.store import readonly

    folds = {}
    root = paths.results_dir("evaluation", phase=phase)
    for path in sorted(Path(root).glob("*.sqlite3")) if Path(root).is_dir() else []:
        with readonly(path) as db:
            for (body,) in db.execute("SELECT body FROM records"):
                record = json.loads(body)
                meta = record.get("metadata", {})
                if meta.get("model_id") != model or not meta.get("fold_recipe_id"):
                    continue
                key = meta["fold_recipe_id"]
                resources = record.get("resources", {})
                row = folds.setdefault(key, {
                    "dataset_id": meta.get("dataset_id"), "n_train": meta.get("n_train") or 0,
                    "wall_seconds": 0.0, "peak_rss_bytes": 0, "peak_gpu_reserved_bytes": 0,
                    "device_workers": 1})
                row["wall_seconds"] = max(row["wall_seconds"],
                                          record.get("timing", {}).get("measured_wall_seconds") or 0.0)
                for name in ("peak_rss_bytes", "peak_gpu_reserved_bytes"):
                    row[name] = max(row[name], resources.get(name) or 0)
                row["device_workers"] = max(row["device_workers"], resources.get("device_workers") or 1)
    return list(folds.values())


def summarize(profiles):
    """Conservative per-model bounds; only exclusive-device profiles give time rates."""
    exclusive = [p for p in profiles if p["device_workers"] == 1 and p["n_train"] > 0]
    if not profiles:
        return None
    return {
        "profiles": len(profiles),
        "seconds_per_training_row": max((p["wall_seconds"] / p["n_train"] for p in exclusive),
                                        default=None),
        "max_fold_seconds": max(p["wall_seconds"] for p in profiles),
        "peak_rss_bytes": max(p["peak_rss_bytes"] for p in profiles),
        "peak_gpu_reserved_bytes": max(p["peak_gpu_reserved_bytes"] for p in profiles),
    }


def workers(summary, *, gpu, threads, arms, requested=None):
    """Concurrent workers per task, each keeping the reviewed thread count."""
    if requested is not None:
        if type(requested) is not int or requested < 1:
            raise ValueError("Workers must be a positive integer")
        limit = requested
    elif gpu:
        if not summary or not summary["peak_gpu_reserved_bytes"]:
            limit = 1  # Never share an unmeasured GPU workload.
        else:
            limit = int(VRAM_HEADROOM * B200_VRAM_BYTES // summary["peak_gpu_reserved_bytes"])
    else:
        limit = MAX_CPU_WORKERS
    if gpu:
        limit = min(limit, B200_CORES_PER_GPU // threads)
        if summary and summary["peak_rss_bytes"]:
            limit = min(limit, int(B200_HOST_MIB_PER_GPU * 2**20 * 0.9 // summary["peak_rss_bytes"]))
    else:
        limit = min(limit, CPU_NODE_CORES // threads)
    return max(1, min(limit, arms))


def estimated_arm_seconds(summary, rows):
    """Five outer folds at the slowest measured per-row rate (inner fits included)."""
    if not summary or summary["seconds_per_training_row"] is None:
        return None
    return 5 * max(60.0, summary["seconds_per_training_row"] * 0.8 * rows)


def resources(summary, *, gpu, threads, workers_per_task, arm_rows):
    """Slurm request for one task; time is bucketed so tasks can share an array."""
    estimates = [estimated_arm_seconds(summary, rows) for rows in arm_rows]
    if any(e is None for e in estimates):
        hours = 48  # Unprofiled default; resume makes a timeout recoverable.
    else:
        seconds = 1.5 * sum(estimates) / workers_per_task + 1800
        hours = next((b for b in TIME_BUCKETS_HOURS if seconds <= b * 3600), TIME_BUCKETS_HOURS[-1])
    cpus = threads * workers_per_task
    mem_mib = None
    if summary and summary["peak_rss_bytes"]:
        mem_mib = int(math.ceil(1.25 * workers_per_task * summary["peak_rss_bytes"] / 2**20)) + 4096
        mem_mib = int(math.ceil(mem_mib / 1024) * 1024)
        ceiling = B200_HOST_MIB_PER_GPU if gpu else CPU_DEFAULT_MIB_PER_CORE * CPU_NODE_CORES
        mem_mib = min(mem_mib, ceiling)
    return {"cpus": cpus, "time_hours": hours, "mem_mib": mem_mib}
