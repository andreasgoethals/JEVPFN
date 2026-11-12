"""Size bundled benchmark tasks from measured 0B full-fold profiles.

0B profiles measure full-size folds at selected dataset extremes. Fold time is
estimated from the slowest measured rate. Without qualified evidence, callers fall
back to fixed defaults and one worker per GPU.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

# Mindwell gpu_b200 ceilings per allocated GPU (vsc-library CLUSTERS.md).
B200_CORES_PER_GPU = 24
B200_HOST_MIB_PER_GPU = 194_400
B200_VRAM_BYTES = 192 * 1024**3
CPU_NODE_CORES = 192
MINDWELL_CPU_MIB_PER_CORE = 4_000
MINDWELL_GPU_MIB_PER_CORE = 8_000
MAX_CPU_WORKERS = 8
TIME_BUCKETS_HOURS = (2, 6, 12, 24, 48, 72)
VRAM_HEADROOM = 0.85  # CUDA contexts, fragmentation and allocator caching.


def profile_records(model, *, phase, cluster, acceptance):
    """Use only current, reviewed full-fold receipts for the requested workload."""
    from src.evaluation.acceptance import evaluation_identity
    from src.evaluation.store import RecordRef, database
    from src.utils.acceptance import reviewed_acceptance

    if not acceptance:
        return []
    current = evaluation_identity()
    approval = reviewed_acceptance(
        acceptance, stage="after_extraction", expected={"evaluation_identity": current}
    )
    folds = {}
    for item in approval["evidence"]:
        report = json.loads((Path(acceptance).parent / item["path"]).read_text(encoding="utf-8"))
        arm = report.get("arm", {})
        if (
            report.get("status") != "full_fold_profile_passed"
            or report.get("evaluation_identity") != current
            or report.get("cluster") != cluster
            or not report.get("job_id")
            or arm.get("model") != model
            or arm.get("phase") != phase
        ):
            continue
        outer = next(item for item in report["roles"] if item["role"] == "outer_test")
        record = json.loads(RecordRef(
            database(arm["dataset_id"], "experiment_0"), outer["record_id"]
        ).read_text())
        meta = record["metadata"]
        if (
            meta.get("prediction_role") != "outer_test"
            or meta.get("model_id") != model
            or meta.get("dataset_id") != arm["dataset_id"]
            or meta.get("split_id") != report["split_id"]
            or meta.get("outer_fold") != 0
            or meta.get("n_train") != report["training_rows"]
            or meta.get("representation_feature_count") != report["predictor_columns"]
            or record["predictions"]["sha256"] != outer["prediction_sha256"]
        ):
            raise ValueError("Benchmark sizing record differs from its reviewed 0B receipt")
        wall_seconds = record.get("timing", {}).get("measured_wall_seconds") or 0.0
        if not math.isfinite(wall_seconds) or wall_seconds <= 0:
            continue
        measured = record.get("resources", {})
        folds[outer["record_id"]] = {
            "n_train": meta["n_train"],
            "wall_seconds": wall_seconds,
            "peak_rss_bytes": measured.get("peak_rss_bytes") or 0,
            "peak_gpu_reserved_bytes": measured.get("peak_gpu_reserved_bytes") or 0,
            "device_workers": measured.get("device_workers") or 1,
        }
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
    if requested is not None and (type(requested) is not int or requested < 1):
        raise ValueError("Workers must be a positive integer")
    if gpu:
        if not summary or not summary["peak_gpu_reserved_bytes"]:
            limit = 1  # Never share an unmeasured GPU workload.
        else:
            limit = int(VRAM_HEADROOM * B200_VRAM_BYTES // summary["peak_gpu_reserved_bytes"])
    else:
        limit = MAX_CPU_WORKERS
    if gpu:
        limit = min(limit, B200_CORES_PER_GPU // threads)
    else:
        limit = min(limit, CPU_NODE_CORES // threads)
    limit = min(limit, arms, requested if requested is not None else arms)
    while limit > 1 and _memory_mib(summary, limit) > _memory_ceiling(gpu):
        limit -= 1
    return max(1, limit)


def _memory_mib(summary, count):
    if not summary or not summary["peak_rss_bytes"]:
        return 0
    return 1024 * math.ceil((1.25 * count * summary["peak_rss_bytes"] / 2**20 + 4096) / 1024)


def _memory_ceiling(gpu):
    if gpu:
        return min(B200_HOST_MIB_PER_GPU, B200_CORES_PER_GPU * MINDWELL_GPU_MIB_PER_CORE)
    return MINDWELL_CPU_MIB_PER_CORE * CPU_NODE_CORES


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
        longest_worker = max(sum(estimates[i::workers_per_task]) for i in range(workers_per_task))
        seconds = 1.5 * longest_worker + 1800
        hours = next((b for b in TIME_BUCKETS_HOURS if seconds <= b * 3600), TIME_BUCKETS_HOURS[-1])
    cpus = threads * workers_per_task
    mem_mib = min(_memory_mib(summary, workers_per_task), _memory_ceiling(gpu)) or None
    return {"cpus": cpus, "time_hours": hours, "mem_mib": mem_mib}
