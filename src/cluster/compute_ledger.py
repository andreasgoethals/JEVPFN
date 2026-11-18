"""Compute-time ledger from Slurm accounting: GPU-hours and CPU core-hours per stage.

Only compute time is reported (no token or API prices). Every terminal JEVPFN job is
counted once; failed, cancelled and timed-out attempts are kept separately so the paper
can report both the successful cost and the total spent. Run on a VSC login node:

    python -m src.cluster.compute_ledger --since 2026-10-01 --output PATH.json
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from collections import defaultdict
from pathlib import Path

FIELDS = ("JobIDRaw", "JobName", "State", "ElapsedRaw", "AllocTRES", "Partition", "Start",
          "End", "Cluster", "SubmitLine")
# Job name -> stage. Names come from scripts/slurm/*.slurm and the dispatcher.
STAGES = (
    ("jevpfn-features", "decision_feature_creation"),
    ("jevpfn-scratch-release", "decision_feature_creation"),
    ("jevpfn-exp0-stage-gpfs", "decision_feature_creation"),
    ("jevpfn-feature-recovery-qa", "decision_feature_creation"),
    ("jevpfn-embeddings", "frozen_embedding_creation"),
    ("jevpfn-exp0-encoder-stage", "frozen_embedding_creation"),
    ("jevpfn-prepare-pca", "fold_pca"),
    ("jevpfn-tar-pca", "tar"),
    ("jevpfn-tar", "tar"),
    ("jevpfn-benchmark", "benchmark"),
    ("jevpfn-rq1", "label_free_studies"),
    ("jevpfn-0b-replay", "label_free_studies"),
    ("jevpfn-exp0", "qualification"),
    ("jevpfn-control", "infrastructure"),
    ("jevpfn-container", "infrastructure"),
    ("jevpfn-migration", "infrastructure"),
    ("jevpfn-input-install", "infrastructure"),
    ("jevpfn-stage-tabstar", "infrastructure"),
)
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY",
            "PREEMPTED", "BOOT_FAIL", "DEADLINE"}


def stage_of(name: str) -> str | None:
    if not name.startswith("jevpfn"):
        return None
    return next((stage for prefix, stage in STAGES if name.startswith(prefix)), "other")


def _tres(value: str, key: str) -> int:
    match = re.search(rf"(?:^|,){re.escape(key)}=(\d+)", value or "")
    return int(match.group(1)) if match else 0


def _decision_model(submit_line: str) -> str | None:
    """The provider is the first argument after the feature-creation script."""
    try:
        tokens = shlex.split(submit_line or "")
    except ValueError:
        return None
    for i, token in enumerate(tokens):
        if token.endswith("feature_creation.slurm") and i + 1 < len(tokens):
            return tokens[i + 1]
    return None


def parse_sacct(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) < len(FIELDS):
            continue
        # SubmitLine is last and may itself contain "|".
        record = dict(zip(FIELDS[:-1], parts[: len(FIELDS) - 1], strict=True))
        record["SubmitLine"] = "|".join(parts[len(FIELDS) - 1:])
        stage = stage_of(record["JobName"])
        state = record["State"].split(" ")[0]
        if stage is None or state not in TERMINAL or not record["ElapsedRaw"].isdigit():
            continue
        seconds = int(record["ElapsedRaw"])
        gpus = _tres(record["AllocTRES"], "gres/gpu")
        cpus = _tres(record["AllocTRES"], "cpu")
        rows.append({
            "job_id": record["JobIDRaw"], "cluster": record["Cluster"], "job_name": record["JobName"],
            "stage": stage, "state": state, "partition": record["Partition"],
            "decision_model": _decision_model(record["SubmitLine"]) if stage == "decision_feature_creation" else None,
            "elapsed_seconds": seconds, "gpus": gpus, "cpus": cpus,
            "gpu_hours": seconds * gpus / 3600, "cpu_core_hours": seconds * cpus / 3600,
            "start": record["Start"], "end": record["End"],
        })
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """Totals per stage and decision model; successful and all-attempt compute separately."""
    totals = defaultdict(lambda: {"jobs": 0, "successful_jobs": 0, "gpu_hours": 0.0,
                                  "cpu_core_hours": 0.0, "successful_gpu_hours": 0.0,
                                  "successful_cpu_core_hours": 0.0})
    for row in rows:
        item = totals[(row["stage"], row["decision_model"] or "")]
        ok = row["state"] == "COMPLETED"
        item["jobs"] += 1
        item["successful_jobs"] += ok
        for unit in ("gpu_hours", "cpu_core_hours"):
            item[unit] += row[unit]
            item["successful_" + unit] += row[unit] if ok else 0.0
    return [{"stage": stage, "decision_model": model or None, **values}
            for (stage, model), values in sorted(totals.items())]


def collect(since: str, clusters=("mindwell", "wice")) -> list[dict]:
    rows = []
    for cluster in clusters:
        text = subprocess.check_output(
            ["sacct", f"--clusters={cluster}", "--allocations", "--noheader", "--parsable2",
             f"--starttime={since}", "--format=" + ",".join(FIELDS)],
            text=True, timeout=300,
        )
        rows += parse_sacct(text)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", required=True, help="Earliest job start, e.g. 2026-10-01")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = collect(args.since)
    from src.utils.serialization import write_json

    write_json(args.output, {"schema": "jevpfn.compute-ledger.v1", "since": args.since,
                             "units": "elapsed wall-clock x allocated GPUs or CPU cores",
                             "summary": summarize(rows), "jobs": rows})
    print(json.dumps({"jobs": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
