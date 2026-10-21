"""Run an allocated command with bounded, compressed GPU telemetry on personal DATA."""

import argparse
import csv
import gzip
import io
import os
import signal
import subprocess
import threading
import time
from contextlib import suppress

from src.utils import paths
from src.utils.serialization import write_json


def gpu_sample():
    devices = os.environ.get("SLURM_STEP_GPUS") or os.environ.get("SLURM_JOB_GPUS")
    if not devices:
        return []  # Never attribute another user's GPU to a CPU-only job.
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            devices,
            "--query-gpu=uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=8,
        check=True,
    )
    return list(csv.reader(io.StringIO(result.stdout), skipinitialspace=True))


def run(command, *, interval=10):
    if not command or interval < 1:
        raise ValueError("A command and positive sampling interval are required")
    job = os.environ.get("SLURM_JOB_ID")
    if not job or not paths.on_vsc():
        raise RuntimeError("Telemetry belongs inside a VSC allocation")
    phase = os.environ.get("JEVPFN_PHASE", "experiment_0")
    from uuid import uuid4

    identity = f"{job}_{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}_{uuid4().hex[:8]}"
    target = paths.logs_dir(phase) / "telemetry" / f"{identity}.csv.gz"
    receipt = paths.manifests_dir(phase) / "telemetry" / f"{identity}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    samples, errors = [], []
    started = time.monotonic()

    def sample():
        with gzip.open(target, "wt", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "elapsed_seconds",
                    "gpu_uuid",
                    "name",
                    "gpu_util_percent",
                    "memory_util_percent",
                    "vram_used_mib",
                    "vram_total_mib",
                    "power_w",
                ]
            )
            while not stop.is_set():
                try:
                    for row in gpu_sample():
                        writer.writerow([round(time.monotonic() - started, 3), *row])
                        if len(row) == 7:
                            with suppress(ValueError):
                                samples.append(float(row[2]))
                    stream.flush()
                except (OSError, subprocess.SubprocessError) as exc:
                    if type(exc).__name__ not in errors:
                        errors.append(type(exc).__name__)
                stop.wait(interval)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    code, child = None, None
    previous = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        if child is not None and child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGTERM, terminate)
    try:
        print(f"GPU telemetry -> {target}", flush=True)
        child = subprocess.Popen(command)
        code = child.wait()
        return code if code >= 0 else 128 - code
    finally:
        stop.set()
        thread.join(timeout=10)
        signal.signal(signal.SIGTERM, previous)
        write_json(
            receipt,
            {
                "job_id": job,
                "cluster": os.environ.get("SLURM_CLUSTER_NAME"),
                "partition": os.environ.get("SLURM_JOB_PARTITION"),
                "cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
                "command": command,
                "samples_file": str(target),
                "exit_code": code,
                "wall_seconds": time.monotonic() - started,
                "gpu_samples": len(samples),
                "sampling_errors": errors,
                "mean_gpu_util_percent": sum(samples) / len(samples) if samples else None,
                "peak_gpu_util_percent": max(samples) if samples else None,
                "note": "Sampled device activity, not a guarantee of kernel efficiency or continuous saturation",
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    from src.utils.config import load_yaml

    interval = args.interval
    if interval is None:
        interval = load_yaml("experiment_0/scheduling")["monitor_interval_seconds"]
    raise SystemExit(run(command, interval=interval))


if __name__ == "__main__":
    main()
