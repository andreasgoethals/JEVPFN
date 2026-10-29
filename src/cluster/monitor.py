"""Run an allocated command with bounded, compressed GPU telemetry on personal DATA."""

import argparse
import csv
import gzip
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import suppress

from src.cluster.control_runtime import child_command, child_environment
from src.utils import paths
from src.utils.serialization import write_json


def gpu_sample():
    devices = os.environ.get("SLURM_STEP_GPUS") or os.environ.get("SLURM_JOB_GPUS")
    if not devices:
        return []  # Never attribute another user's GPU to a CPU-only job.
    executable = os.environ.get("JEVPFN_NVIDIA_SMI") or "nvidia-smi"
    query = [
        executable, "-i", devices,
        "--query-gpu=uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    arguments = dict(capture_output=True, text=True, timeout=8, check=True)
    try:
        result = subprocess.run(query, **arguments)
    except subprocess.CalledProcessError as exc:
        # wICE's cgroup may expose allocation-global GPU 3 as its sole local GPU 0.
        # Never fall back to querying every GPU: resolve a UUID only if precisely
        # one GPU was allocated AND nvidia-smi can see precisely that one device.
        if (
            exc.returncode != 6
            or "No devices were found" not in str(exc.stdout or "")
            or not devices.isdecimal()
            or os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
        ):
            raise
        inventory = subprocess.run(
            [executable, "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            **arguments,
        )
        visible = list(csv.reader(io.StringIO(inventory.stdout), skipinitialspace=True))
        if len(visible) != 1 or len(visible[0]) != 2 or visible[0][0].strip() != "0":
            raise exc
        uuid = visible[0][1].strip()
        if not uuid.startswith("GPU-") or any(c.isspace() or c == "," for c in uuid):
            raise exc
        query[2] = uuid
        result = subprocess.run(query, **arguments)
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
    samples, errors, error_details = [], [], []
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
                    detail = {"type": type(exc).__name__}
                    if isinstance(exc, subprocess.CalledProcessError):
                        detail.update(
                            returncode=exc.returncode, command=exc.cmd,
                            stderr=str(exc.stderr or "")[:2000],
                            stdout=str(exc.stdout or "")[:2000],
                        )
                    else:
                        detail["message"] = str(exc)[:2000]
                    if detail not in error_details and len(error_details) < 8:
                        error_details.append(detail)
                        print("GPU telemetry warning: " + json.dumps(detail), flush=True)
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
        child = subprocess.Popen(child_command(command), env=child_environment())
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
                "sampling_error_details": error_details,
                "monitor_python": sys.executable,
                "nvidia_smi": os.environ.get("JEVPFN_NVIDIA_SMI") or shutil.which("nvidia-smi"),
                "gpu_allocation": {
                    k: os.environ.get(k) for k in (
                        "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "CUDA_VISIBLE_DEVICES"
                    )
                },
                "device_resolution": "allocation_ids_or_verified_single_visible_gpu_uuid",
                "sampling_status": "sampled" if samples else "failed" if errors else "no_samples",
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
