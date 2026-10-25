"""Opt-in two-node PROJECT lock and SQLite snapshot/restore check; no research data.

The local test suite cannot establish cross-node flock behavior on Lustre or GPFS.
This probe therefore runs in its own tiny two-node CPU allocation before extraction.
"""

import argparse
import json
import os
import re
import socket
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from src.cluster.preflight import storage_policy_failures
from src.utils import paths
from src.utils.locking import file_lock
from src.utils.serialization import write_json


def node_name():
    return os.environ.get("SLURMD_NODENAME", socket.gethostname()).split(".")[0]


def storage_backend(folder):
    """Identify the native root already checked by the cluster storage policy."""
    match = re.match(r"^/(gpfs|lustre)\d+(?:/|$)", str(folder).replace("\\", "/"))
    if not match:
        raise ValueError(f"Storage probe needs a recognized native GPFS or Lustre path: {folder}")
    return match.group(1)


def worker(folder, *, expect_busy):
    folder = paths.require_project_output(Path(folder)).resolve()
    if not (folder / "probe.sqlite3").is_file():
        raise FileNotFoundError("The parent node's probe database is not visible")
    with closing(sqlite3.connect(folder / "probe.sqlite3")) as db:
        if db.execute("SELECT value FROM probe WHERE id=1").fetchone() != ("original",):
            raise AssertionError("Other node cannot read the committed PROJECT database")
    try:
        with file_lock(folder / "writer.lock", timeout=1):
            if expect_busy:
                raise AssertionError("Both nodes acquired the same exclusive lock")
            with closing(sqlite3.connect(folder / "probe.sqlite3")) as db, db:
                db.execute("INSERT INTO probe VALUES (2,'second_node')")
    except TimeoutError:
        if not expect_busy:
            raise
        return {"status": "blocked_as_expected", "node": node_name()}
    return {"status": "acquired_after_release", "node": node_name()}


def worker_receipt(run_id, *, expect_busy):
    return (
        paths.manifests_dir("experiment_0")
        / "before_extraction"
        / "storage_contracts"
        / paths.output_component(run_id)
        / ("blocked.json" if expect_busy else "released.json")
    )


def read_worker_receipt(receipt, *, timeout=60):
    """Allow bounded cross-node metadata propagation after a successful worker exit.

    Only absence is transient. Atomic publication means malformed JSON, permission
    errors and other I/O failures must propagate immediately, never become a pass.
    Do not relaunch the worker: its database write may already have completed.
    """
    started = time.monotonic()
    while True:
        try:
            result = json.loads(receipt.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError(
                    f"Worker exited successfully but its receipt was not visible within "
                    f"{timeout}s: {receipt}"
                ) from exc
            time.sleep(min(0.5, remaining))
        else:
            return result, time.monotonic() - started


def remote_worker(command, *, run_id, expect_busy):
    """Exchange structured evidence through DATA, not the launcher's stdout stream.

    Preserve stdout/stderr separately, including on timeout or nonzero exit. A unique
    challenge ties each receipt to this invocation; stale/foreign results cannot pass.
    """
    receipt = worker_receipt(run_id, expect_busy=expect_busy)
    if receipt.exists():
        raise FileExistsError(f"Refusing an existing worker receipt: {receipt}")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    log_dir = paths.logs_dir("experiment_0") / "storage_contracts" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    probe_id = uuid4().hex
    argv = [*command, "--probe-id", probe_id]
    if expect_busy:
        argv.append("--expect-busy")
    with (
        (log_dir / f"{receipt.stem}.stdout.log").open("w", encoding="utf-8") as out,
        (log_dir / f"{receipt.stem}.stderr.log").open("w", encoding="utf-8") as err,
    ):
        subprocess.run(argv, stdout=out, stderr=err, check=True, timeout=60)
    try:
        result, visibility_wait = read_worker_receipt(receipt)
    except TimeoutError as exc:
        raise TimeoutError(f"{exc}; inspect stdout/stderr in {log_dir}") from exc
    expected = {
        "probe_id": probe_id,
        "job_id": os.environ["SLURM_JOB_ID"],
        "expect_busy": expect_busy,
    }
    if not isinstance(result, dict) or any(result.get(k) != v for k, v in expected.items()):
        raise ValueError("Worker receipt does not match this job/probe/lock phase")
    result["receipt_visibility_wait_seconds"] = visibility_wait
    return result


def run(*, execute=False):
    if not execute or not paths.on_vsc() or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Storage contracts require --execute in a two-node CPU allocation")
    if failures := storage_policy_failures(os.environ.get("SLURM_CLUSTER_NAME")):
        raise ValueError("; ".join(failures))
    nodes = subprocess.check_output(
        ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]], text=True
    ).split()
    local = node_name()
    if local not in [n.split(".")[0] for n in nodes]:
        raise ValueError("The executing node is not in the supplied Slurm allocation")
    others = [n for n in nodes if n.split(".")[0] != local]
    if len(nodes) < 2 or not others:
        raise ValueError("Two distinct allocated nodes are required for this filesystem test")
    run_id = uuid4().hex
    folder = paths.require_project_output(
        paths.results_dir("storage_contracts", run_id, phase="experiment_0")
    )
    backend = storage_backend(folder)
    folder.mkdir(parents=True, exist_ok=False)
    manifest = (
        paths.manifests_dir("experiment_0")
        / "before_extraction"
        / "storage_contracts"
        / f"{run_id}.json"
    )
    report = {
        "status": "running",
        "job_id": os.environ["SLURM_JOB_ID"],
        "cluster": os.environ.get("SLURM_CLUSTER_NAME"),
        "storage_backend": backend,
        "nodes": [local, others[0]],
        "results": str(folder),
        "worker_receipts": {
            name: str(worker_receipt(run_id, expect_busy=busy))
            for name, busy in (("blocked", True), ("released", False))
        },
        "worker_logs": str(paths.logs_dir("experiment_0") / "storage_contracts" / run_id),
        "quota_verified": False,
        "independent_backup_verified": False,
    }
    from src.cluster.evidence import current_identity

    report["validation_identity"] = current_identity()
    write_json(manifest, report)
    try:
        with closing(sqlite3.connect(folder / "probe.sqlite3")) as db, db:
            db.execute("CREATE TABLE probe (id INTEGER PRIMARY KEY,value TEXT)")
            db.execute("INSERT INTO probe VALUES (1,'original')")
        command = [
            "srun",
            "--overlap",
            "--exact",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=1",
            f"--nodelist={others[0]}",
            sys.executable,
            "-m",
            "src.cluster.storage_contracts",
            "--worker",
            str(folder),
        ]
        with file_lock(folder / "writer.lock", timeout=2):
            first = remote_worker(command, run_id=run_id, expect_busy=True)
        report["cross_node_exclusion"] = first
        second = remote_worker(command, run_id=run_id, expect_busy=False)
        report["release_and_reacquire"] = second
        if first["status"] != "blocked_as_expected" or second["status"] != "acquired_after_release":
            raise AssertionError("Cross-node locking behavior differs from the required contract")
        if {first["node"], second["node"]} != {others[0].split(".")[0]}:
            raise AssertionError("Both filesystem probes must execute on the other allocated node")
        with (
            closing(sqlite3.connect(folder / "probe.sqlite3")) as source,
            closing(sqlite3.connect(folder / "restored.sqlite3")) as restored,
        ):
            source.backup(restored)
            if restored.execute("PRAGMA integrity_check").fetchone() != ("ok",) or restored.execute(
                "SELECT * FROM probe ORDER BY id"
            ).fetchall() != [(1, "original"), (2, "second_node")]:
                raise AssertionError("SQLite snapshot/restore lost committed data")
        report.update(
            status="storage_contracts_passed_requires_quota_and_backup_review",
            cross_node_exclusion=first,
            release_and_reacquire=second,
            snapshot_restore=True,
        )
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, error_message=str(exc))
        raise
    finally:
        write_json(manifest, report)
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker")
    parser.add_argument("--expect-busy", action="store_true")
    parser.add_argument("--probe-id")
    args = parser.parse_args()
    if args.worker:
        if not paths.on_vsc() or not os.environ.get("SLURM_JOB_ID"):
            parser.error("Worker requires the active Slurm test allocation")
        if not args.probe_id:
            parser.error("Worker requires its parent's unique --probe-id")
        result = worker(args.worker, expect_busy=args.expect_busy)
        result.update(
            probe_id=args.probe_id,
            job_id=os.environ["SLURM_JOB_ID"],
            expect_busy=args.expect_busy,
        )
        write_json(worker_receipt(Path(args.worker).name, expect_busy=args.expect_busy), result)
        print(json.dumps(result), flush=True)
    elif args.execute:
        run(execute=True)
    else:
        print("Prepared two-node filesystem test; no writes or allocation submission performed.")


if __name__ == "__main__":
    main()
