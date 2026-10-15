"""Opt-in two-node PROJECT lock and SQLite snapshot/restore check; no research data.

The local test suite cannot establish cross-node flock behavior on a Lustre mount.
This probe therefore runs in its own tiny two-node CPU allocation before extraction.
"""

import argparse
import json
import os
import socket
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from src.cluster.preflight import storage_policy_failures
from src.utils import paths
from src.utils.locking import file_lock
from src.utils.serialization import write_json


def node_name():
    return os.environ.get("SLURMD_NODENAME", socket.gethostname()).split(".")[0]


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
        "nodes": [local, others[0]],
        "results": str(folder),
        "quota_verified": False,
        "independent_backup_verified": False,
    }
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
            first = json.loads(
                subprocess.check_output(command + ["--expect-busy"], text=True, timeout=60)
            )
        second = json.loads(subprocess.check_output(command, text=True, timeout=60))
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
        report.update(status="failed", error_type=type(exc).__name__)
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
    args = parser.parse_args()
    if args.worker:
        if not paths.on_vsc() or not os.environ.get("SLURM_JOB_ID"):
            parser.error("Worker requires the active Slurm test allocation")
        print(json.dumps(worker(args.worker, expect_busy=args.expect_busy)))
    elif args.execute:
        run(execute=True)
    else:
        print("Prepared two-node filesystem test; no writes or allocation submission performed.")


if __name__ == "__main__":
    main()
