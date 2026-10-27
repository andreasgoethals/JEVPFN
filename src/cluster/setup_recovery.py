"""Record a confirmed, terminal job after an ambiguous setup submission.

Preview by default. This only repairs the job ID in the stopped setup receipt;
it never submits jobs, grants acceptance or deletes/replaces computed results.
"""

import argparse
import getpass
import json
import re
import shlex
import subprocess
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from src.cluster.maintenance import require_idle_vsc
from src.utils import paths
from src.utils.locking import file_lock
from src.utils.serialization import digest, write_json

FIELDS = ("JobIDRaw", "User", "Account", "State", "ExitCode", "WorkDir", "SubmitLine")
TERMINAL = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "BOOT_FAIL",
    "DEADLINE",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
}


def _accounting(job_id, cluster):
    result = subprocess.run(
        [
            "sacct",
            f"--clusters={cluster}",
            "-j",
            job_id,
            "--allocations",
            "--noheader",
            "--parsable2",
            "--format=" + ",".join(FIELDS),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    rows = [line.strip().split("|") for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != len(FIELDS) or rows[0][0] != job_id:
        raise ValueError("Expected exactly one complete Slurm allocation record; receipt unchanged")
    return dict(zip(FIELDS, (value.strip() for value in rows[0]), strict=True))


def _validate(step, evidence):
    command = step.get("command", [])
    if not command or PurePosixPath(command[0]).name != "sbatch":
        raise ValueError("Recorded step has no sbatch command")
    account = [part.split("=", 1)[1] for part in command if part.startswith("--account=")]
    if len(account) != 1 or evidence["Account"] != account[0]:
        raise ValueError("Slurm account does not match the recorded submission")
    if evidence["User"] != getpass.getuser():
        raise ValueError("Slurm job belongs to a different user")
    if (
        not Path(evidence["WorkDir"]).is_absolute()
        or Path(evidence["WorkDir"]).resolve() != paths.REPO_ROOT.resolve()
    ):
        raise ValueError("Slurm job belongs to a different checkout")
    submitted = shlex.split(evidence["SubmitLine"])
    if submitted:
        submitted[0] = PurePosixPath(submitted[0]).name
    if submitted != ["sbatch", *command[1:]]:
        raise ValueError("Slurm submission command does not match the recorded step")
    if not evidence["State"] or evidence["State"].split()[0] not in TERMINAL:
        raise ValueError("Slurm job is not in a recognized terminal state")
    if not re.fullmatch(r"\d+:\d+", evidence["ExitCode"]):
        raise ValueError("Slurm exit code is unavailable")


def run(step_name, job_id, *, apply=False):
    if not re.fullmatch(r"[1-9]\d*", job_id):
        raise ValueError("Provide one numeric allocation job ID")
    if apply:
        require_idle_vsc()
    directory = paths.manifests_dir("experiment_0") / "container_setup"
    receipt = directory / "open_models_mindwell_b200.json"
    if apply:
        with file_lock(directory / ".submission.lock", timeout=1):
            return _recover(receipt, step_name, job_id, apply=True)
    return _recover(receipt, step_name, job_id, apply=False)


def _recover(receipt, step_name, job_id, *, apply):
    report = json.loads(receipt.read_text(encoding="utf-8"))
    if report.get("status") != "stopped_requires_review":
        raise ValueError("Recovery requires a stopped setup receipt")
    matches = [step for step in report["steps"] if step["name"] == step_name]
    if len(matches) != 1:
        raise ValueError("Select exactly one recorded setup step")
    step = matches[0]
    already = bool(step.get("submission_recovery")) and step.get("job_id") == job_id
    if not already and (step.get("status") != "starting" or step.get("job_id")):
        raise ValueError("Only an unrecorded, ambiguous submission can be recovered")
    if step.get("cluster") not in {"wice", "mindwell"}:
        raise ValueError("Unrecognized submission cluster")
    if any(
        s is not step and s.get("cluster") == step["cluster"] and s.get("job_id") == job_id
        for s in report["steps"]
    ):
        raise ValueError("Job ID is already assigned to another step")
    evidence = _accounting(job_id, step["cluster"])
    _validate(step, evidence)
    result = dict(
        status="already_recorded" if already else "preview",
        step=step_name,
        job_id=job_id,
        cluster=step["cluster"],
        accounting=evidence,
        receipt=str(receipt),
        jobs_submitted=False,
        final_acceptance=False,
    )
    if apply and not already:
        step["submission_recovery"] = dict(
            recorded_utc=datetime.now(UTC).isoformat(),
            prior_receipt_sha256=digest(report),
            accounting=evidence,
        )
        # Keep the original diagnostics and stopped status. The normal restart
        # path will review accounting and recheck the current source independently.
        step.update(job_id=job_id, status="submitted")
        write_json(receipt, report)
        result["status"] = "recorded_requires_restart"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.step, args.job_id, apply=args.apply), indent=2))


if __name__ == "__main__":
    main()
