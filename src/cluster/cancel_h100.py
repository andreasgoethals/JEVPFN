"""Preview/cancel only this user's verified JEVPFN H100 jobs; CPU jobs are untouched.

No cancellation happens without --apply. A project-like job name alone is never
enough: live Slurm WorkDir or Command must also resolve inside this checkout.
"""

import argparse
import getpass
import json
import os
import re
import subprocess
from pathlib import Path

from src.utils import paths


def _field(record, key):
    match = re.search(
        r"(?:^|\s)" + re.escape(key) + r"=(.*?)(?=\s+[A-Za-z][A-Za-z0-9_]*=|$)", record
    )
    return match.group(1).strip() if match else None


def _inside(value, checkout):
    if not value or not value.startswith("/"):
        return False
    return Path(value).resolve().is_relative_to(checkout)


def inspect_jobs():
    if not paths.on_vsc() or os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run from the VSC login shell, outside a Slurm job")
    user = getpass.getuser()
    checkout = paths.REPO_ROOT.resolve()
    listing = subprocess.check_output(
        ["squeue", "--clusters=wice", "--me", "--array", "--noheader", "--format=%i|%u|%P|%j"],
        text=True,
        timeout=30,
    )
    eligible, excluded = [], []
    for line in listing.splitlines():
        if not line.strip() or line.startswith("CLUSTER:"):
            continue
        fields = line.strip().split("|")
        if len(fields) != 4:
            raise ValueError("Cannot interpret Slurm job listing; no jobs were cancelled")
        job, owner, partition, name = (value.strip() for value in fields)
        if owner != user or partition != "gpu_h100":
            continue
        if not re.fullmatch(r"\d+(?:_\d+)?", job):
            raise ValueError("Ambiguous job/array identifier; no jobs were cancelled")
        record = subprocess.check_output(
            ["scontrol", "--clusters=wice", "--oneliner", "show", "job", job],
            text=True,
            timeout=30,
        )
        live_owner = (_field(record, "UserId") or "").split("(")[0]
        live_job = _field(record, "JobId")
        array_job = _field(record, "ArrayJobId")
        array_task = _field(record, "ArrayTaskId")
        same_job = live_job == job or (
            array_job is not None and array_task is not None and f"{array_job}_{array_task}" == job
        )
        matches = (
            live_owner == user
            and same_job
            and _field(record, "Partition") == "gpu_h100"
            and (
                _inside(_field(record, "WorkDir"), checkout)
                or _inside(_field(record, "Command"), checkout)
            )
        )
        item = {"job_id": job, "name": name, "partition": partition}
        (eligible if matches else excluded).append(item)
    return {
        "cluster": "wice",
        "checkout": str(checkout),
        "eligible": eligible,
        "excluded_h100": excluded,
    }


def run(*, apply=False):
    report = inspect_jobs()
    report.update(action="cancel_verified_project_h100_jobs", applied=False)
    if not apply or not report["eligible"]:
        return report
    # Re-read ownership/partition/paths immediately before mutation. Do not cancel
    # newly appearing jobs without including them in this invocation's selection.
    verified = {item["job_id"] for item in inspect_jobs()["eligible"]}
    selected = [item["job_id"] for item in report["eligible"] if item["job_id"] in verified]
    if selected:
        subprocess.run(["scancel", "--clusters=wice", *selected], check=True, timeout=30)
    report.update(applied=True, cancellation_requested=selected)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    print(json.dumps(run(apply=parser.parse_args().apply), indent=2))


if __name__ == "__main__":
    main()
