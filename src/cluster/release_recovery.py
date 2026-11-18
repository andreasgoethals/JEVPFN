"""Release with explicitly reviewed, hash-bound independent Slurm consumers.

This companion can run alongside frozen scientific source. It only removes exact
reviewed jobs from the consumer query; all normal publication and lease checks run.
Unknown jobs and changed scripts continue to prevent retirement.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


def filtered_reader(read, proof, runtime):
    if proof["release_runtime"] != runtime:
        raise ValueError("Recovery proof names a different workspace")
    records = {(x["cluster"], x["job_id"]): x for x in proof["independent_jobs"]}
    if len(records) != len(proof["independent_jobs"]):
        raise ValueError("Duplicate job proof")

    def checked(command):
        output = read(command)
        if command[0] != "squeue":
            return output
        cluster = next(x.split("=", 1)[1] for x in command if x.startswith("--clusters="))
        kept = []
        for line in output.splitlines():
            job = line.strip().split("|", 1)[0]
            record = records.get((cluster, job))
            if record is None and re.fullmatch(r"[0-9]+_(?:[0-9]+|\[[0-9,%\-]+\])", job):
                candidate = records.get((cluster, job.split("_", 1)[0]))
                if candidate and candidate.get("array") is True:
                    record = candidate
            if record is None:
                kept.append(line)
                continue
            # squeue's compact array ranges are display syntax, not scontrol IDs.
            query_job = record["job_id"] if "_[" in job else job
            detail = read(["scontrol", "--clusters=" + cluster, "show", "job", query_job, "-o"])
            commands = re.findall(r"(?:^|\s)Command=(.*?)(?=\s[A-Za-z][A-Za-z0-9_]*=|$)", detail)
            if not commands or any(Path(p).resolve() != Path(record["script"]).resolve()
                                   for p in commands):
                raise ValueError("Reviewed job command changed")
            if hashlib.sha256(Path(commands[0]).read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError("Reviewed independent job script changed")
        return "\n".join(kept)

    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--proof-sha256", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    payload = args.proof.read_bytes()
    if hashlib.sha256(payload).hexdigest() != args.proof_sha256:
        raise ValueError("Reviewed recovery proof changed")
    # Import the frozen checkout, not another version of scientific code.
    sys.path.insert(0, str(Path.cwd()))
    from src.cluster import scratch_capacity

    scratch_capacity._scheduler_read = filtered_reader(
        scratch_capacity._scheduler_read, json.loads(payload), args.runtime
    )
    sys.argv = [sys.argv[0], "--runtime", args.runtime] + (["--apply"] if args.apply else [])
    return scratch_capacity.main()


if __name__ == "__main__":
    raise SystemExit(main())
