"""Explicit Experiment 0A local/VSC entrypoint. Preview by default; never enables extraction.

VSC submits one job at a time and resumes from its receipt after a terminal disconnect.
Local smoke responses reuse their durable cache. --dataset-pilot additionally requires
a reviewed private Jev configuration. No passed receipt is inferred from submission.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from uuid import uuid4

from src.cluster.maintenance import require_idle_vsc
from src.utils import paths
from src.utils.logging_setup import get_logger
from src.utils.serialization import digest, write_json

PROVIDERS = ("cloudflare", "intern", "liquid", "strands", "perplexity")


def steps(site, *, dataset_pilot=False, config="feature_creation/jev.local"):
    if site == "vsc" and dataset_pilot:
        raise ValueError(
            "--dataset-pilot selects the local Jev pilot; VSC always tests its five providers"
        )
    if site == "local":
        items = [
            ("dependencies", ["pip", "check"]),
            ("questions", ["src.jev.question_audit", "--phase", "experiment_0"]),
            ("cpu", ["src.cluster.preflight", "--local", "--verify-data", "--tests"]),
            ("jev_contracts", ["src.jev.smoke", "--execute"]),
            ("jev_robustness", ["src.jev.smoke", "--robustness", "--execute"]),
        ]
        if dataset_pilot:
            items.append(("jev_dataset_pilot", ["src.jev.hosted", "--config", config, "--execute"]))
        return [
            {"name": n, "command": [sys.executable, "-X", "utf8", "-u", "-m", *c]} for n, c in items
        ]
    if site != "vsc":
        raise ValueError("Site must be local or vsc")
    items = [
        ("cpu", "experiment_0_cpu.slurm", []),
        ("storage", "experiment_0_storage.slurm", []),
        ("tokens", "experiment_0_tokens.slurm", []),
        ("embeddings", "experiment_0_embeddings.slurm", []),
    ]
    for provider in PROVIDERS:
        for task, script in (
            ("contracts", "experiment_0_decision_contracts.slurm"),
            ("pilot", "experiment_0_decision_pilot.slurm"),
        ):
            items.append((f"{provider}_{task}", script, [provider]))
    return [
        {
            "name": name,
            "command": [
                "sbatch",
                "--parsable",
                *(["--mem=112000M"] if name.startswith("perplexity_") else []),
                f"scripts/slurm/{script}",
                *args,
            ],
        }
        for name, script, args in items
    ]


def run(site, *, execute=False, dataset_pilot=False, config="feature_creation/jev.local"):
    plan = steps(site, dataset_pilot=dataset_pilot, config=config)
    report = {
        "stage": "0A_" + site,
        "status": "preview",
        "steps": plan,
        "full_feature_creation_enabled": False,
        "acceptance": "Separate review of current reports, quota, backup and throughput is required",
    }
    if not execute:
        return report
    if (site == "vsc") != paths.on_vsc() or os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Use the local laptop or VSC login shell matching --site")
    from src.cluster.freeze import check_if_present, runtime_pins

    frozen = check_if_present()
    if not frozen["frozen"]:
        raise ValueError("Record the final frozen revision after runtime setup and cleanup first")
    identity = digest({"source": frozen, "runtime_pins": runtime_pins(), "steps": plan})
    report["stage_identity"] = identity
    if site == "local" and dataset_pilot:
        from src.jev.extraction import campaign_limits
        from src.jev.hosted import ReviewedContextEnvelope
        from src.utils.config import load_yaml

        cfg = load_yaml(config)
        campaign_limits(cfg)
        ReviewedContextEnvelope(cfg)  # Fail before running tests if configuration is missing.
        if cfg["live_extraction"].get("enabled") is not True:
            raise ValueError("Enable the reviewed private pilot configuration first")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    folder = paths.manifests_dir("experiment_0") / report["stage"]
    receipt = folder / f"{run_id}.json"
    if site == "vsc":
        receipts = list(folder.glob("*.json"))
        if folder.exists() and len(receipts) != 1:
            raise ValueError("Ambiguous 0A_VSC receipt directory; inspect before submitting")
        if receipts:
            receipt = receipts[0]
            report = json.loads(receipt.read_text(encoding="utf-8"))
            if report.get("stage_identity") != identity:
                raise ValueError(
                    "Saved 0A sequence belongs to different source/runtime/settings; review before resuming"
                )
            plan = report["steps"]
            run_id = report["run_id"]
            if report["status"] == "stopped_requires_review":
                raise ValueError(
                    "Earlier 0A submission stopped; inspect job receipts before retrying"
                )
        else:
            require_idle_vsc()
            folder.mkdir(parents=True, exist_ok=False)
    log = get_logger(
        "jevpfn.exp0", paths.logs_dir("experiment_0") / report["stage"] / f"{run_id}.log"
    )
    report.update(status="running" if site == "local" else "submitting", run_id=run_id)
    write_json(receipt, report)
    try:
        for step in plan:
            if step.get("status") == "completed":
                continue
            if site == "vsc" and step.get("job_id"):
                state = scheduler_state(step["job_id"])
                if state is None:
                    report["status"] = "waiting_for_job"
                    log.info("Waiting for %s (job %s)", step["name"], step["job_id"])
                    return report
                if state != ("COMPLETED", "0:0"):
                    raise RuntimeError(f"Job {step['job_id']} ended {state}; inspect its Slurm log")
                step["status"] = "completed"
                write_json(receipt, report)
                continue
            if site == "vsc" and step.get("status") == "starting":
                raise ValueError(
                    "Interrupted submission has no job ID; inspect Slurm before retrying"
                )
            if site == "vsc":
                require_idle_vsc()
            command = step["command"][:]
            step.update(status="starting", command=command)
            write_json(receipt, report)  # Interrupted submission is explicit; never auto-resubmit.
            log.info("Starting %s", step["name"])
            if site == "vsc":
                result = subprocess.run(
                    command,
                    cwd=paths.REPO_ROOT,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=60,
                )
                job = result.stdout.strip()
                if not re.fullmatch(r"[0-9]+(?:;wice)?", job):
                    raise ValueError(
                        "Unrecognized sbatch receipt; inspect scheduler before any retry"
                    )
                job_id = job.split(";")[0]
                step.update(status="submitted_not_verified", job_id=job_id)
                log.info("Submitted %s as %s", step["name"], job_id)
                report["status"] = "waiting_for_job"
                return report
            else:
                with subprocess.Popen(
                    command,
                    cwd=paths.REPO_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                ) as process:
                    for line in process.stdout:
                        log.info("%s", line.rstrip())
                    if process.wait():
                        raise RuntimeError(f"{step['name']} failed; see {receipt}")
                step["status"] = "completed"
            write_json(receipt, report)
        report["status"] = (
            "vsc_checks_completed_requires_review"
            if site == "vsc"
            else (
                "local_checks_completed_requires_review"
                if dataset_pilot
                else "local_checks_completed_dataset_pilot_pending"
            )
        )
    except BaseException as exc:
        report.update(status="stopped_requires_review", error_type=type(exc).__name__)
        raise
    finally:
        write_json(receipt, report)
        for handler in list(log.handlers):
            handler.close()
            log.removeHandler(handler)
    return report


def scheduler_state(job_id):
    result = subprocess.run(
        [
            "sacct",
            "--clusters=wice",
            "-j",
            job_id,
            "--noheader",
            "--parsable2",
            "--format=JobIDRaw,State,ExitCode",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    for line in result.stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) >= 3 and fields[0] == job_id:
            state = fields[1].split()[0]
            if state in {
                "PENDING",
                "RUNNING",
                "CONFIGURING",
                "COMPLETING",
                "SUSPENDED",
                "REQUEUED",
                "RESIZING",
            }:
                return None
            return state, fields[2]
    return None  # Scheduler accounting may briefly lag job submission.


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True, choices=("local", "vsc"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--dataset-pilot",
        action="store_true",
        help="Include the explicitly configured local paid dataset pilot",
    )
    parser.add_argument("--config", default="feature_creation/jev.local")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="VSC: poll once/minute and submit the next test after success; safe to interrupt/resume",
    )
    args = parser.parse_args(argv)
    if args.watch and (args.site != "vsc" or not args.execute):
        parser.error("--watch requires --site vsc --execute")
    from src.utils.locking import file_lock

    while True:
        if args.execute and args.site == "vsc":
            with file_lock(paths.manifests_dir("experiment_0") / ".0A_submission.lock", timeout=1):
                report = run(
                    args.site, execute=True, dataset_pilot=args.dataset_pilot, config=args.config
                )
        else:
            report = run(
                args.site,
                execute=args.execute,
                dataset_pilot=args.dataset_pilot,
                config=args.config,
            )
        print(json.dumps(report, indent=2) if not args.watch else report["status"], flush=True)
        if not args.watch or report["status"] != "waiting_for_job":
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
