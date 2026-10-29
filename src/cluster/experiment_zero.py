"""Explicit Experiment 0A local/VSC entrypoint. Preview by default; never enables extraction.

VSC submits independent ready jobs concurrently, resuming its receipt after a disconnect.
Local smoke responses reuse their durable cache. --dataset-pilot additionally requires
a reviewed private Jev configuration. No passed receipt is inferred from submission.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from uuid import uuid4

from src.cluster.job_graph import AccountingUnavailable, advance
from src.cluster.maintenance import require_idle_vsc
from src.cluster.token_shards import plan as token_plan
from src.jev.catalog import SELF_HOSTED
from src.utils import paths
from src.utils.logging_setup import get_logger
from src.utils.serialization import digest, write_json

PROVIDERS = SELF_HOSTED


def steps(site, *, dataset_pilot=False, config="feature_creation/jev.local"):
    if site == "vsc" and dataset_pilot:
        raise ValueError(
            "--dataset-pilot selects the local Jev pilot; VSC always tests its eleven open-weight providers"
        )
    if site == "local":
        items = [
            ("libraries", ["src.utils.check_libraries"]),
            ("dependencies", ["pip", "check"]),
            ("questions", ["src.jev.question_audit", "--phase", "experiment_0"]),
            ("cpu", ["src.cluster.preflight", "--local", "--verify-data", "--tests"]),
            ("jev_contracts", ["src.jev.smoke", "--execute"]),
            ("jev_robustness", ["src.jev.smoke", "--robustness", "--execute"]),
        ]
        if dataset_pilot:
            items.append(
                (
                    "jev_real_robustness",
                    ["src.jev.smoke", "--dataset-robustness", "--config", config, "--execute"],
                )
            )
            items.append(("jev_dataset_pilot", ["src.jev.hosted", "--config", config, "--execute"]))
        return [
            {"name": n, "command": [sys.executable, "-X", "utf8", "-u", "-m", *c]} for n, c in items
        ]
    if site != "vsc":
        raise ValueError("Site must be local or vsc")
    from src.cluster.scheduling import gpu_profile, project_export

    default_profile = gpu_profile()
    durable = default_profile.get("durable_root", default_profile["staging_root"])
    export_root = project_export(durable)
    items = [
        ("cpu", "experiment_0_cpu.slurm", [durable]),
        ("storage", "experiment_0_storage.slurm", [durable]),
    ]
    from src.utils.config import load_yaml

    # Fresh CPU evidence must be staged to each native B200 working set.
    for runtime in PROVIDERS:
        items.append((runtime + "_scratch", "experiment_0_scratch_stage.slurm", [runtime]))
    for runtime in PROVIDERS:
        storage = "gpfs_storage" if runtime == "cloudflare" else runtime + "_gpfs_storage"
        items.append((storage, "experiment_0_storage_gpfs.slurm", [runtime]))
    for encoder_id in load_yaml("feature_creation/embedding_models")["models"]:
        items.append(
            (
                "embeddings_" + encoder_id,
                "experiment_0_embeddings.slurm",
                ["cloudflare", durable, encoder_id],
            )
        )
    for provider in PROVIDERS:
        for task, script in (
            ("contracts", "experiment_0_decision_contracts.slurm"),
            ("pilot", "experiment_0_decision_pilot.slurm"),
        ):
            items.append((f"{provider}_{task}", script, [provider]))
    scheduling = load_yaml("experiment_0/scheduling")
    token_steps = [dict(name=s["name"], cluster=s["cluster"], depends_on=s["dependencies"],
                   **({"pool": s["pool"]} if "pool" in s else {}),
                   command=["sbatch", "--parsable", export_root, *s["flags"], f"scripts/slurm/{s['script']}", *s["args"]])
              for provider in PROVIDERS for s in token_plan(provider, durable)]
    result = []
    for name, script, args in items:
        independent = name in {"cpu", "storage"} or name.endswith("_tokens")
        dependencies = [] if independent else ["cpu"]
        cluster, flags = "wice", []
        if name.endswith("_scratch"):
            dependencies = ["cpu", "storage", name.removesuffix("_scratch") + "_tokens"]
        elif name.endswith("gpfs_storage"):
            cluster = "mindwell"
            flags = ["--clusters=mindwell", "--partition=batch_graniterapids"]
            dependencies = [args[0] + "_scratch"]
        elif not independent:
            runtime = "cloudflare" if name.startswith("embeddings_") else args[0]
            profile = gpu_profile(
                scheduling["routes"].get(name, scheduling["default_gpu_profile"]), runtime=runtime
            )
            cluster, root = profile["cluster"], profile["staging_root"]
            flags = [f"--clusters={cluster}", f"--partition={profile['partition']}"]
            if name.startswith("embeddings_"):
                args = ["cloudflare", root, name.removeprefix("embeddings_")]
                dependencies = ["cloudflare_scratch", "gpfs_storage"]
            else:
                provider = args[0]
                args = [provider, provider, root]
                dependencies = [provider + "_scratch", "gpfs_storage" if provider == "cloudflare" else provider + "_gpfs_storage"]
                if name.endswith("_pilot"):
                    dependencies.append(provider + "_contracts")
                if provider in {"perplexity", "sii", "kev", "decision3"}:
                    flags.extend(["--mem=112000M", "--cpus-per-task=14"])
        result.append(
            {
                "name": name,
                "cluster": cluster,
                "depends_on": dependencies,
                "command": ["sbatch", "--parsable", export_root, *flags, f"scripts/slurm/{script}", *args],
            }
        )
    from src.cluster.workspace_graph import add_releases

    return add_releases(result[:2] + token_steps + result[2:], command_style=True)


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
    if site == "local":
        # Verify the live interface once for this frozen campaign. Preserve every
        # earlier diagnostic ledger; resuming this exact round never pays twice.
        for step in plan:
            if "src.jev.smoke" in step["command"]:
                step["command"] += ["--round-id", frozen["source_sha256"]]
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
        if site == "vsc":
            import re

            from src.utils.config import load_yaml

            def submit(step):
                step["status"] = "starting"
                write_json(receipt, report)
                try:
                    result = subprocess.run(step["command"], cwd=paths.REPO_ROOT,
                                            capture_output=True, text=True, check=True, timeout=60)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                    step["submission_diagnostics"] = {
                        key: str(getattr(exc, key, None) or "")[-4096:]
                        for key in ("stdout", "stderr")
                    }
                    log.error("Submission of %s failed; inspect Slurm before retrying. %s",
                              step["name"], json.dumps(step["submission_diagnostics"]))
                    raise
                step["submission_diagnostics"] = {
                    key: str(getattr(result, key, None) or "")[-4096:]
                    for key in ("stdout", "stderr")
                }
                job = result.stdout.strip()
                if not re.fullmatch(r"[0-9]+(?:;" + step["cluster"] + r")?", job):
                    log.error("Unrecognized submission response: %s", step["submission_diagnostics"])
                    raise ValueError("Unrecognized sbatch receipt; inspect scheduler before any retry")
                step.update(status="submitted_not_verified", job_id=job.split(";")[0])
                write_json(receipt, report)
                log.info("Submitted %s as %s", step["name"], step["job_id"])

            report["summary"] = advance(plan, settings=load_yaml("experiment_0/scheduling"),
                                        state=scheduler_state, submit=submit,
                                        save=lambda: write_json(receipt, report))
            report["status"] = (
                ("vsc_checks_completed_with_failures" if report["summary"]["failures"]
                 else "vsc_checks_completed_requires_review") if report["summary"]["all_terminal"]
                else "waiting_for_job"
            )
            return report
        for step in plan:
            command = step["command"][:]
            step["status"] = "starting"
            write_json(receipt, report)
            log.info("Starting %s", step["name"])
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
            "local_checks_completed_requires_review" if dataset_pilot
            else "local_checks_completed_dataset_pilot_pending"
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


def scheduler_state(job_id, *, cluster="wice"):
    command = [
        "sacct", f"--clusters={cluster}", "-j", job_id, "--noheader", "--parsable2",
        "--format=JobIDRaw,State,ExitCode",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        # A failed read cannot establish a job outcome. Submission exceptions are
        # deliberately handled elsewhere and must never use this retry path.
        raise AccountingUnavailable({
            "cluster": cluster, "job_id": job_id, "command": command,
            "error_type": type(exc).__name__, "returncode": getattr(exc, "returncode", None),
            **{key: str(getattr(exc, key, None) or "")[-4096:]
               for key in ("stdout", "stderr")},
        }) from exc
    for line in result.stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) >= 3 and fields[0] == job_id:
            if not fields[1].strip() or not fields[2].strip():
                return None  # An incomplete accounting record is not a terminal result.
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
        help="VSC: poll once/minute and submit ready independent tests after prerequisites pass; safe to interrupt/resume",
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
        print(json.dumps(report, indent=2) if not args.watch else
              json.dumps({"status": report["status"], **report.get("summary", {})}), flush=True)
        if not args.watch or report["status"] != "waiting_for_job":
            return 1 if report["status"] == "vsc_checks_completed_with_failures" else 0
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
