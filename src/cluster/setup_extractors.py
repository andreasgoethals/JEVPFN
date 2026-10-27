"""Prepare native B200 working sets and run bounded contracts; preview by default.

Each invocation reconciles recorded jobs and submits ready steps. --watch polls
once per minute; interruption is safe and the same command resumes. Cross-cluster
dependencies are checked through accounting, never assumed to be federated.
No final acceptance is granted and no environment is deleted.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime

from src.cluster.freeze import check_if_present, source_snapshot
from src.cluster.job_graph import advance
from src.cluster.maintenance import require_idle_vsc
from src.cluster.scheduling import gpu_profile
from src.cluster.token_shards import plan as token_plan
from src.jev.registry import ADDED, SELF_HOSTED
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.locking import file_lock
from src.utils.serialization import digest, write_json

LARGE = {"perplexity", "sii", "kev", "decision3"}


def plan(providers, *, reuse=()):
    providers = list(dict.fromkeys(providers))
    if not providers or set(providers) - set(SELF_HOSTED):
        raise ValueError("Choose selected open-weight providers")
    profile = gpu_profile()
    if profile.get("storage_mode") != "scratch_mirror":
        raise ValueError("Open-weight setup requires the B200 scratch-mirror profile")
    root = profile["durable_root"]
    steps = [
        dict(name="cpu", dependencies=[], script="experiment_0_cpu.slurm", args=[], flags=[]),
        dict(
            name="storage", dependencies=[], script="experiment_0_storage.slurm", args=[], flags=[]
        ),
    ]
    for step in steps:
        step["cluster"] = "wice"
    if "cloudflare" in providers:
        steps.append(
            dict(
                name="embedding_stage",
                dependencies=[],
                script="experiment_0_embedding_stage.slurm",
                args=[],
                flags=[],
                cluster="wice",
            )
        )
    for provider in providers:
        steps.append(
            dict(
                name=provider + "_stage",
                dependencies=[],
                script="experiment_0_stage.slurm",
                args=[provider, root],
                flags=[],
                cluster="wice",
            )
        )
        ready = provider + "_stage"
        if provider not in reuse:
            deps = [ready]
            steps.append(
                dict(
                    name=provider + "_build",
                    dependencies=deps,
                    script="experiment_0_container_build.slurm",
                    args=[provider, "new", root, profile["runtime_profile"]],
                    flags=[],
                    cluster="wice",
                )
            )
            ready = provider + "_build"
        steps.extend(token_plan(provider, root, dependencies=[provider + "_stage"]))
        steps.append(
            dict(
                name=provider + "_scratch",
                dependencies=list(
                    dict.fromkeys(
                        [
                            ready,
                            provider + "_tokens",
                            *(["embedding_stage"] if provider == "cloudflare" else []),
                        ]
                    )
                ),
                script="experiment_0_scratch_stage.slurm",
                args=[provider],
                flags=[],
                cluster="wice",
            )
        )
        storage = "gpfs_storage" if provider == "cloudflare" else provider + "_gpfs_storage"
        steps.append(dict(name=storage, dependencies=[provider + "_scratch"],
                          script="experiment_0_storage_gpfs.slurm", args=[provider],
                          flags=[], cluster="mindwell"))
        gpu_root = gpu_profile(runtime=provider)["staging_root"]
        gpu = [
            f"--partition={profile['partition']}",
            "--time=00:30:00",
            "--cpus-per-task=14" if provider in LARGE else "--cpus-per-task=4",
            "--mem=112000M" if provider in LARGE else "--mem=28000M",
        ]
        steps.append(
            dict(
                name=provider + "_contracts",
                dependencies=["cpu", "storage", storage, provider + "_scratch"],
                script="experiment_0_decision_contracts.slurm",
                args=[provider, provider, gpu_root],
                flags=gpu,
                cluster=profile["cluster"],
            )
        )
        steps.append(
            dict(
                name=provider + "_pilot",
                dependencies=[provider + "_contracts"],
                script="experiment_0_decision_pilot.slurm",
                args=[provider, provider, gpu_root],
                flags=[f for f in gpu if not f.startswith("--time=")] + ["--time=01:00:00"],
                cluster=profile["cluster"],
            )
        )
    if "cloudflare" in providers:
        root = gpu_profile(runtime="cloudflare")["staging_root"]
        for encoder in load_yaml("feature_creation/embedding_models")["models"]:
            steps.append(
                dict(
                    name="embeddings_" + encoder,
                    dependencies=["cpu", "storage", "gpfs_storage", "cloudflare_scratch"],
                    script="experiment_0_embeddings.slurm",
                    args=["cloudflare", root, encoder],
                    flags=["--partition=gpu_b200"],
                    cluster="mindwell",
                )
            )
    return steps


def run(providers, *, execute=False, restart=False):
    if execute:
        lock = paths.manifests_dir("experiment_0") / "container_setup" / ".submission.lock"
        with file_lock(lock, timeout=1):
            return _run(providers, execute=True, restart=restart)
    return _run(providers, restart=restart)


def _run(providers, *, execute=False, restart=False):
    from src.cluster import containers

    receipt = (
        paths.manifests_dir("experiment_0") / "container_setup" / "open_models_mindwell_b200.json"
    )
    runtime_profile = gpu_profile()["runtime_profile"]
    previous = json.loads(receipt.read_text()) if (execute or restart) and receipt.exists() else None
    replaced = None
    if restart:
        if previous is None:
            raise ValueError("No recorded setup to restart; use the normal setup command")
        replaced, previous = previous, None
    if previous and previous["providers"] != providers:
        raise ValueError(
            "Earlier setup selected different providers; inspect its jobs and receipt first"
        )
    reuse = (
        previous["reused_containers"]
        if previous
        else [
            p
            for p in providers
            if (containers.container_dir(p, profile=runtime_profile) / containers.MARKER).is_file()
        ]
    )
    steps = plan(providers, reuse=reuse)
    report = dict(
        status="preview",
        providers=providers,
        reused_containers=reuse,
        steps=steps,
        full_extraction_enabled=False,
        final_acceptance=False,
        runtime_profile=runtime_profile,
    )
    if replaced is not None:
        # Keep the evidence in the same compact receipt, without copying any results.
        report["prior_attempts"] = [
            *replaced.get("prior_attempts", []),
            {k: v for k, v in replaced.items() if k != "prior_attempts"},
        ]
        report["restart"] = "new_checks_reuse_verified_assets_preserve_outputs"
    elif previous and previous.get("prior_attempts"):
        report["prior_attempts"] = previous["prior_attempts"]
    if not execute:
        return report
    if not paths.on_vsc() or os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Submit from the VSC login terminal")
    if check_if_present()["frozen"]:
        raise ValueError("Runtime setup must precede the final frozen 0A campaign")
    identity = digest(dict(source=source_snapshot(), providers=providers, steps=steps))
    receipt.parent.mkdir(parents=True, exist_ok=True)
    paths.logs_dir("experiment_0").mkdir(parents=True, exist_ok=True)
    if previous:
        saved = json.loads(receipt.read_text())
        if saved["identity"] != identity:
            raise ValueError(
                "Earlier setup belongs to different source/providers; review it, wait for idle jobs, then use --restart"
            )
        steps = saved["steps"]
        if saved["status"] == "stopped_requires_review":
            raise ValueError(
                "Earlier setup stopped; inspect its recorded jobs and errors, then use --restart after all jobs finish"
            )
    else:
        require_idle_vsc()
        if replaced is not None:
            # An empty squeue does not resolve an interrupted submission. Require
            # accounting evidence for every known job and refuse ambiguous submits.
            from src.cluster.experiment_zero import scheduler_state

            for step in replaced["steps"]:
                if step.get("job_id"):
                    if scheduler_state(step["job_id"], cluster=step["cluster"]) is None:
                        raise ValueError("Recorded setup job is active or absent from accounting; restart refused")
                elif step.get("status") == "starting":
                    raise ValueError("Unrecorded submission requires Slurm review; restart refused")
        # Reused images are verified in each provider's CPU scratch-stage job.
        # Hashing large images on the login node would serialize setup and let one
        # corrupt image prevent unrelated providers from being diagnosed.
    report.update(identity=identity, status="submitting", steps=steps)
    report["created_utc"] = previous.get("created_utc") if previous else datetime.now(UTC).isoformat()
    report["updated_utc"] = datetime.now(UTC).isoformat()
    write_json(receipt, report)
    active_step = None
    try:
        from src.cluster.experiment_zero import scheduler_state

        def save():
            write_json(receipt, report)

        def submit(step):
            nonlocal active_step
            active_step = step
            command = ["sbatch", "--parsable", f"--clusters={step['cluster']}",
                       f"--account={load_yaml('experiment_0/scheduling')['account']}",
                       *step["flags"], "scripts/slurm/" + step["script"], *step["args"]]
            step.update(status="starting", command=command)
            save()
            result = subprocess.run(command, cwd=paths.REPO_ROOT, text=True,
                                    capture_output=True, timeout=60, check=True)
            step["submission_diagnostics"] = {
                key: str(getattr(result, key, None) or "")[-4096:]
                for key in ("stdout", "stderr")
            }
            value = result.stdout.strip()
            if not re.fullmatch(r"\d+(;" + step["cluster"] + r")?", value):
                raise ValueError("Ambiguous sbatch response; inspect Slurm rather than submitting a duplicate")
            step.update(job_id=value.split(";")[0], status="submitted")
            save()
            print(f"Submitted {step['name']}: {step['job_id']}", flush=True)

        report["summary"] = advance(steps, settings=load_yaml("experiment_0/scheduling"),
                                    state=scheduler_state, submit=submit, save=save)
        report["status"] = (
            ("setup_completed_with_failures" if report["summary"]["failures"]
             else "setup_completed_requires_review") if report["summary"]["all_terminal"]
            else "waiting_for_job"
        )
    except BaseException as exc:
        report.update(
            status="stopped_requires_review", error_type=type(exc).__name__, error=str(exc)
        )
        if active_step is not None:
            diagnostics = active_step.setdefault("submission_diagnostics", {})
            for key in ("stdout", "stderr"):
                value = getattr(exc, key, None)
                if value is not None:
                    diagnostics[key] = str(value)[-4096:]
            print(json.dumps({"step": active_step["name"], "receipt": str(receipt),
                              "submission_diagnostics": diagnostics,
                              "action": "Inspect Slurm before retrying; acceptance may be ambiguous"}),
                  file=sys.stderr, flush=True)
        raise
    finally:
        write_json(receipt, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", action="append", choices=SELF_HOSTED)
    parser.add_argument(
        "--all", action="store_true", help="Set up/test all ten open-weight extractors"
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--restart", action="store_true",
        help="After reviewing a stopped/old setup, start fresh checks while preserving outputs and sealed assets; requires idle jobs",
    )
    parser.add_argument(
        "--watch", action="store_true", help="Poll once/minute; safe to interrupt/resume"
    )
    args = parser.parse_args(argv)
    if args.all and args.provider:
        parser.error("Use --all or --provider, not both")
    if args.watch and not args.execute:
        parser.error("--watch requires --execute")
    while True:
        report = run(
            list(SELF_HOSTED if args.all else args.provider or ADDED),
            execute=args.execute, restart=args.restart,
        )
        args.restart = False  # Subsequent watcher polls resume the new attempt.
        print(json.dumps(report, indent=2) if not args.watch else
              json.dumps({"status": report["status"], **report.get("summary", {})}), flush=True)
        if not args.watch or report["status"] != "waiting_for_job":
            return 1 if report["status"] == "setup_completed_with_failures" else 0
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
