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
import time

from src.cluster.freeze import check_if_present, source_snapshot
from src.cluster.maintenance import require_idle_vsc
from src.cluster.scheduling import gpu_profile
from src.cluster.token_preparation import resources as token_resources
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
    for i, provider in enumerate(providers):
        steps.append(
            dict(
                name=provider + "_stage",
                dependencies=[providers[i - 2] + "_stage"] if i >= 2 else [],
                script="experiment_0_stage.slurm",
                args=[provider, root],
                flags=[],
                cluster="wice",
            )
        )
        ready = provider + "_stage"
        if provider not in reuse:
            deps = [ready]
            if i >= 2 and providers[i - 2] not in reuse:
                deps.append(providers[i - 2] + "_build")
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
        steps.append(
            dict(
                name=provider + "_tokens",
                dependencies=[provider + "_stage"],
                script="experiment_0_tokens.slurm",
                args=[root, provider],
                flags=token_resources(provider),
                cluster="wice",
            )
        )
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
        if i == 0:
            steps.append(
                dict(
                    name="gpfs_storage",
                    dependencies=[provider + "_scratch"],
                    script="experiment_0_storage_gpfs.slurm",
                    args=[provider],
                    flags=[],
                    cluster="mindwell",
                )
            )
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
                dependencies=["cpu", "storage", "gpfs_storage", provider + "_scratch"],
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


def run(providers, *, execute=False):
    if execute:
        lock = paths.manifests_dir("experiment_0") / "container_setup" / ".submission.lock"
        with file_lock(lock, timeout=1):
            return _run(providers, execute=True)
    return _run(providers)


def _run(providers, *, execute=False):
    from src.cluster import containers

    receipt = (
        paths.manifests_dir("experiment_0") / "container_setup" / "open_models_mindwell_b200.json"
    )
    runtime_profile = gpu_profile()["runtime_profile"]
    previous = json.loads(receipt.read_text()) if execute and receipt.exists() else None
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
                "Earlier setup belongs to different source/providers; review or clean that receipt first"
            )
        steps = saved["steps"]
        if saved["status"] == "stopped_requires_review":
            raise ValueError(
                "Earlier setup stopped; inspect its recorded jobs and errors before resuming"
            )
    else:
        require_idle_vsc()
        for provider in reuse:
            manifest = containers.verify(provider, profile=runtime_profile)
            if manifest.get("runtime_profile") != runtime_profile:
                raise ValueError("Existing image does not match the B200 runtime profile")
    report.update(identity=identity, status="submitting", steps=steps)
    write_json(receipt, report)
    active_step = None
    try:
        from src.cluster.experiment_zero import scheduler_state

        # Reconcile every active job before dispatching any downstream work. An
        # accounting delay keeps the job pending; a nonzero terminal state stops.
        for step in steps:
            if step.get("job_id") and step.get("status") != "completed":
                state = scheduler_state(step["job_id"], cluster=step["cluster"])
                if state is not None:
                    if state != ("COMPLETED", "0:0"):
                        raise RuntimeError(
                            f"Setup job {step['job_id']} ended {state}; inspect its log"
                        )
                    step["status"] = "completed"
            elif step.get("status") == "starting":
                raise ValueError(
                    "Submission was interrupted without a receipt; inspect Slurm before retrying"
                )
        write_json(receipt, report)
        limit = load_yaml("experiment_0/scheduling")["max_active_jobs"]
        if type(limit) is not int or limit < 1:
            raise ValueError("max_active_jobs must be a positive integer")
        for step in steps:
            active_step = step
            if step.get("job_id"):
                continue
            if step.get("status") == "starting":
                raise ValueError(
                    "Submission was interrupted without a receipt; inspect Slurm before retrying"
                )
            completed = {s["name"] for s in steps if s.get("status") == "completed"}
            active = sum(bool(s.get("job_id")) and s.get("status") != "completed" for s in steps)
            if active >= limit or not set(step["dependencies"]) <= completed:
                continue
            command = [
                "sbatch",
                "--parsable",
                f"--clusters={step['cluster']}",
                f"--account={load_yaml('experiment_0/scheduling')['account']}",
                *step["flags"],
            ]
            command += ["scripts/slurm/" + step["script"], *step["args"]]
            step.update(status="starting", command=command)
            write_json(receipt, report)
            result = subprocess.run(
                command, cwd=paths.REPO_ROOT, text=True, capture_output=True, timeout=60, check=True
            )
            value = result.stdout.strip()
            if not re.fullmatch(r"\d+(;" + step["cluster"] + r")?", value):
                raise ValueError(
                    "Ambiguous sbatch response; inspect Slurm rather than submitting a duplicate"
                )
            step.update(job_id=value.split(";")[0], status="submitted")
            write_json(receipt, report)
            print(f"Submitted {step['name']}: {step['job_id']}", flush=True)
        report["status"] = (
            "setup_completed_requires_review"
            if all(s.get("status") == "completed" for s in steps)
            else "waiting_for_job"
        )
    except BaseException as exc:
        report.update(
            status="stopped_requires_review", error_type=type(exc).__name__, error=str(exc)
        )
        if active_step is not None:
            active_step["submission_diagnostics"] = {
                key: str(getattr(exc, key, None) or "")[-4096:] for key in ("stdout", "stderr")
            }
        raise
    finally:
        write_json(receipt, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", action="append", choices=SELF_HOSTED)
    parser.add_argument(
        "--all", action="store_true", help="Set up/test all ten open-weight extractors"
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--watch", action="store_true", help="Poll once/minute; safe to interrupt/resume"
    )
    args = parser.parse_args()
    if args.all and args.provider:
        parser.error("Use --all or --provider, not both")
    if args.watch and not args.execute:
        parser.error("--watch requires --execute")
    while True:
        report = run(
            list(SELF_HOSTED if args.all else args.provider or ADDED), execute=args.execute
        )
        print(json.dumps(report, indent=2) if not args.watch else report["status"], flush=True)
        if not args.watch or report["status"] != "waiting_for_job":
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
