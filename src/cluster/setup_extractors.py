"""Submit a resumable CPU-stage/container/token/GPU setup graph; preview by default.

No final acceptance is granted. Slurm dependencies survive a disconnected terminal.
No environment is deleted. Existing sealed images are reused after verification.
"""

import argparse
import json
import os
import re
import subprocess

from src.cluster.freeze import check_if_present, source_snapshot
from src.cluster.maintenance import require_idle_vsc
from src.cluster.scheduling import gpu_profile
from src.jev.registry import ADDED, SELF_HOSTED
from src.utils import paths
from src.utils.locking import file_lock
from src.utils.serialization import digest, write_json

LARGE = {"perplexity", "sii", "kev", "decision3"}


def plan(providers, *, reuse=()):
    providers = list(dict.fromkeys(providers))
    if not providers or set(providers) - set(SELF_HOSTED):
        raise ValueError("Choose selected open-weight providers")
    profile = gpu_profile("wice_h100")
    root = profile["staging_root"]
    steps = [
        dict(name="cpu", dependencies=[], script="experiment_0_cpu.slurm", args=[], flags=[]),
        dict(
            name="storage", dependencies=[], script="experiment_0_storage.slurm", args=[], flags=[]
        ),
    ]
    for i, provider in enumerate(providers):
        steps.append(
            dict(
                name=provider + "_stage",
                dependencies=[providers[i - 2] + "_stage"] if i >= 2 else [],
                script="experiment_0_stage.slurm",
                args=[provider, root],
                flags=[],
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
                    args=[provider, "new", root],
                    flags=[],
                )
            )
            ready = provider + "_build"
        steps.append(
            dict(
                name=provider + "_tokens",
                dependencies=[provider + "_stage"],
                script="experiment_0_tokens.slurm",
                args=[root, provider],
                flags=["--cpus-per-task=2", "--mem=6800M"],
            )
        )
        gpu = [
            f"--partition={profile['partition']}",
            "--time=00:30:00",
            "--cpus-per-task=8" if provider in LARGE else "--cpus-per-task=4",
            "--mem=112000M" if provider in LARGE else "--mem=28000M",
        ]
        steps.append(
            dict(
                name=provider + "_contracts",
                dependencies=["cpu", "storage", ready, provider + "_tokens"],
                script="experiment_0_decision_contracts.slurm",
                args=[provider, provider, root],
                flags=gpu,
            )
        )
        steps.append(
            dict(
                name=provider + "_pilot",
                dependencies=[provider + "_contracts"],
                script="experiment_0_decision_pilot.slurm",
                args=[provider, provider, root],
                flags=[f for f in gpu if not f.startswith("--time=")] + ["--time=01:00:00"],
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

    receipt = paths.manifests_dir("experiment_0") / "container_setup" / "open_models.json"
    previous = json.loads(receipt.read_text()) if execute and receipt.exists() else None
    if previous and previous["providers"] != providers:
        raise ValueError(
            "Earlier setup selected different providers; inspect its jobs and receipt first"
        )
    reuse = (
        previous["reused_containers"]
        if previous
        else [p for p in providers if (containers.container_dir(p) / containers.MARKER).is_file()]
    )
    steps = plan(providers, reuse=reuse)
    report = dict(
        status="preview",
        providers=providers,
        reused_containers=reuse,
        steps=steps,
        full_extraction_enabled=False,
        final_acceptance=False,
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
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved["identity"] != identity:
            raise ValueError(
                "Earlier setup belongs to different source/providers; review or clean that receipt first"
            )
        steps = saved["steps"]
    else:
        require_idle_vsc()
    for provider in reuse:
        containers.verify(provider)
    report.update(identity=identity, status="submitting", steps=steps)
    write_json(receipt, report)
    active_step = None
    try:
        ids = {s["name"]: s["job_id"] for s in steps if s.get("job_id")}
        for step in steps:
            active_step = step
            if step.get("job_id"):
                continue
            if step.get("status") == "starting":
                raise ValueError(
                    "Submission was interrupted without a receipt; inspect Slurm before retrying"
                )
            deps = list(dict.fromkeys(ids[n] for n in step["dependencies"]))
            command = ["sbatch", "--parsable", "--clusters=wice", *step["flags"]]
            if deps:
                command.append("--dependency=afterok:" + ":".join(deps))
            command += ["scripts/slurm/" + step["script"], *step["args"]]
            step.update(status="starting", command=command)
            write_json(receipt, report)
            result = subprocess.run(
                command, cwd=paths.REPO_ROOT, text=True, capture_output=True, timeout=60, check=True
            )
            value = result.stdout.strip()
            if not re.fullmatch(r"\d+(;wice)?", value):
                raise ValueError(
                    "Ambiguous sbatch response; inspect Slurm rather than submitting a duplicate"
                )
            step.update(job_id=value.split(";")[0], status="submitted")
            ids[step["name"]] = step["job_id"]
            write_json(receipt, report)
            print(f"Submitted {step['name']}: {step['job_id']}", flush=True)
        report["status"] = "submitted_requires_live_results"
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
    args = parser.parse_args()
    if args.all and args.provider:
        parser.error("Use --all or --provider, not both")
    print(
        json.dumps(
            run(list(SELF_HOSTED if args.all else args.provider or ADDED), execute=args.execute),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
