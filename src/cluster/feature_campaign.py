"""Bounded open-weight feature creation after reviewed 0A; preview by default.

Each GPU allocation handles one dataset, with two concurrent allocations per
resident decision runtime. The existing two-workspace lease/quota/publication
protocol bounds storage. Jev and all predictive benchmarks are excluded.
"""

import argparse
import json
import os
import subprocess
import time

from src.cluster.experiment_zero import scheduler_state
from src.cluster.freeze import check_if_present, runtime_pins
from src.cluster.job_graph import advance
from src.cluster.maintenance import require_idle_vsc
from src.cluster.scheduling import gpu_profile, project_export
from src.cluster.workspace_graph import add_releases
from src.data.loaders import load_catalog
from src.data.selection import selected_entries
from src.jev.catalog import SELF_HOSTED
from src.utils import paths
from src.utils.acceptance import reviewed_acceptance
from src.utils.config import load_yaml
from src.utils.locking import file_lock
from src.utils.serialization import digest, write_json


def plan():
    profile = gpu_profile()
    root = profile["durable_root"]
    export = project_export(root)
    ids = [e["id"] for e in selected_entries(load_catalog(), load_yaml("feature_creation"))]
    steps = []
    for provider in SELF_HOSTED:
        native = gpu_profile(runtime=provider)["staging_root"]
        steps.append(
            dict(
                name=provider + "_scratch",
                cluster="wice",
                depends_on=[],
                command=[
                    "sbatch",
                    "--parsable",
                    export,
                    "scripts/slurm/experiment_0_scratch_stage.slurm",
                    provider,
                ],
            )
        )
        for dataset in ids:
            large = provider in {"perplexity", "sii", "kev", "decision3"}
            steps.append(
                dict(
                    name=provider + "_" + dataset,
                    cluster="mindwell",
                    pool=provider,
                    depends_on=[provider + "_scratch"],
                    command=[
                        "sbatch",
                        "--parsable",
                        export,
                        "--clusters=mindwell",
                        "--partition=gpu_b200",
                        "--time=2-00:00:00",
                        "--cpus-per-task=14" if large else "--cpus-per-task=4",
                        "--mem=112000M" if large else "--mem=28000M",
                        "scripts/slurm/feature_creation.slurm",
                        provider,
                        provider,
                        native,
                        dataset,
                    ],
                )
            )
        if provider == "cloudflare":
            for encoder in load_yaml("feature_creation/embedding_models")["models"]:
                steps.append(
                    dict(
                        name="embeddings_" + encoder,
                        cluster="mindwell",
                        pool="gpu",
                        depends_on=[provider + "_scratch"],
                        command=[
                            "sbatch",
                            "--parsable",
                            export,
                            "--clusters=mindwell",
                            "scripts/slurm/embedding_creation.slurm",
                            encoder,
                            provider,
                            native,
                        ],
                    )
                )
    return add_releases(steps, command_style=True)


def run(*, execute=False):
    steps = plan()
    if not execute:
        return dict(
            status="preview",
            decision_datasets=18 * len(SELF_HOSTED),
            encoders=3,
            max_resident_workspaces=2,
            decision_gpu_jobs_per_runtime=2,
            steps=steps,
        )
    if not paths.on_vsc() or os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run the open-weight campaign on VSC")
    frozen = check_if_present()
    if not frozen["frozen"]:
        raise ValueError("Freeze accepted extraction configuration before starting")
    cfg = load_yaml("experiment_0/self_hosted")
    if cfg["full_feature_creation_enabled"] is not True:
        raise ValueError("Full extraction is disabled")
    for provider in SELF_HOSTED:
        reviewed_acceptance(
            cfg["accepted_before_extraction_manifests"].get(provider), stage="before_extraction"
        )
    for encoder in load_yaml("feature_creation/embedding_models")["models"]:
        from src.data.text_embeddings import encoder_config

        if encoder_config(encoder)["enabled"] is not True:
            raise ValueError("Enable the reviewed encoder configuration first")
    settings = load_yaml("experiment_0/scheduling")
    settings["test_pool_limits"] = {**settings["test_pool_limits"], **dict.fromkeys(SELF_HOSTED, 2)}
    receipt = paths.manifests_dir("feature_creation") / "open_weight_campaign.json"
    identity = digest(
        dict(frozen=frozen, runtime_pins=runtime_pins(), steps=steps, settings=settings)
    )
    if receipt.exists():
        report = json.loads(receipt.read_text())
        if report["identity"] != identity or any(
            s.get("status") == "starting" for s in report["steps"]
        ):
            raise ValueError("Campaign changed or submission is ambiguous; inspect before retrying")
        steps = report["steps"]
    else:
        require_idle_vsc()
        report = dict(identity=identity, steps=steps)
    for phase in ("experiment_0", "feature_creation"):
        paths.logs_dir(phase).mkdir(parents=True, exist_ok=True)

    def save():
        write_json(receipt, report)

    def submit(step):
        step["status"] = "starting"
        save()
        result = subprocess.run(
            step["command"],
            cwd=paths.REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        value = result.stdout.strip().split(";")[0]
        if not value.isdigit():
            raise ValueError("Ambiguous sbatch receipt; inspect accounting")
        step.update(job_id=value, status="submitted")
        save()

    report["summary"] = advance(
        steps, settings=settings, state=scheduler_state, submit=submit, save=save
    )
    report["status"] = (
        (
            "completed_requires_feature_validation"
            if not report["summary"]["failures"]
            else "completed_with_failures"
        )
        if report["summary"]["all_terminal"]
        else "running"
    )
    save()
    return dict(status=report["status"], **report["summary"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    if args.watch and not args.execute:
        parser.error("--watch requires --execute")
    if not args.execute:
        print(json.dumps(run()))
        return 0
    while True:
        with file_lock(paths.manifests_dir("feature_creation") / ".campaign.lock", timeout=1):
            report = run(execute=args.execute)
        print(json.dumps(report), flush=True)
        if not args.watch or report["status"] != "running":
            return int(report["status"] == "completed_with_failures")
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
