"""Write a bounded Slurm array plan; never submit jobs or enable experiments.

One item is one dataset/model/representation (five folds, resumed by the evaluation
store). CPU-only predictors request no GPU. One profile per manifest prevents
speculative duplicate submissions on several clusters.
"""

import argparse
import json
import os
import shlex

from src.evaluation.campaign import benchmark_plan, run_arm
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.serialization import digest, write_json


def plan(phase, model, *, profile=None, concurrency=8, controls=False, profile_fold=False):
    from src.cluster.scheduling import gpu_profile

    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("Positive array concurrency required")
    runtime = load_yaml("experiment_1/models")["models"][model]["runtime"]
    gpu = runtime["device"].startswith("cuda")
    route = gpu_profile(profile if gpu else "wice_h100", runtime=model)
    candidates = [
        a
        for a in benchmark_plan(include_controls=controls)
        if a["phase"] == phase and a["model"] == model
    ]
    arms = [a for a in candidates if a["width_status"] != "unsupported_width"]
    excluded = [a for a in candidates if a["width_status"] == "unsupported_width"]
    if not arms:
        raise ValueError("No supported arms for this phase/model")
    return {
        "version": 1,
        "phase": phase,
        "model": model,
        "route": route,
        "concurrency": concurrency,
        "gpu": gpu,
        "arms": arms,
        "excluded_arms": excluded,
        "profile_fold": profile_fold,
    }


def write_plan(body):
    from src.cluster.freeze import check_if_present

    frozen = check_if_present()
    if not frozen["frozen"]:
        raise ValueError("Freeze the reviewed source before creating execution manifests")
    body = {**body, "frozen": frozen}
    output_phase = "experiment_0" if body["profile_fold"] else body["phase"]
    target = paths.manifests_dir(output_phase) / "arrays" / f"{digest(body)}.json"
    write_json(target, body)
    logs = paths.logs_dir(output_phase)
    logs.mkdir(parents=True, exist_ok=True)
    route = body["route"]
    partition = (
        route["partition"]
        if body["gpu"]
        else ("batch_graniterapids" if route["cluster"] == "mindwell" else "batch")
    )
    command = [
        "sbatch",
        "--parsable",
        f"--clusters={route['cluster']}",
        f"--partition={partition}",
        f"--account={load_yaml('experiment_0/scheduling')['account']}",
        f"--array=0-{len(body['arms']) - 1}%{body['concurrency']}",
        f"--output={logs}/jevpfn-benchmark_%A_%a.log",
        f"--error={logs}/jevpfn-benchmark_%A_%a.log",
    ]
    if body["gpu"]:
        command.append("--gpus-per-node=1")
    command += [
        "scripts/slurm/benchmark_array.slurm",
        str(target),
        digest(body),
        body["model"],
        route["staging_root"],
        output_phase,
    ]
    return {
        "manifest": str(target),
        "array_tasks": len(body["arms"]),
        "excluded_arms": body["excluded_arms"],
        "submission_command": shlex.join(command),
        "submitted": False,
    }


def execute(path, expected, index):
    from pathlib import Path

    from src.cluster.freeze import check_if_present

    if not paths.on_vsc() or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Benchmark arrays require a compute allocation")
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    if digest(body) != expected or body["frozen"] != check_if_present():
        raise ValueError("Array manifest or frozen source changed")
    if os.environ.get("SLURM_CLUSTER_NAME") != body["route"]["cluster"]:
        raise ValueError("Array launched on a different cluster")
    if not 0 <= index < len(body["arms"]):
        raise ValueError("Invalid array item")
    return run_arm(body["arms"][index], allow_model_runs=True, profile=body["profile_fold"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase")
    parser.add_argument("--model")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--profile-fold", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--run-manifest")
    parser.add_argument("--sha256")
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.run_manifest:
        execute(args.run_manifest, args.sha256, args.index)
        return
    body = plan(
        args.phase,
        args.model,
        profile=args.profile,
        concurrency=args.concurrency,
        controls=args.controls,
        profile_fold=args.profile_fold,
    )
    print(
        json.dumps(
            write_plan(body)
            if args.write
            else {
                **{k: v for k, v in body.items() if k != "arms"},
                "array_tasks": len(body["arms"]),
                "submitted": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
