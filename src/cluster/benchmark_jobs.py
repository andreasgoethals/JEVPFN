"""Write bounded, bundled Slurm array plans; never submit jobs or enable experiments.

One array task runs every supported arm of one dataset for one predictor and phase
(each arm: five folds, resumed by the evaluation store). Inputs are loaded once per
process; every arm and fit is evaluated exactly as a single-arm task would evaluate
it. A task may run several worker processes concurrently on one B200 or on disjoint
CPU cores; each worker keeps the reviewed thread count, so settings identities and
resume are unchanged. Concurrent GPU workers are recorded in each fold's resources.
CPU-only predictors (CatBoost, linear/logistic regression) request no GPU.
Smaller datasets are ordered first. One profile per manifest prevents speculative
duplicate submissions on several clusters.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys

from src.cluster import benchmark_sizing as sizing
from src.evaluation.campaign import benchmark_plan, run_arm, task_cache
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.serialization import digest, write_json

CPU_ONLY_MODELS = ("catboost", "linear")
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")


def _dataset_rows():
    from src.data.loaders import load_catalog

    rows = {}
    for entry in load_catalog()["datasets"]:
        stats = entry.get("paper_summary") or {}
        rows[entry["id"]] = (int(stats.get("rows") or 0),
                             int(stats.get("text_columns") or 0) + int(stats.get("non_text_columns") or 0))
    return rows


def plan(phase, model, *, profile=None, concurrency=8, profile_fold=False, workers=None,
         profile_summary=None):
    from src.cluster.scheduling import gpu_profile

    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("Positive array concurrency required")
    runtime = load_yaml("experiment_1/models")["models"][model]["runtime"]
    gpu = runtime["device"].startswith("cuda")
    if model in CPU_ONLY_MODELS and gpu:
        raise ValueError(f"{model} is a CPU-node predictor; its runtime must not request CUDA")
    threads = int(runtime.get("threads", 8))
    route = gpu_profile(profile, runtime=model)
    if not gpu and route.get("storage_mode") == "scratch_mirror":
        # CPU predictors run natively against PROJECT; they have no GPU scratch
        # staging dependency and must not receive an uncreated GPU working set.
        route = {**route, "staging_root": route["durable_root"], "partition": "batch_graniterapids"}
        route.pop("storage_mode")
    candidates = [a for a in benchmark_plan() if a["phase"] == phase and a["model"] == model]
    arms = [a for a in candidates if a["width_status"] != "unsupported_width"]
    if profile_fold:
        from src.evaluation.qualification import profile_plan

        arms = profile_plan(arms)
    excluded = [a for a in candidates if a["width_status"] == "unsupported_width"]
    if not arms:
        raise ValueError("No supported arms for this phase/model")
    if profile_summary is None and not profile_fold and paths.on_vsc():
        from src.evaluation.qualification import PHASE_CONFIGS

        acceptance = load_yaml(PHASE_CONFIGS[phase]).get("experiment_0_acceptance")
        profile_summary = sizing.summarize(sizing.profile_records(
            model, phase=phase, cluster=route["cluster"], acceptance=acceptance,
        ))
    sizes = _dataset_rows()
    groups = {}
    for index, arm in enumerate(arms):
        groups.setdefault(arm.get("dataset_id"), []).append(index)
    tasks = []
    for dataset_id, indices in groups.items():
        rows, width = sizes.get(dataset_id, (0, 0))
        count = sizing.workers(
            profile_summary, gpu=gpu, threads=threads, arms=len(indices),
            requested=1 if profile_fold or (gpu and route["cluster"] != "mindwell") else workers,
        )
        tasks.append({
            "dataset_id": dataset_id,
            "arms": indices,
            "workers": count,
            "resources": sizing.resources(profile_summary, gpu=gpu, threads=threads,
                                          workers_per_task=count, arm_rows=[rows] * len(indices)),
            "size": [rows, width],
        })
    # Small and cheap first: failures surface before the expensive tasks start.
    tasks.sort(key=lambda t: (t["size"][0] * max(1, t["size"][1]), str(t["dataset_id"])))
    return {
        "version": 2,
        "phase": phase,
        "model": model,
        "route": route,
        "concurrency": concurrency,
        "gpu": gpu,
        "threads_per_worker": threads,
        "arms": arms,
        "tasks": tasks,
        "excluded_arms": excluded,
        "profile_fold": profile_fold,
        "sizing": profile_summary,
    }


def _tiers(tasks):
    """Tasks with an identical Slurm request share one array, smallest requests first."""
    tiers = {}
    for task in tasks:
        r = task["resources"]
        tiers.setdefault((r["time_hours"], r["cpus"], r["mem_mib"] or 0), []).append(task)
    return [tiers[key] for key in sorted(tiers)]


def write_plan(body):
    from src.cluster.freeze import check_if_present
    from src.cluster.scheduling import project_export

    frozen = check_if_present()
    if not frozen["frozen"]:
        raise ValueError("Freeze the reviewed source before creating execution manifests")
    output_phase = "experiment_0" if body["profile_fold"] else body["phase"]
    logs = paths.logs_dir(output_phase)
    logs.mkdir(parents=True, exist_ok=True)
    route = body["route"]
    partition = (
        route["partition"]
        if body["gpu"]
        else ("batch_graniterapids" if route["cluster"] == "mindwell" else "batch")
    )
    submissions = []
    for tier in _tiers(body["tasks"]):
        manifest = {**body, "tasks": tier, "frozen": frozen}
        target = paths.manifests_dir(output_phase) / "arrays" / f"{digest(manifest)}.json"
        write_json(target, manifest)
        request = tier[0]["resources"]
        cpus = request["cpus"]
        if route["cluster"] == "mindwell" and request["mem_mib"]:
            mib_per_cpu = sizing.MINDWELL_GPU_MIB_PER_CORE if body["gpu"] else sizing.MINDWELL_CPU_MIB_PER_CORE
            cpus = max(cpus, (request["mem_mib"] + mib_per_cpu - 1) // mib_per_cpu)
        command = [
            "sbatch",
            "--parsable",
            project_export(route.get("durable_root", route["staging_root"])),
            f"--clusters={route['cluster']}",
            f"--partition={partition}",
            f"--account={load_yaml('experiment_0/scheduling')['account']}",
            f"--array=0-{len(tier) - 1}%{body['concurrency']}",
            f"--cpus-per-task={cpus}",
            f"--time={request['time_hours']}:00:00",
            f"--output={logs}/jevpfn-benchmark_%A_%a.log",
            f"--error={logs}/jevpfn-benchmark_%A_%a.log",
        ]
        if request["mem_mib"] and route["cluster"] != "mindwell":
            command.append(f"--mem={request['mem_mib']}M")
        if body["gpu"]:
            command.append("--gpus-per-node=1")
        command += [
            "scripts/slurm/benchmark_array.slurm",
            str(target),
            digest(manifest),
            body["model"],
            route["staging_root"],
            output_phase,
        ]
        submissions.append({
            "manifest": str(target),
            "array_tasks": len(tier),
            "arms": sum(len(t["arms"]) for t in tier),
            "workers_per_task": sorted({t["workers"] for t in tier}),
            "resources": {**request, "cpus": cpus},
            "submission_command": shlex.join(command),
        })
    return {"submissions": submissions, "excluded_arms": body["excluded_arms"], "submitted": False}


def array_item(path, expected, index):
    from pathlib import Path

    from src.cluster.freeze import check_if_present

    if not paths.on_vsc() or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Benchmark arrays require a compute allocation")
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    if digest(body) != expected or body["frozen"] != check_if_present():
        raise ValueError("Array manifest or frozen source changed")
    if os.environ.get("SLURM_CLUSTER_NAME") != body["route"]["cluster"]:
        raise ValueError("Array launched on a different cluster")
    if body.get("version") != 2 or not 0 <= index < len(body["tasks"]):
        raise ValueError("Invalid array item")
    task = body["tasks"][index]
    return body, task


def worker_environment(base, *, threads, workers, worker, gpu):
    """Each worker sees exactly the reviewed thread count, as a single-arm task did."""
    env = {**base, "SLURM_CPUS_PER_TASK": str(threads), "JEVPFN_WORKER_INDEX": str(worker),
           "JEVPFN_WORKERS_PER_DEVICE": str(workers if gpu else 1)}
    env.update({name: str(threads) for name in THREAD_VARIABLES})
    return env


def _pin_cores(threads, worker):
    """Disjoint cores per worker inside the allocation (Linux); placement only."""
    if not hasattr(os, "sched_getaffinity"):
        return
    allowed = sorted(os.sched_getaffinity(0))
    cores = allowed[worker * threads:(worker + 1) * threads]
    if len(cores) != threads:
        raise RuntimeError("Benchmark worker has too few allocated CPU cores")
    os.sched_setaffinity(0, cores)


def run_arms(body, task, worker=0):
    """Run this worker's share of the task's arms; continue past a failed arm."""
    workers = task["workers"]
    mine = [body["arms"][i] for i in task["arms"][worker::workers]]
    failures = []
    with task_cache():
        for arm in mine:
            try:
                run_arm(arm, allow_model_runs=True, profile=body["profile_fold"])
            except Exception as exc:  # run_arm already wrote the failure manifest.
                failures.append({"arm": arm, "error_type": type(exc).__name__})
                print(json.dumps({"event": "arm_failed", "error_type": type(exc).__name__,
                                  "dataset_id": arm.get("dataset_id"),
                                  "feature_set": arm.get("feature_set"),
                                  "provider": arm.get("provider")}), flush=True)
    return failures


def execute(path, expected, index, worker=None):
    body, task = array_item(path, expected, index)
    threads = body["threads_per_worker"]
    if worker is not None:
        _pin_cores(threads, worker)
        if run_arms(body, task, worker):
            raise SystemExit(1)
        return
    workers = task["workers"]
    if hasattr(os, "sched_getaffinity") and len(os.sched_getaffinity(0)) < threads * workers:
        raise RuntimeError("Benchmark task has too few allocated CPU cores for its worker plan")
    if workers == 1:
        _pin_cores(threads, 0)
        os.environ.update(worker_environment({}, threads=threads, workers=1, worker=0, gpu=body["gpu"]))
        if run_arms(body, task):
            raise SystemExit(1)
        return
    processes = [
        subprocess.Popen(
            [sys.executable, "-m", "src.cluster.benchmark_jobs", "--run-manifest", str(path),
             "--sha256", expected, "--index", str(index), "--worker", str(w)],
            env=worker_environment(dict(os.environ), threads=threads, workers=workers, worker=w,
                                   gpu=body["gpu"]),
        )
        for w in range(workers)
    ]
    codes = [p.wait() for p in processes]
    if any(codes):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase")
    parser.add_argument("--model")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--workers", type=int, default=None,
                        help="Cap workers per task below the resource-based limit")
    parser.add_argument("--profile-fold", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--run-manifest")
    parser.add_argument("--sha256")
    parser.add_argument("--index", type=int)
    parser.add_argument("--worker", type=int, default=None)
    parser.add_argument("--dataset-for-index", action="store_true")
    args = parser.parse_args()
    if args.dataset_for_index:
        _, task = array_item(args.run_manifest, args.sha256, args.index)
        print(task["dataset_id"])
        return
    if args.run_manifest:
        execute(args.run_manifest, args.sha256, args.index, worker=args.worker)
        return
    body = plan(
        args.phase,
        args.model,
        profile=args.profile,
        concurrency=args.concurrency,
        profile_fold=args.profile_fold,
        workers=args.workers,
    )
    print(
        json.dumps(
            write_plan(body)
            if args.write
            else {
                **{k: v for k, v in body.items() if k not in ("arms", "tasks")},
                "array_tasks": len(body["tasks"]),
                "workers_per_task": sorted({t["workers"] for t in body["tasks"]}),
                "submitted": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
