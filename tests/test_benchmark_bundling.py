"""Bundled benchmark tasks change scheduling only, never what an arm evaluates."""

import pandas as pd
import pytest

from src.cluster import benchmark_jobs, benchmark_sizing


def arm(dataset, feature_set="non_text", provider=None):
    return dict(phase="experiment_1", model="catboost", dataset_id=dataset, feature_set=feature_set,
                provider=provider, width_status="within_known_width_limit")


@pytest.fixture
def two_datasets(monkeypatch):
    arms = [arm("big"), arm("small"), arm("big", "non_text_plus_joint", "intern"),
            arm("small", "non_text_plus_joint", "intern")]
    linear = [{**a, "model": "linear"} for a in arms]
    monkeypatch.setattr(benchmark_jobs, "benchmark_plan", lambda **kw: arms + linear)
    monkeypatch.setattr(benchmark_jobs, "_dataset_rows", lambda: {"big": (100_000, 10), "small": (500, 3)})
    return arms


def test_one_task_per_dataset_with_smaller_datasets_first(two_datasets):
    body = benchmark_jobs.plan("experiment_1", "catboost")
    assert body["arms"] == two_datasets  # Arm definitions are unchanged.
    assert [t["dataset_id"] for t in body["tasks"]] == ["small", "big"]
    assert sorted(i for t in body["tasks"] for i in t["arms"]) == [0, 1, 2, 3]
    assert all(len({body["arms"][i]["dataset_id"] for i in t["arms"]}) == 1 for t in body["tasks"])


def test_cpu_predictors_stay_on_cpu_nodes_with_packed_eight_thread_workers(two_datasets):
    for model in ("catboost", "linear"):
        body = benchmark_jobs.plan("experiment_1", model)
        assert body["gpu"] is False and body["route"]["partition"] == "batch_graniterapids"
        assert body["threads_per_worker"] == 8
        for task in body["tasks"]:
            assert task["workers"] == 2  # Capped by the task's two arms.
            assert task["resources"]["cpus"] == 16


def test_linear_covers_logistic_regression_on_cpu():
    from src.utils.config import load_yaml

    runtime = load_yaml("experiment_1/models")["models"]["linear"]["runtime"]
    assert runtime["device"] == "cpu"
    assert set(benchmark_jobs.CPU_ONLY_MODELS) == {"catboost", "linear"}


def test_unmeasured_gpu_work_is_never_shared():
    assert benchmark_sizing.workers(None, gpu=True, threads=8, arms=37) == 1


def test_measured_gpu_sharing_respects_vram_cores_and_host_memory():
    summary = dict(peak_gpu_reserved_bytes=20 * 1024**3, peak_rss_bytes=10 * 1024**3,
                   seconds_per_training_row=0.001, max_fold_seconds=100, profiles=4)
    # VRAM alone would allow 8; the 24-core B200 ceiling with 8 threads allows 3.
    assert benchmark_sizing.workers(summary, gpu=True, threads=8, arms=37) == 3
    heavy = {**summary, "peak_rss_bytes": 90 * 1024**3}
    assert benchmark_sizing.workers(heavy, gpu=True, threads=8, arms=37) == 1
    big = {**summary, "peak_gpu_reserved_bytes": 100 * 1024**3}
    assert benchmark_sizing.workers(big, gpu=True, threads=8, arms=37) == 1


def test_resources_bucket_time_and_scale_memory_with_workers():
    summary = dict(peak_gpu_reserved_bytes=0, peak_rss_bytes=4 * 1024**3,
                   seconds_per_training_row=0.01, max_fold_seconds=100, profiles=2)
    small = benchmark_sizing.resources(summary, gpu=False, threads=8, workers_per_task=4,
                                       arm_rows=[1000] * 8)
    assert small["cpus"] == 32 and small["time_hours"] == 2
    assert small["mem_mib"] >= 4 * 4 * 1024
    unprofiled = benchmark_sizing.resources(None, gpu=True, threads=8, workers_per_task=1,
                                            arm_rows=[1000])
    assert unprofiled == {"cpus": 8, "time_hours": 48, "mem_mib": None}


def test_profiles_from_shared_devices_never_set_time_rates():
    shared = dict(dataset_id="d", n_train=1000, wall_seconds=50.0, peak_rss_bytes=1,
                  peak_gpu_reserved_bytes=1, device_workers=3)
    summary = benchmark_sizing.summarize([shared])
    assert summary["seconds_per_training_row"] is None
    assert benchmark_sizing.estimated_arm_seconds(summary, 1000) is None


def test_workers_keep_the_reviewed_thread_count_and_record_sharing():
    env = benchmark_jobs.worker_environment({"SLURM_CPUS_PER_TASK": "24"}, threads=8, workers=3,
                                            worker=1, gpu=True)
    assert env["SLURM_CPUS_PER_TASK"] == "8" and env["OMP_NUM_THREADS"] == "8"
    assert env["JEVPFN_WORKERS_PER_DEVICE"] == "3" and env["JEVPFN_WORKER_INDEX"] == "1"
    cpu = benchmark_jobs.worker_environment({}, threads=8, workers=3, worker=0, gpu=False)
    assert cpu["JEVPFN_WORKERS_PER_DEVICE"] == "1"  # Disjoint pinned cores, no shared device.


def test_a_failed_arm_does_not_stop_the_rest_of_the_task(monkeypatch):
    seen = []

    def fake(arm, **kwargs):
        seen.append(arm["feature_set"])
        if arm["feature_set"] == "bad":
            raise RuntimeError("fit failed")

    monkeypatch.setattr(benchmark_jobs, "run_arm", fake)
    body = {"arms": [arm("d", "bad"), arm("d", "good"), arm("d", "also")], "profile_fold": False}
    task = {"arms": [0, 1, 2], "workers": 1}
    failures = benchmark_jobs.run_arms(body, task)
    assert seen == ["bad", "good", "also"] and len(failures) == 1


def test_workers_split_arms_disjointly(monkeypatch):
    body = {"arms": [arm("d", str(i)) for i in range(7)], "profile_fold": False}
    task = {"arms": list(range(7)), "workers": 3}
    shares = []
    for worker in range(3):
        seen = []
        monkeypatch.setattr(benchmark_jobs, "run_arm", lambda a, seen=seen, **k: seen.append(a["feature_set"]))
        benchmark_jobs.run_arms(body, task, worker)
        shares.append(seen)
    assert sorted(x for s in shares for x in s) == [str(i) for i in range(7)]
    assert not set(shares[0]) & set(shares[1])


def test_multi_worker_task_launches_one_process_per_worker(monkeypatch):
    body = {"arms": [arm("d")] * 4, "profile_fold": False, "threads_per_worker": 8, "gpu": True}
    task = {"arms": [0, 1, 2, 3], "workers": 2, "dataset_id": "d"}
    monkeypatch.setattr(benchmark_jobs, "array_item", lambda *a: (body, task))
    launched = []

    class Process:
        def __init__(self, command, env):
            launched.append((command, env))

        def wait(self):
            return 0

    monkeypatch.setattr(benchmark_jobs.subprocess, "Popen", Process)
    benchmark_jobs.execute("manifest.json", "sha", 0)
    assert [c[c.index("--worker") + 1] for c, _ in launched] == ["0", "1"]
    assert all(env["JEVPFN_WORKERS_PER_DEVICE"] == "2" for _, env in launched)


def test_task_cache_hands_each_arm_an_independent_copy():
    from src.evaluation import campaign

    calls = []
    with campaign.task_cache():
        first = campaign._cached(("x",), lambda: calls.append(1) or {"rows": [1]})
        first["rows"].append(2)  # One arm mutating its input ...
        second = campaign._cached(("x",), lambda: calls.append(1) or {"rows": [1]})
    assert second == {"rows": [1]} and calls == [1]  # ... never reaches the next arm.
    # Outside a bundled task nothing is cached.
    campaign._cached(("x",), lambda: calls.append(1))
    campaign._cached(("x",), lambda: calls.append(1))
    assert calls == [1, 1, 1]


def test_verified_hash_is_reused_only_while_the_file_is_unchanged(tmp_path, monkeypatch):
    from src.evaluation import checkpoint_paths

    path = tmp_path / "weights.bin"
    path.write_bytes(b"a")
    monkeypatch.setattr(checkpoint_paths, "_VERIFIED", {})
    calls = []
    real = checkpoint_paths.file_sha256
    monkeypatch.setattr(checkpoint_paths, "file_sha256", lambda p: calls.append(p) or real(p))
    first = checkpoint_paths.verified_sha256(path)
    assert checkpoint_paths.verified_sha256(path) == first and len(calls) == 1
    path.write_bytes(b"bb")
    assert checkpoint_paths.verified_sha256(path) != first and len(calls) == 2


def test_device_workers_recorded_and_validated(monkeypatch):
    from src.evaluation import records

    monkeypatch.delenv("JEVPFN_WORKERS_PER_DEVICE", raising=False)
    assert records.device_workers() == 1
    monkeypatch.setenv("JEVPFN_WORKERS_PER_DEVICE", "3")
    assert records.device_workers() == 3
    monkeypatch.setenv("JEVPFN_WORKERS_PER_DEVICE", "0")
    with pytest.raises(ValueError):
        records.device_workers()


def test_burden_table_excludes_timings_from_shared_devices():
    from src.evaluation.notebook_study import IDENTITY, burden_table

    base = {k: "x" for k in IDENTITY}
    rows = []
    for arm_id, shared in (("exclusive", 1), ("shared", 2)):
        for fold in range(5):
            ident = {**base, "arm_id": arm_id, "phase": "experiment_1", "outer_fold": fold}
            rows += [{**ident, "metric": "seconds.fit", "value": 10.0},
                     {**ident, "metric": "seconds.predict", "value": 1.0},
                     {**ident, "metric": "columns.representation", "value": 5},
                     {**ident, "metric": "resources.device_workers", "value": shared}]
    table = burden_table(pd.DataFrame(rows), ["experiment_1"])
    assert list(table.arm_id) == ["exclusive", "shared"]
    shared_row = table[table.arm_id.eq("shared")].iloc[0]
    assert pd.isna(shared_row.get("seconds.fit")) and shared_row["columns.representation"] == 5
