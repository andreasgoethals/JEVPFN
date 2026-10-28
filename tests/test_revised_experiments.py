from collections import Counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.evaluation.campaign import benchmark_plan
from src.evaluation.embedding_control import EmbeddingControl
from src.evaluation.widths import pca_width_table


def test_primary_grid_and_widths():
    plan = benchmark_plan()
    assert Counter(a["phase"] for a in plan) == {
        "experiment_1": 3996,
        "experiment_2": 2268,
        "experiment_3": 54,
    }
    assert all(a["provider"] is None for a in plan if a["phase"] != "experiment_1")
    assert all(a["width_status"] == "within_known_width_limit" for a in plan)
    table = pca_width_table()
    assert table.per_column_32.max() == 431
    assert table.joint_256.max() == 300
    assert table.max_with_one_provider_both.max() == 445
    assert table.conservative_kumo_upper_bound.max() == 447


def test_selected_followups_exist_only_in_exploratory_phase(monkeypatch, tmp_path):
    from src.evaluation import design
    from src.utils.serialization import file_sha256

    evidence = tmp_path / "reviewed.csv"
    evidence.write_text("reviewed summary", encoding="utf-8")
    original = design.load_yaml
    selection = dict(
        provider="jev",
        mode="joint",
        evidence=str(evidence),
        evidence_sha256=file_sha256(evidence),
        rationale="Reviewed mean ranks",
        embedding={"encoder": "e5_small_v2", "scope": "joint", "components": 32},
        inference_status="exploratory_post_selection",
    )
    monkeypatch.setattr(
        design,
        "load_yaml",
        lambda name: selection if name.endswith("followup_selection") else original(name),
    )
    arms = [
        a for a in benchmark_plan(include_controls=True) if a.get("control") == "winner_combination"
    ]
    assert len(arms) == 108 + 54
    assert all(a["phase"] == "experiment_5" and a["provider"] == "jev" for a in arms)
    assert all(a["inference_status"] == "exploratory_post_selection" for a in arms)
    evidence.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        design.followup_selection()


@pytest.mark.parametrize("scope", ["joint", "per_column"])
def test_cpu_prepared_pca_is_identical_and_never_refits_on_gpu(tmp_path, monkeypatch, scope):
    from src.evaluation import embedding_control
    from src.evaluation.pca_cache import identity

    rng = np.random.default_rng(9)
    x = pd.DataFrame(
        rng.normal(size=(28, 8)), columns=list("abcdefgh"), index=[f"csv:{i}" for i in range(28)]
    )
    x.loc["csv:0", :] = np.nan
    x.loc["csv:1", list("abcd")] = np.nan
    train, test = x.iloc[:20], x.iloc[20:]
    groups = [list("abcd"), list("efgh")]
    path = tmp_path / "pca.sqlite3"
    args = dict(components=3, scope=scope)
    direct = EmbeddingControl(SimpleNamespace(fit=lambda x, y: None), groups, **args).fit(
        train, None
    )
    EmbeddingControl(
        None, groups, **args, prepared_cache=path, embedding_artifacts=["pinned"], prepare=True
    ).fit(train, None)
    monkeypatch.setattr(
        embedding_control.PCA, "fit", lambda *a, **k: pytest.fail("Must reuse CPU PCA")
    )
    cached = EmbeddingControl(
        SimpleNamespace(fit=lambda x, y: None),
        groups,
        **args,
        prepared_cache=path,
        embedding_artifacts=["pinned"],
    ).fit(train, None)
    np.testing.assert_allclose(cached.transform(test), direct.transform(test), equal_nan=True)
    assert list(cached.transform(train)) == list(cached.transform(test))
    with pytest.raises(ValueError, match="CPU-prepared"):
        EmbeddingControl(
            None, groups, **args, prepared_cache=path, embedding_artifacts=["pinned"]
        ).fit(x.iloc[1:21], None)
    assert identity(train, groups[0], 3, scope, 1, ["pinned"]) != identity(
        train, groups[0], 3, scope, 1, ["other"]
    )


def test_training_partitions_exclude_outer_test_rows():
    from src.cluster.predictor_contracts import fixtures
    from src.evaluation.prepare_pca import training_partitions
    from src.evaluation.protocol import shared_splits

    for dataset in fixtures():
        splits = shared_splits(dataset)
        parts = list(training_partitions(dataset, splits))
        per_fold = 4 if dataset.metadata.task_type == "binary_classification" else 1
        assert len(parts) == 5 * per_fold
        for i, fold in enumerate(splits["folds"]):
            for train in parts[i * per_fold : (i + 1) * per_fold]:
                assert set(train) <= set(fold["train"])
                assert not set(train) & set(fold["test"])


def test_gpu_routes_use_native_scratch_and_cpu_plan_has_no_gpu():
    from src.cluster.benchmark_jobs import plan
    from src.cluster.scheduling import gpu_profile

    assert gpu_profile("mindwell_b200")["storage_mode"] == "scratch_mirror"
    assert gpu_profile("wice_h100")["partition"] == "gpu_h100"
    assert plan("experiment_1", "catboost")["gpu"] is False
    assert plan("experiment_1", "tabpfn_3_5")["gpu"] is True


def test_gpu_sampling_does_not_claim_unallocated_devices(monkeypatch):
    from src.cluster import monitor

    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)
    monkeypatch.setattr(monitor.subprocess, "run", lambda *a, **k: pytest.fail("No GPU allocation"))
    assert monitor.gpu_sample() == []


def test_gpu_sampling_uses_slurm_device_zero(monkeypatch):
    from src.cluster import monitor

    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)
    monkeypatch.setenv("SLURM_JOB_GPUS", "0")

    def query(command, **kwargs):
        assert command[:3] == ["nvidia-smi", "-i", "0"]
        return SimpleNamespace(stdout="uuid, H100, 80, 20, 500, 80000, 190\n")

    monkeypatch.setattr(monitor.subprocess, "run", query)
    assert monitor.gpu_sample()[0][2] == "80"


def test_gpu_sampling_uses_the_captured_host_binary_and_allocated_device(monkeypatch):
    from src.cluster import monitor

    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)
    monkeypatch.setenv("SLURM_JOB_GPUS", "3")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("JEVPFN_NVIDIA_SMI", "/host/bin/nvidia-smi")

    def query(command, **kwargs):
        assert command[:3] == ["/host/bin/nvidia-smi", "-i", "3"]
        return SimpleNamespace(stdout="allocated-uuid, H100, 80, 20, 500, 80000, 190\n")

    monkeypatch.setattr(monitor.subprocess, "run", query)
    assert monitor.gpu_sample()[0][0] == "allocated-uuid"


def test_telemetry_failure_preserves_diagnostics_without_changing_model_result(
    monkeypatch, tmp_path
):
    import json
    import subprocess
    import sys

    from src.cluster import monitor

    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    monkeypatch.setenv("SLURM_JOB_GPUS", "3")
    monkeypatch.setattr(monitor.paths, "on_vsc", lambda: True)
    monkeypatch.setattr(monitor.paths, "logs_dir", lambda phase: tmp_path / "logs")
    monkeypatch.setattr(monitor.paths, "manifests_dir", lambda phase: tmp_path / "manifests")

    def failed_query():
        raise subprocess.CalledProcessError(
            6, ["nvidia-smi", "-i", "3"], output="fixture stdout", stderr="fixture NVML error"
        )

    monkeypatch.setattr(monitor, "gpu_sample", failed_query)
    assert monitor.run([sys.executable, "-c", "pass"], interval=1) == 0
    report = json.loads(next((tmp_path / "manifests").rglob("*.json")).read_text())
    assert report["sampling_status"] == "failed"
    assert report["gpu_samples"] == 0 and report["mean_gpu_util_percent"] is None
    assert report["sampling_error_details"] == [
        {
            "type": "CalledProcessError",
            "returncode": 6,
            "command": ["nvidia-smi", "-i", "3"],
            "stdout": "fixture stdout",
            "stderr": "fixture NVML error",
        }
    ]
    assert report["gpu_allocation"]["SLURM_JOB_GPUS"] == "3"


def test_telemetry_preserves_child_failure_and_uses_separate_small_outputs(monkeypatch, tmp_path):
    import gzip
    import json
    import sys

    from src.cluster import monitor

    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    monkeypatch.setattr(monitor.paths, "on_vsc", lambda: True)
    monkeypatch.setattr(monitor.paths, "logs_dir", lambda phase: tmp_path / "logs")
    monkeypatch.setattr(monitor.paths, "manifests_dir", lambda phase: tmp_path / "manifests")
    monkeypatch.setattr(monitor, "gpu_sample", lambda: [])
    assert monitor.run([sys.executable, "-c", "raise SystemExit(7)"], interval=1) == 7
    receipt = json.loads(next((tmp_path / "manifests").rglob("*.json")).read_text())
    assert receipt["exit_code"] == 7
    assert receipt["gpu_samples"] == 0
    assert receipt["mean_gpu_util_percent"] is None
    with gzip.open(receipt["samples_file"], "rt") as stream:
        assert "gpu_uuid" in stream.read()


def test_array_plan_reports_width_exclusions(monkeypatch):
    from src.cluster import benchmark_jobs

    valid = dict(phase="experiment_1", model="catboost", width_status="within_known_width_limit")
    excluded = {**valid, "width_status": "unsupported_width", "predictor_columns": 501}
    monkeypatch.setattr(benchmark_jobs, "benchmark_plan", lambda **kw: [valid, excluded])
    result = benchmark_jobs.plan("experiment_1", "catboost")
    assert result["arms"] == [valid]
    assert result["excluded_arms"] == [excluded]
