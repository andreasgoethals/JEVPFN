"""Owner hardware policy holds for submissions and direct inference entrypoints."""

import pytest

from src.cluster import scheduling


def test_default_gpu_plans_use_h100_and_cpu_jobs_stay_on_cpu():
    from src.cluster.benchmark_jobs import plan
    from src.cluster.experiment_zero import steps

    for step in steps("vsc"):
        command = step["command"]
        if step["name"] in {"cpu", "storage", "tokens"}:
            assert not any(flag.startswith("--partition=gpu") for flag in command)
        else:
            assert "--partition=gpu_h100" in command
    assert plan("experiment_1", "tabpfn_3_5")["route"]["partition"] == "gpu_h100"
    assert plan("experiment_1", "catboost")["gpu"] is False
    with pytest.raises(ValueError, match="Unknown GPU"):
        scheduling.gpu_profile("wice_a100")


@pytest.mark.parametrize("partition", ["gpu_a100", "gpu_a100_debug", "gpu", "interactive"])
def test_old_profiles_cannot_reintroduce_a100(monkeypatch, partition):
    monkeypatch.setattr(
        scheduling, "load_yaml", lambda _: {"profiles": {"legacy": {
            "cluster": "wice", "partition": partition,
            "staging_root": "/lustre1/project/stg_00211",
        }}}
    )
    with pytest.raises(ValueError, match="Unsupported cluster/partition"):
        scheduling.gpu_profile("legacy")


@pytest.mark.parametrize("partition", ["gpu_a100", "gpu_a100_debug", "gpu", "interactive"])
def test_a100_allocation_override_is_rejected(monkeypatch, partition):
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "wice")
    monkeypatch.setenv("SLURM_JOB_PARTITION", partition)
    monkeypatch.setenv("SLURM_JOB_GPUS", "0")
    with pytest.raises(RuntimeError, match="A100 allocations are disabled"):
        scheduling.reject_a100_allocation()


@pytest.mark.parametrize("cluster,partition,gpus", [
    ("wice", "gpu_h100", "0"), ("mindwell", "gpu_b200", "0"),
    ("wice", "interactive", ""), ("wice", "batch", ""), ("", "", ""),
])
def test_allowed_hardware_and_local_cpu_are_not_blocked(monkeypatch, cluster, partition, gpus):
    monkeypatch.setenv("SLURM_CLUSTER_NAME", cluster)
    monkeypatch.setenv("SLURM_JOB_PARTITION", partition)
    monkeypatch.setenv("SLURM_JOB_GPUS", gpus)
    scheduling.reject_a100_allocation()


def test_direct_inference_rejects_a100_before_any_model_loading(monkeypatch):
    from src.data.text_embeddings import E5Encoder
    from src.evaluation.models import estimator_factory
    from src.jev.self_hosted import SelfHostedClient
    from src.utils import paths

    monkeypatch.setenv("SLURM_CLUSTER_NAME", "wice")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "gpu_a100")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(paths, "on_vsc", lambda: True)
    for invoke in (
        lambda: E5Encoder({}),
        lambda: SelfHostedClient("cloudflare", allow_model_runs=True),
        lambda: estimator_factory("tabpfn_3_5", None, {"device": "cuda"}, allow_model_runs=True),
    ):
        with pytest.raises(RuntimeError, match="A100 allocations are disabled"):
            invoke()
