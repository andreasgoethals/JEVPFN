"""Owner hardware policy holds for submissions and direct inference entrypoints."""

import pytest

from src.cluster import scheduling


@pytest.mark.parametrize("variable", ["JEVPFN_PROJECT_ROOT", "VSC_PROJECT_ROOT"])
def test_project_override_reaches_every_submission_and_release(monkeypatch, variable):
    from src.cluster.experiment_zero import steps
    from src.cluster.setup_extractors import plan

    root = "/lustre1/project/test_allocation"
    monkeypatch.setenv(variable, root)
    monkeypatch.setenv("VSC_SCRATCH_GPFS1", "/gpfs1/scratch/example/user")
    flag = scheduling.project_export(root)
    final = steps("vsc")
    setup = plan(["liquid", "cloudflare"])
    assert all(flag in s["command"] for s in final)
    assert all(flag in s["flags"] for s in setup)
    for step in final:
        assert "/lustre1/project/stg_00211" not in " ".join(step["command"])
    assert next(s for s in setup if s["name"] == "cpu")["args"] == [root]
    assert next(s for s in setup if s["name"] == "liquid_stage")["args"] == ["liquid", root]
    profile = scheduling.gpu_profile(runtime="liquid")
    assert profile["durable_root"] == root
    assert profile["staging_root"].startswith("/gpfs1/scratch/")


@pytest.mark.parametrize("root", ["relative", "/lustre1/project/JEVPFN", "/lustre1/../data", "/lustre1/a,OTHER=x", "/lustre1/a\n"])
def test_project_export_rejects_ambiguous_roots(root):
    with pytest.raises(ValueError, match="allocated shared root"):
        scheduling.project_export(root)


def test_default_gpu_plans_use_b200_and_cpu_jobs_stay_on_cpu():
    from src.cluster.benchmark_jobs import plan
    from src.cluster.experiment_zero import steps

    for step in steps("vsc"):
        command = step["command"]
        if step["cluster"] == "wice" or step["name"].endswith("gpfs_storage"):
            assert not any(flag.startswith("--partition=gpu") for flag in command)
        else:
            assert "--partition=gpu_b200" in command
    by_name = {step["name"]: step for step in steps("vsc")}
    assert by_name["gpfs_storage"]["cluster"] == "mindwell"
    assert "--partition=batch_graniterapids" in by_name["gpfs_storage"]["command"]
    assert by_name["gpfs_storage"]["depends_on"] == ["cloudflare_scratch"]
    for step in by_name.values():
        if "--partition=gpu_b200" in step["command"]:
            assert any(d.endswith("gpfs_storage") for d in step["depends_on"])
    assert plan("experiment_1", "tabpfn_3_5")["route"]["partition"] == "gpu_b200"
    assert plan("experiment_1", "catboost")["gpu"] is False
    with pytest.raises(ValueError, match="Unknown GPU"):
        scheduling.gpu_profile("wice_a100")


def test_b200_runtime_workspaces_use_native_scratch_and_durable_lustre(monkeypatch):
    monkeypatch.setenv("VSC_SCRATCH_GPFS1", "/gpfs1/scratch/example/user")
    profile = scheduling.gpu_profile(runtime="intern")
    assert profile["staging_root"] == "/gpfs1/scratch/example/user/jevpfn-work/intern"
    assert profile["durable_root"] == "/lustre1/project/stg_00211"
    assert profile["runtime_profile"] == "blackwell-cu130"
    with pytest.raises(ValueError, match="Invalid runtime"):
        scheduling.gpu_profile(runtime="../../another_project")
    monkeypatch.setenv("VSC_SCRATCH_GPFS1", "/lustre1/scratch/example")
    with pytest.raises(ValueError, match="native GPFS"):
        scheduling.gpu_profile(runtime="intern")


@pytest.mark.parametrize("partition", ["gpu_a100", "gpu_a100_debug", "gpu", "interactive"])
def test_old_profiles_cannot_reintroduce_a100(monkeypatch, partition):
    monkeypatch.setattr(
        scheduling,
        "load_yaml",
        lambda _: {
            "profiles": {
                "legacy": {
                    "cluster": "wice",
                    "partition": partition,
                    "staging_root": "/lustre1/project/stg_00211",
                }
            }
        },
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


@pytest.mark.parametrize(
    "cluster,partition,gpus",
    [
        ("wice", "gpu_h100", "0"),
        ("mindwell", "gpu_b200", "0"),
        ("wice", "interactive", ""),
        ("wice", "batch", ""),
        ("", "", ""),
    ],
)
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
