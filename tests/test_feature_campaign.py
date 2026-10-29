"""Full feature scheduling covers exactly the accepted open-weight scope."""

import pytest

from src.cluster.feature_campaign import plan, run
from src.cluster.job_graph import validate
from src.jev.catalog import SELF_HOSTED
from src.utils.config import load_yaml


def test_complete_open_weight_grid_and_workspace_lifetimes():
    steps = plan()
    settings = load_yaml("experiment_0/scheduling")
    settings["test_pool_limits"].update(dict.fromkeys(SELF_HOSTED, 2))
    validate(steps, settings)
    for provider in SELF_HOSTED:
        jobs = [s for s in steps if s.get("pool") == provider]
        assert len(jobs) == 18
        assert len({s["command"][-1] for s in jobs}) == 18
        assert all("--partition=gpu_b200" in s["command"] for s in jobs)
        release = next(s for s in steps if s["name"] == provider + "_release")
        assert {s["name"] for s in jobs} <= set(release["terminal_dependencies"])
    assert sum(s["name"].startswith("embeddings_") for s in steps) == 3
    assert not any(s["name"].startswith("jev_") for s in steps)


def test_campaign_refuses_local_execution_before_any_submission():
    with pytest.raises(RuntimeError, match="VSC"):
        run(execute=True)


def test_dataset_partitioning_cannot_shrink_the_acceptance_pilot():
    from src.jev.self_hosted import run

    with pytest.raises(ValueError, match="pilots cover all datasets"):
        run("liquid", dataset_id="MUL_TEXT_WINE_REVIEW")


def test_ambiguous_submission_retains_diagnostics_and_cannot_duplicate(isolated_output, monkeypatch):
    import json
    import subprocess

    from src.cluster import feature_campaign as campaign
    from src.data import text_embeddings
    from src.utils import paths

    settings = load_yaml("experiment_0/scheduling")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(paths, "on_vsc", lambda: True)
    monkeypatch.setattr(campaign, "check_if_present", lambda: {"frozen": True})
    monkeypatch.setattr(campaign, "runtime_pins", lambda: {})
    monkeypatch.setattr(campaign, "reviewed_acceptance", lambda *a, **k: {})
    monkeypatch.setattr(campaign, "require_idle_vsc", lambda: None)
    monkeypatch.setattr(text_embeddings, "encoder_config", lambda _: {"enabled": True})
    monkeypatch.setattr(campaign, "plan", lambda: [dict(name="fixture", cluster="wice", command=["sbatch"], depends_on=[])])
    monkeypatch.setattr(campaign, "load_yaml", lambda name: settings if name.endswith("scheduling") else
                        {"models": {"e5_small_v2": {}}, "full_feature_creation_enabled": True,
                         "accepted_before_extraction_manifests": dict.fromkeys(SELF_HOSTED, "reviewed.json")})
    calls = []

    def uncertain(command, **kwargs):
        calls.append(command)
        raise subprocess.TimeoutExpired(command, 60, stderr="scheduler timed out")

    monkeypatch.setattr(campaign.subprocess, "run", uncertain)
    with pytest.raises(subprocess.TimeoutExpired):
        campaign.run(execute=True)
    receipt = paths.manifests_dir("feature_creation") / "open_weight_campaign.json"
    step = json.loads(receipt.read_text())["steps"][0]
    assert step["status"] == "starting"
    assert step["submission_diagnostics"]["stderr"] == "scheduler timed out"
    with pytest.raises(ValueError, match="ambiguous"):
        campaign.run(execute=True)
    assert len(calls) == 1
