"""Full feature scheduling covers exactly the accepted open-weight scope."""

from pathlib import Path

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
    with pytest.raises(ValueError, match="Unresolved ambiguous"):
        campaign.run(execute=True, recover=True, reviewed_by="fixture reviewer")
    assert len(calls) == 1


@pytest.mark.parametrize("stdout,stderr,expected", [
    ("", "sbatch: error: QOSMaxSubmitJobPerUserLimit", True),
    ("123;mindwell", "QOSMaxSubmitJobPerUserLimit", False),
    ("", "Socket timed out on send/recv operation", False),
    ("", "Job violates accounting/QOS policy", False),
])
def test_only_definite_admission_refusals_can_be_retried(stdout, stderr, expected):
    from src.cluster.feature_campaign import capacity_rejection

    assert capacity_rejection(stdout, stderr) is expected


def test_full_account_defers_admission_without_abandoning_the_graph():
    from src.cluster.job_graph import SubmissionDeferred, advance

    steps = [dict(name=name, cluster="mindwell", pool="gpu") for name in ("first", "second")]
    settings = load_yaml("experiment_0/scheduling")
    calls = []

    def submit(step):
        calls.append(step["name"])
        step["status"] = "pending"
        raise SubmissionDeferred("definite rejection")

    result = advance(steps, settings=settings, state=lambda *a, **k: None,
                     submit=submit, save=lambda: None)
    assert calls == ["first"]
    assert result["counts"] == {"pending": 2}
    assert result["all_terminal"] is False


def test_recovery_reuses_completed_jobs_and_retries_only_terminal_failures(isolated_output, monkeypatch):
    import json
    import subprocess

    from src.cluster import feature_campaign as campaign
    from src.data import text_embeddings
    from src.utils import paths
    from src.utils.serialization import write_json

    settings = load_yaml("experiment_0/scheduling")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(paths, "on_vsc", lambda: True)
    monkeypatch.setattr(campaign, "check_if_present", lambda: {"frozen": True})
    monkeypatch.setattr(campaign, "runtime_pins", lambda: {})
    monkeypatch.setattr(campaign, "reviewed_acceptance", lambda *a, **k: {})
    monkeypatch.setattr(campaign, "require_idle_vsc", lambda: None)
    monkeypatch.setattr(text_embeddings, "encoder_config", lambda _: {"enabled": True})
    base = [dict(name=n, cluster="wice", command=["sbatch", n], depends_on=[])
            for n in ("done", "failed", "refused")]
    monkeypatch.setattr(campaign, "plan", lambda: [dict(s) for s in base])
    monkeypatch.setattr(campaign, "load_yaml", lambda name: settings if name.endswith("scheduling") else
                        {"models": {"e5_small_v2": {}}, "full_feature_creation_enabled": True,
                         "accepted_before_extraction_manifests": dict.fromkeys(SELF_HOSTED, "reviewed.json")})
    old = [dict(s) for s in base]
    old[0].update(job_id="1", status="submitted")
    old[1].update(job_id="2", status="submitted")
    old[2].update(status="starting", submission_diagnostics={
        "error_type": "CalledProcessError", "stdout": "", "stderr": "QOSMaxSubmitJobPerUserLimit"})
    receipt = paths.manifests_dir("feature_creation") / "open_weight_campaign.json"
    write_json(receipt, dict(identity="old", steps=old))
    monkeypatch.setattr(campaign, "scheduler_state", lambda job, **k:
                        ("COMPLETED", "0:0") if job == "1" else ("FAILED", "1:0"))
    calls = []

    def submit(command, **kwargs):
        calls.append(command[-1])
        return subprocess.CompletedProcess(command, 0, stdout=str(100 + len(calls)), stderr="")

    monkeypatch.setattr(campaign.subprocess, "run", submit)
    campaign.run(execute=True, recover=True, reviewed_by="fixture reviewer")
    saved = json.loads(receipt.read_text())
    assert calls == ["failed", "refused"]
    assert saved["steps"][0]["job_id"] == "1"
    assert saved["steps"][1]["previous_attempt"]["job_id"] == "2"
    history = json.loads(Path(saved["recovery"]["previous_receipt"]).read_text())
    assert history["identity"] == "old"


def test_named_admission_ceiling_leaves_campaign_pending(isolated_output, monkeypatch):
    import subprocess

    from src.cluster import feature_campaign as campaign
    from src.data import text_embeddings

    settings = load_yaml("experiment_0/scheduling")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(campaign.paths, "on_vsc", lambda: True)
    monkeypatch.setattr(campaign, "check_if_present", lambda: {"frozen": True})
    monkeypatch.setattr(campaign, "runtime_pins", lambda: {})
    monkeypatch.setattr(campaign, "reviewed_acceptance", lambda *a, **k: {})
    monkeypatch.setattr(campaign, "require_idle_vsc", lambda: None)
    monkeypatch.setattr(text_embeddings, "encoder_config", lambda _: {"enabled": True})
    monkeypatch.setattr(campaign, "plan", lambda: [dict(name="fixture", cluster="wice", command=["sbatch"], depends_on=[])])
    monkeypatch.setattr(campaign, "load_yaml", lambda name: settings if name.endswith("scheduling") else
                        {"models": {"e5_small_v2": {}}, "full_feature_creation_enabled": True,
                         "accepted_before_extraction_manifests": dict.fromkeys(SELF_HOSTED, "reviewed.json")})

    def refuse(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, output="", stderr="QOSMaxSubmitJobPerUserLimit")

    monkeypatch.setattr(campaign.subprocess, "run", refuse)
    result = campaign.run(execute=True)
    assert result["status"] == "running"
    assert result["counts"] == {"pending": 1}
