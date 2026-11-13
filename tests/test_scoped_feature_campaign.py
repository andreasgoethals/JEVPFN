"""A new provider must not replace or extend the historical extraction receipt."""

import copy
import json
import subprocess
from contextlib import contextmanager

import pytest

from src.cluster import feature_campaign as campaign
from src.cluster.job_graph import validate
from src.jev.catalog import SELF_HOSTED
from src.utils.config import load_yaml


@pytest.mark.parametrize("providers", [[], (), ["unknown"], ["jev"], "glide"])
def test_invalid_scopes_fail_before_planning(providers):
    with pytest.raises(ValueError, match="Provider scope"):
        campaign.plan(providers)


@pytest.mark.parametrize("provider", ["glide", "cloudflare"])
def test_provider_extension_contains_only_its_eighteen_datasets_and_release(provider):
    steps = campaign.plan([provider])
    settings = load_yaml("experiment_0/scheduling")
    settings["test_pool_limits"][provider] = 2
    validate(steps, settings)
    assert len(steps) == 20
    datasets = [s for s in steps if s.get("pool") == provider]
    assert len(datasets) == 18
    assert len({s["command"][-1] for s in datasets}) == 18
    assert all(s["depends_on"] == [provider + "_scratch"] for s in datasets)
    release = next(s for s in steps if s.get("workspace_release"))
    assert set(release["terminal_dependencies"]) == {s["name"] for s in steps[:-1]}
    assert not any(s["name"].startswith("embeddings_") for s in steps)
    preview = campaign.run(providers=[provider])
    assert preview["decision_datasets"] == 18 and preview["encoders"] == 0
    assert preview["receipt"].endswith(f"open_weight_campaign_{provider}.json")


def test_scope_order_deduplication_and_default_receipt_stay_deterministic():
    assert campaign.plan(["glide", "intern", "glide"]) == campaign.plan(["intern", "glide"])
    assert campaign.campaign_receipt(["glide", "intern"]) == campaign.campaign_receipt(["intern", "glide"])
    assert campaign.campaign_receipt().name == "open_weight_campaign.json"
    assert campaign.campaign_receipt(SELF_HOSTED) != campaign.campaign_receipt()
    preview = campaign.run()
    assert preview["decision_datasets"] == sum(s.get("pool") in SELF_HOSTED for s in preview["steps"])
    assert preview["encoders"] == 3


def ready_scope(monkeypatch):
    """Mock host/scheduler only; exercise actual scoped grid and receipt handling."""
    original_load = campaign.load_yaml
    cfg = copy.deepcopy(original_load("experiment_0/self_hosted"))
    cfg["full_feature_creation_enabled"] = True
    cfg["accepted_before_extraction_manifests"]["glide"] = "glide-reviewed.json"
    monkeypatch.setattr(campaign, "load_yaml", lambda name: cfg if name == "experiment_0/self_hosted" else original_load(name))
    monkeypatch.setattr(campaign.paths, "on_vsc", lambda: True)
    monkeypatch.setattr(campaign, "check_if_present", lambda: {"frozen": True, "source_sha256": "frozen-source"})
    monkeypatch.setattr(campaign, "runtime_pins", lambda: {"glide": "sealed-image"})
    monkeypatch.setattr(campaign, "scheduler_state", lambda *a, **k: None)
    approvals, idle, submissions = [], [], []
    monkeypatch.setattr(campaign, "reviewed_acceptance", lambda path, **kw: approvals.append((path, kw)))
    monkeypatch.setattr(campaign, "require_idle_vsc", lambda: idle.append(True))

    def submit(command, **kwargs):
        submissions.append(command)
        cluster = next(arg.split("=", 1)[1] for arg in command if arg.startswith("--clusters="))
        return subprocess.CompletedProcess(command, 0, stdout=f"{100 + len(submissions)};{cluster}", stderr="")

    monkeypatch.setattr(campaign.subprocess, "run", submit)
    return cfg, approvals, idle, submissions


def test_scoped_execution_preserves_original_receipt_and_checks_only_selected_acceptance(isolated_output, monkeypatch):
    _, approvals, idle, submissions = ready_scope(monkeypatch)
    original = campaign.campaign_receipt()
    original.parent.mkdir(parents=True)
    old_bytes = b'{"historical_completed_campaign":true}\n'
    original.write_bytes(old_bytes)
    result = campaign.run(providers=["glide"], execute=True)
    assert result["status"] == "running"
    assert approvals == [("glide-reviewed.json", {"stage": "before_extraction", "expected": {"provider": "glide"}})]
    assert idle == [True]
    assert len(submissions) == 1 and submissions[0][-1] == "glide"
    saved = json.loads(campaign.campaign_receipt(["glide"]).read_text())
    assert saved["provider_scope"] == ["glide"]
    assert saved["steps"][0]["job_id"] == "101"
    assert original.read_bytes() == old_bytes
    # A watcher resumes its own known allocation without requiring it to be idle
    # or submitting a duplicate. It never reads the old campaign as its own.
    campaign.run(providers=["glide"], execute=True)
    assert idle == [True] and len(submissions) == 1
    assert original.read_bytes() == old_bytes


def test_pending_glide_acceptance_prevents_any_submission(isolated_output, monkeypatch):
    from src.utils.acceptance import reviewed_acceptance

    cfg, _, idle, submissions = ready_scope(monkeypatch)
    cfg["accepted_before_extraction_manifests"]["glide"] = None
    monkeypatch.setattr(campaign, "reviewed_acceptance", reviewed_acceptance)
    with pytest.raises(ValueError, match="reviewed before_extraction"):
        campaign.run(providers=["glide"], execute=True)
    assert not idle and not submissions
    assert not campaign.campaign_receipt(["glide"]).exists()


def test_scoped_start_requires_frozen_source_and_no_existing_consumers(isolated_output, monkeypatch):
    _, _, idle, submissions = ready_scope(monkeypatch)
    monkeypatch.setattr(campaign, "check_if_present", lambda: {"frozen": False})
    with pytest.raises(ValueError, match="Freeze accepted"):
        campaign.run(providers=["glide"], execute=True)
    assert not idle and not submissions
    monkeypatch.setattr(campaign, "check_if_present", lambda: {"frozen": True})

    def busy():
        raise RuntimeError("queued/running JEVPFN jobs")

    monkeypatch.setattr(campaign, "require_idle_vsc", busy)
    with pytest.raises(RuntimeError, match="queued/running"):
        campaign.run(providers=["glide"], execute=True)
    assert not submissions and not campaign.campaign_receipt(["glide"]).exists()


def test_scoped_recovery_still_refuses_changed_grid(isolated_output, monkeypatch):
    ready_scope(monkeypatch)
    campaign.run(providers=["glide"], execute=True)
    receipt = campaign.campaign_receipt(["glide"])
    saved = json.loads(receipt.read_text())
    saved["steps"][1]["command"][-1] = "UNREVIEWED_DATASET"
    receipt.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="cannot change the campaign plan"):
        campaign.run(providers=["glide"], execute=True, recover=True, reviewed_by="test reviewer")


def test_scoped_recovery_restages_a_released_workspace_before_retrying_gpu(isolated_output, monkeypatch):
    _, _, _, submissions = ready_scope(monkeypatch)
    campaign.run(providers=["glide"], execute=True)
    receipt = campaign.campaign_receipt(["glide"])
    saved = json.loads(receipt.read_text())
    for index, step in enumerate(saved["steps"]):
        step.update(job_id=str(1001 + index), status="completed")
    failed = saved["steps"][1]
    failed["status"] = "failed"
    receipt.write_text(json.dumps(saved))
    monkeypatch.setattr(campaign, "scheduler_state", lambda job, **kw:
                        ("FAILED", "1:0") if job == failed["job_id"] else ("COMPLETED", "0:0"))
    campaign.run(providers=["glide"], execute=True, recover=True, reviewed_by="test reviewer")
    result = json.loads(receipt.read_text())
    stage, release = result["steps"][0], result["steps"][-1]
    assert len(submissions) == 2
    assert "scripts/slurm/experiment_0_scratch_stage.slurm" in submissions[-1]
    assert stage["previous_attempt"]["job_id"] == "1001" and stage["status"] == "submitted"
    assert release["previous_attempt"]["job_id"] == "1020" and "job_id" not in release
    assert "job_id" not in result["steps"][1]
    assert result["steps"][2]["job_id"] == "1003"  # A successful dataset is reused.


def test_scoped_cli_holds_original_dispatcher_and_campaign_locks(monkeypatch, capsys):
    import sys

    locks, calls = [], []

    @contextmanager
    def lock(path, **kwargs):
        locks.append(path.name)
        yield

    monkeypatch.setattr(campaign, "file_lock", lock)
    monkeypatch.setattr(campaign, "run", lambda **kwargs: calls.append(kwargs) or {"status": "completed_requires_feature_validation"})
    monkeypatch.setattr(sys, "argv", ["feature_campaign", "--provider", "glide", "--execute"])
    assert campaign.main() == 0
    assert locks == [".parallel-controller.lock", ".campaign.lock"]
    assert calls[0]["providers"] == ["glide"]
    assert "completed_requires_feature_validation" in capsys.readouterr().out
