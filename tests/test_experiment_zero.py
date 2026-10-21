import json
from types import SimpleNamespace

import pytest

from src.cluster import experiment_zero as exp0
from src.utils import paths


def test_preview_never_submits_or_calls_api(monkeypatch):
    monkeypatch.setattr(exp0.subprocess, "run", lambda *a, **k: pytest.fail("No execution"))
    for site in ("local", "vsc"):
        report = exp0.run(site)
        assert report["status"] == "preview" and not report["full_feature_creation_enabled"]
    assert "jev_dataset_pilot" not in [s["name"] for s in exp0.steps("local")]
    assert exp0.steps("local", dataset_pilot=True)[-1]["name"] == "jev_dataset_pilot"


def test_vsc_jobs_resume_after_disconnect_and_submit_independent_ready_jobs(
    isolated_output, monkeypatch
):
    from src.cluster import freeze

    monkeypatch.setattr(freeze, "check_if_present", lambda: {"frozen": True})
    monkeypatch.setattr(exp0, "require_idle_vsc", lambda: None)
    commands = []
    states = {}

    def submit(command, **kwargs):
        if command[0] == "sacct":
            job = command[command.index("-j") + 1]
            return SimpleNamespace(stdout=f"{job}|{states.get(job, 'RUNNING')}|0:0\n")
        commands.append(command)
        return SimpleNamespace(stdout=f"{100 + len(commands)};wice\n")

    monkeypatch.setattr(exp0.subprocess, "run", submit)
    report = exp0.run("vsc", execute=True)
    assert len(commands) == 3  # CPU, storage and exact tokens are independent.
    while report["status"] == "waiting_for_job":
        previous = len(commands)
        exp0.run("vsc", execute=True)
        assert len(commands) == previous  # Resume does not duplicate submitted jobs.
        complete = {s["name"] for s in report["steps"] if s.get("status") == "completed"}
        for step in report["steps"]:
            if step.get("job_id"):
                assert set(step["depends_on"]) <= complete or step.get("status") == "completed"
                states[step["job_id"]] = "COMPLETED"
        report = exp0.run("vsc", execute=True)
    report = exp0.run("vsc", execute=True)
    assert report["status"] == "vsc_checks_completed_requires_review"
    assert all("--wait" not in c for c in commands)
    assert all("--mem=112000M" in c for c in commands if "perplexity" in c)
    exp0.run("vsc", execute=True)
    assert len(commands) == len(exp0.steps("vsc"))


def test_submission_failure_is_recorded_and_stops_rest(isolated_output, monkeypatch):
    monkeypatch.setattr(exp0, "require_idle_vsc", lambda: None)
    from src.cluster import freeze

    monkeypatch.setattr(freeze, "check_if_present", lambda: {"frozen": True})
    monkeypatch.setattr(exp0.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="uncertain"))
    with pytest.raises(ValueError, match="Unrecognized"):
        exp0.run("vsc", execute=True)
    receipt = next((paths.manifests_dir("experiment_0") / "0A_vsc").glob("*.json"))
    body = json.loads(receipt.read_text())
    assert body["status"] == "stopped_requires_review"
    assert body["steps"][0]["status"] == "starting"
    assert "status" not in body["steps"][1]


def test_failed_vsc_job_never_starts_the_next_test(isolated_output, monkeypatch):
    from src.cluster import freeze

    monkeypatch.setattr(freeze, "check_if_present", lambda: {"frozen": True})
    monkeypatch.setattr(exp0, "require_idle_vsc", lambda: None)
    monkeypatch.setattr(exp0.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="777;wice"))
    exp0.run("vsc", execute=True)
    monkeypatch.setattr(
        exp0.subprocess,
        "run",
        lambda c, **k: (
            SimpleNamespace(stdout="777|FAILED|1:0\n")
            if c[0] == "sacct"
            else pytest.fail("Must not submit")
        ),
    )
    with pytest.raises(RuntimeError, match="ended"):
        exp0.run("vsc", execute=True)
    with pytest.raises(ValueError, match="Earlier"):
        exp0.run("vsc", execute=True)


def test_saved_sequence_cannot_be_reused_after_source_change(isolated_output, monkeypatch):
    from src.cluster import freeze

    monkeypatch.setattr(
        freeze, "check_if_present", lambda: {"frozen": True, "source_sha256": "old"}
    )
    monkeypatch.setattr(exp0, "require_idle_vsc", lambda: None)
    monkeypatch.setattr(exp0.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="777;wice"))
    exp0.run("vsc", execute=True)
    monkeypatch.setattr(
        freeze, "check_if_present", lambda: {"frozen": True, "source_sha256": "new"}
    )
    monkeypatch.setattr(
        exp0.subprocess, "run", lambda *a, **k: pytest.fail("Must stop before querying/submitting")
    )
    with pytest.raises(ValueError, match="different source"):
        exp0.run("vsc", execute=True)
