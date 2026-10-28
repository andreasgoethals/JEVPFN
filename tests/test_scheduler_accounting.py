"""Accounting outages must not discard known jobs or retry ambiguous submissions."""

import json
import subprocess
from copy import deepcopy
from types import SimpleNamespace

import pytest

from src.cluster import experiment_zero, freeze, job_graph, setup_extractors
from src.utils import paths
from src.utils.config import load_yaml


@pytest.mark.parametrize("timeout", [False, True])
def test_query_failure_is_distinct_and_retains_diagnostics(monkeypatch, timeout):
    def fail(command, **kwargs):
        assert command[0] == "sacct" and kwargs["timeout"] == 30
        if timeout:
            raise subprocess.TimeoutExpired(command, 30, stderr="Connection refused")
        raise subprocess.CalledProcessError(1, command, stderr="Connection refused")

    monkeypatch.setattr(experiment_zero.subprocess, "run", fail)
    with pytest.raises(job_graph.AccountingUnavailable) as caught:
        experiment_zero.scheduler_state("62452620")
    assert caught.value.diagnostics["stderr"] == "Connection refused"
    assert caught.value.diagnostics["job_id"] == "62452620"
    assert caught.value.__cause__ is not None


@pytest.mark.parametrize("row", ["", "42||0:0", "42|COMPLETED|", "42.batch|COMPLETED|0:0"])
def test_missing_or_incomplete_allocation_does_not_establish_success(monkeypatch, row):
    monkeypatch.setattr(experiment_zero.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=row))
    assert experiment_zero.scheduler_state("42") is None


@pytest.mark.parametrize("site", ["setup", "frozen"])
@pytest.mark.parametrize("timeout", [False, True])
def test_watchers_preserve_jobs_during_outage_and_resume_without_duplicates(
    tmp_path, monkeypatch, site, timeout
):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(paths, "on_vsc", lambda: True)
    monkeypatch.setattr(paths, "manifests_dir", lambda *a: tmp_path / "manifests")
    monkeypatch.setattr(paths, "logs_dir", lambda *a: tmp_path / "logs")
    monkeypatch.setattr(setup_extractors, "require_idle_vsc", lambda: None)
    monkeypatch.setattr(experiment_zero, "require_idle_vsc", lambda: None)
    monkeypatch.setattr(setup_extractors, "source_snapshot", lambda: {"source": "unchanged"})
    monkeypatch.setattr(setup_extractors, "check_if_present", lambda: {"frozen": False})
    monkeypatch.setattr(freeze, "check_if_present", lambda: {"frozen": True})
    monkeypatch.setattr(freeze, "runtime_pins", lambda: {})
    plan = [dict(name="cpu", cluster="wice", dependencies=[]),
            dict(name="storage", cluster="wice", dependencies=[]),
            dict(name="decision3_contracts", cluster="mindwell", dependencies=["cpu", "storage"])]
    for step in plan:
        step.update(script=step["name"] + ".slurm", args=[], flags=[],
                    command=["sbatch", f"--clusters={step['cluster']}", step["name"]])
    monkeypatch.setattr(setup_extractors, "plan", lambda *a, **k: deepcopy(plan))
    monkeypatch.setattr(experiment_zero, "steps", lambda *a, **k: deepcopy(plan))
    run = (lambda: setup_extractors.run(["decision3"], execute=True)) if site == "setup" else (
        lambda: experiment_zero.run("vsc", execute=True)
    )
    submissions, queries = [], []
    phase = "initial"

    def scheduler(command, **kwargs):
        if command[0] == "sacct":
            queries.append(command)
            if phase == "outage":
                if timeout:
                    raise subprocess.TimeoutExpired(command, 30, stderr="Connection refused")
                raise subprocess.CalledProcessError(1, command, stderr="Connection refused")
            job = command[command.index("-j") + 1]
            return SimpleNamespace(stdout=f"{job}|COMPLETED|0:0\n")
        submissions.append(command)
        cluster = next(x.split("=", 1)[1] for x in command if x.startswith("--clusters="))
        return SimpleNamespace(stdout=f"{len(submissions)};{cluster}\n")

    monkeypatch.setattr(experiment_zero.subprocess, "run", scheduler)
    initial = run()
    assert len(submissions) == 2
    phase = "outage"
    for _ in range(2):
        waiting = run()
        assert waiting["status"] == "waiting_for_job"
        assert waiting["summary"]["accounting_unavailable"][0]["job_id"] == "1"
        assert not waiting["summary"]["failures"]
        assert [s.get("job_id") for s in waiting["steps"]] == [s.get("job_id") for s in initial["steps"]]
        assert len(submissions) == 2
    assert len(queries) == 2  # Once per cluster per pass, not once per queued job.
    receipt = next((tmp_path / "manifests").rglob("*.json"))
    saved = json.loads(receipt.read_text())
    assert saved["steps"][0]["accounting_diagnostics"]["stderr"] == "Connection refused"
    phase = "recovered"
    resumed = run()
    assert resumed["summary"]["accounting_unavailable"] == []
    assert resumed["steps"][0]["last_accounting_failure"]["stderr"] == "Connection refused"
    assert len(submissions) == 3
    finished = run()
    assert finished["summary"]["all_terminal"] and not finished["summary"]["failures"]
    assert len(submissions) == 3


def test_other_cluster_can_be_reconciled_but_new_submissions_wait():
    steps = [dict(name="cpu", cluster="wice", job_id="1", status="submitted"),
             dict(name="gpu_contracts", cluster="mindwell", job_id="2", status="submitted"),
             dict(name="gpu_pilot", cluster="mindwell", dependencies=["gpu_contracts"])]
    calls = []

    def state(job, *, cluster):
        calls.append(job)
        if cluster == "wice":
            raise job_graph.AccountingUnavailable(dict(cluster=cluster, job_id=job, error_type="test"))
        return "COMPLETED", "0:0"

    summary = job_graph.advance(steps, settings=load_yaml("experiment_0/scheduling"),
                                state=state, submit=lambda s: pytest.fail("Wait for accounting"),
                                save=lambda: None)
    assert calls == ["1", "2"] and steps[1]["status"] == "completed"
    assert not summary["all_terminal"] and not summary["failures"]


def test_ambiguous_submission_still_stops_even_during_outage():
    steps = [dict(name="cpu", cluster="wice", job_id="1", status="submitted"),
             dict(name="storage", cluster="wice", status="starting")]
    with pytest.raises(ValueError, match="Interrupted submission"):
        job_graph.advance(steps, settings=load_yaml("experiment_0/scheduling"),
                          state=lambda *a, **k: pytest.fail("Ambiguity must be checked first"),
                          submit=lambda s: pytest.fail("No new submissions"), save=lambda: None)


def test_programming_errors_are_not_swallowed_as_accounting_outages():
    def broken(*a, **k):
        raise ValueError("implementation error")

    with pytest.raises(ValueError, match="implementation error"):
        job_graph.advance([dict(name="cpu", cluster="wice", job_id="1", status="submitted")],
                          settings=load_yaml("experiment_0/scheduling"), state=broken,
                          submit=lambda s: pytest.fail("No submission"), save=lambda: None)
