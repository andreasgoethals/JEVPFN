"""Parallel queue admission preserves the frozen grid, storage gates and job IDs."""

import copy
import json
import subprocess

import pytest

from src.cluster import feature_admission as dispatch
from src.cluster.job_graph import AccountingUnavailable, advance
from src.cluster.workspace_graph import add_releases
from src.jev.catalog import SELF_HOSTED
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.serialization import digest, write_json


def grid():
    steps = []
    for provider in ("liquid", "strands", "perplexity", "cloudflare"):
        steps.append(dict(name=provider + "_scratch", cluster="wice", depends_on=[],
                          command=["sbatch", "stage", provider], workspace_stage=provider))
        for number in range(18):
            steps.append(dict(name=f"{provider}_dataset_{number}", cluster="mindwell", pool=provider,
                              depends_on=[provider + "_scratch"],
                              command=["sbatch", "infer", provider, str(number)]))
        if provider == "cloudflare":
            for encoder in ("small", "large", "qwen"):
                steps.append(dict(name="embeddings_" + encoder, cluster="mindwell", pool="gpu",
                                  depends_on=["cloudflare_scratch"], command=["sbatch", encoder]))
    return add_releases(steps, command_style=True)


@pytest.fixture
def campaign(isolated_output, monkeypatch):
    original = grid()
    base = load_yaml("experiment_0/scheduling")
    frozen = {"frozen": True, "source_sha256": "frozen-source", "git_head": "frozen-commit"}
    legacy = {**base, "test_pool_limits": {**base["test_pool_limits"], **dict.fromkeys(SELF_HOSTED, 2)}}
    identity = digest(dict(frozen=frozen, runtime_pins={}, steps=original, settings=legacy))
    report = {"identity": identity, "steps": copy.deepcopy(original)}
    for step in report["steps"]:
        if step["name"] in {"liquid_scratch", "strands_scratch"}:
            step.update(status="completed", job_id=step["name"] + "_job")
    path = paths.manifests_dir("feature_creation") / "open_weight_campaign.json"
    write_json(path, report)
    monkeypatch.setattr(paths, "on_vsc", lambda: True)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(dispatch, "check_if_present", lambda: frozen)
    monkeypatch.setattr(dispatch, "runtime_pins", lambda: {})
    monkeypatch.setattr(dispatch, "plan", lambda: copy.deepcopy(original))
    monkeypatch.setattr(dispatch, "reviewed_acceptance", lambda *a, **k: {})
    monkeypatch.setattr(dispatch, "load_yaml", lambda name: copy.deepcopy(base) if name.endswith("scheduling") else
                        {"full_feature_creation_enabled": True,
                         "accepted_before_extraction_manifests": dict.fromkeys(SELF_HOSTED, "accepted"),
                         "models": {"small": {}}})
    from src.data import text_embeddings

    monkeypatch.setattr(text_embeddings, "encoder_config", lambda _: {"enabled": True})
    monkeypatch.setattr(dispatch, "scheduler_state", lambda *a, **k: None)
    return path, identity, original, base


def execute(campaign):
    return dispatch.run(execute=True, expected_campaign=campaign[1],
                        expected_source="frozen-source", reviewed_by="owner-authorized fixture")


def test_all_36_ready_datasets_are_admitted_without_repeating_known_ids(campaign, monkeypatch):
    calls = []

    def submit(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=f"{1000 + len(calls)};mindwell", stderr="")

    monkeypatch.setattr(dispatch.subprocess, "run", submit)
    result = execute(campaign)
    assert len(calls) == 36
    assert result["counts"]["submitted"] == 36
    execute(campaign)
    assert len(calls) == 36
    saved = json.loads(campaign[0].read_text())
    assert saved["origin_identity"] == campaign[1]
    assert saved["identity"] != campaign[1]  # An old dispatcher cannot restart full encoders.
    assert [s["name"] for s in saved["origin_steps"]] == [s["name"] for s in campaign[2]]
    assert not any(c[1] == "stage" for c in calls)  # Two resident workspaces remain the limit.


def test_embedding_stage_gets_next_slot_before_another_decision_runtime(campaign):
    report = json.loads(campaign[0].read_text())
    report["steps"] = dispatch.partition_encoders(report["steps"], ["embeddings_small", "embeddings_large", "embeddings_qwen"])
    for step in report["steps"]:
        if step.get("pool") == "liquid" or step["name"] == "liquid_release":
            step.update(status="completed", job_id="done-" + step["name"])
        elif step.get("pool") == "strands":
            step.update(status="submitted", job_id="active-" + step["name"])
    settings = dispatch.admission_settings(campaign[3], report["steps"])
    calls = []

    def submit(step):
        calls.append(step["name"])
        step.update(status="submitted", job_id="new-" + step["name"])

    advance(dispatch.dispatch_order(report["steps"]), settings=settings,
            state=lambda *a, **k: None, submit=submit, save=lambda: None)
    assert calls == ["cloudflare_scratch"]
    cloud = next(s for s in report["steps"] if s["name"] == "cloudflare_scratch")
    cloud["status"] = "completed"
    advance(dispatch.dispatch_order(report["steps"]), settings=settings,
            state=lambda *a, **k: None, submit=submit, save=lambda: None)
    assert all(name.startswith("embeddings_") for name in calls[1:55])
    assert len(calls) == 73  # One stage, fifty-four encoder/datasets and eighteen decision datasets.


def test_accounting_outage_keeps_ids_and_does_not_submit(campaign, monkeypatch):
    report = json.loads(campaign[0].read_text())
    report["steps"][1].update(status="submitted", job_id="known")
    write_json(campaign[0], report)

    def unavailable(*a, **k):
        raise AccountingUnavailable(dict(cluster="mindwell", job_id="known", error_type="timeout"))

    monkeypatch.setattr(dispatch, "scheduler_state", unavailable)
    monkeypatch.setattr(dispatch.subprocess, "run", lambda *a, **k: pytest.fail("No admission during outage"))
    result = execute(campaign)
    assert result["status"] == "running"
    assert result["accounting_unavailable"][0]["job_id"] == "known"
    assert json.loads(campaign[0].read_text())["steps"][1]["job_id"] == "known"


def test_definite_full_queue_waits_but_uncertain_submission_stays_blocked(campaign, monkeypatch):
    def full(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, output="", stderr="QOSMaxSubmitJobPerUserLimit")

    monkeypatch.setattr(dispatch.subprocess, "run", full)
    result = execute(campaign)
    assert result["failures"] == []
    assert not any(s.get("status") == "starting" for s in json.loads(campaign[0].read_text())["steps"])

    def uncertain(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 60)

    monkeypatch.setattr(dispatch.subprocess, "run", uncertain)
    with pytest.raises(subprocess.TimeoutExpired):
        execute(campaign)
    monkeypatch.setattr(dispatch.subprocess, "run", lambda *a, **k: pytest.fail("Ambiguous intent must not be retried"))
    with pytest.raises(ValueError, match="Interrupted submission"):
        execute(campaign)


def test_changed_frozen_source_or_plan_refuses_admission(campaign, monkeypatch):
    monkeypatch.setattr(dispatch.subprocess, "run", lambda *a, **k: pytest.fail("No submission on mismatch"))
    with pytest.raises(ValueError, match="frozen inference source"):
        dispatch.run(execute=True, expected_campaign=campaign[1], expected_source="changed", reviewed_by="reviewer")
    report = json.loads(campaign[0].read_text())
    report["steps"][1]["command"].append("changed")
    write_json(campaign[0], report)
    with pytest.raises(ValueError, match="plan/dependencies"):
        execute(campaign)


def test_encoder_partitions_cover_each_dataset_once_and_bind_publication():
    steps = grid()
    selected = [s["name"] for s in steps if s["name"].startswith("embeddings_")]
    expanded = dispatch.partition_encoders(steps, selected)
    for encoder in selected:
        children = [s for s in expanded if s.get("encoder_group") == encoder]
        assert len(children) == 18
        assert {s["command"][-1] for s in children} == {str(n) for n in range(18)}
        assert all("JEVPFN_PUBLISH_DATASET=" + s["dataset"] in s["command"][1] for s in children)
    release = next(s for s in expanded if s["name"] == "cloudflare_release")
    assert all(s["name"] in release["terminal_dependencies"] for s in expanded if s.get("encoder_group"))
    already = copy.deepcopy(steps)
    next(s for s in already if s["name"] == selected[0])["job_id"] = "known"
    with pytest.raises(ValueError, match="already submitted"):
        dispatch.partition_encoders(already, selected)


def test_source_recovery_preserves_completed_grid_and_retries_only_failed_attempts(campaign, monkeypatch):
    from src.cluster import freeze

    counter = iter(range(1000, 1100))
    monkeypatch.setattr(dispatch.subprocess, "run", lambda command, **kw:
                        subprocess.CompletedProcess(command, 0, stdout=f"{next(counter)};mindwell", stderr=""))
    execute(campaign)
    report = json.loads(campaign[0].read_text())
    old_identity = report["identity"]
    release = next(s for s in report["steps"] if s["name"] == "liquid_release")
    release.update(status="failed", job_id="bad-release")
    next(s for s in report["steps"] if s["name"] == "cloudflare_scratch").update(
        status="blocked", blocked_by=["liquid_release"])
    write_json(campaign[0], report)
    old_freeze = campaign[0].parent / "completed.json"
    write_json(old_freeze, {"frozen_receipt": {"files": {}, "runtime_pins": {}, "source_sha256": "frozen-source"}})
    monkeypatch.setattr(dispatch, "require_idle_vsc", lambda: None)
    monkeypatch.setattr(freeze, "source_snapshot", lambda: {"files": {}})
    monkeypatch.setattr(dispatch, "scheduler_state", lambda job, **kw:
                        ("TIMEOUT", "0:0") if job == "bad-release" else ("COMPLETED", "0:0"))
    identity = dispatch.recover_campaign(previous_identity=old_identity, previous_freeze=old_freeze,
                                        expected_source="frozen-source", reviewed_by="owner-authorized fixture")
    recovered = json.loads(campaign[0].read_text())
    assert sum(s.get("status") == "completed" for s in recovered["steps"]) == 38
    retry = next(s for s in recovered["steps"] if s["name"] == "liquid_release")
    assert "job_id" not in retry and retry["previous_attempt"]["job_id"] == "bad-release"
    assert next(s for s in recovered["steps"] if s["name"] == "cloudflare_scratch").get("status", "pending") == "pending"
    assert len([s for s in recovered["steps"] if s.get("encoder_group")]) == 54
    # The recovered identity passes the normal immutable-policy checks.
    monkeypatch.setattr(dispatch.subprocess, "run", lambda command, **kw:
                        subprocess.CompletedProcess(command, 0, stdout="2000;wice", stderr=""))
    dispatch.run(execute=True, expected_campaign=identity, expected_source="frozen-source",
                 reviewed_by="owner-authorized fixture")


def test_source_recovery_refuses_unapproved_scientific_changes_before_accounting(campaign, monkeypatch):
    from src.cluster import freeze

    old = campaign[0].parent / "completed.json"
    write_json(old, {"frozen_receipt": {"files": {}, "runtime_pins": {}}})
    monkeypatch.setattr(dispatch, "require_idle_vsc", lambda: None)
    monkeypatch.setattr(freeze, "source_snapshot", lambda: {"files": {"src/jev/self_hosted.py": "changed"}})
    monkeypatch.setattr(dispatch, "scheduler_state", lambda *a, **k: pytest.fail("Refuse before scheduler work"))
    with pytest.raises(ValueError, match="changed science"):
        dispatch.recover_campaign(previous_identity="any", previous_freeze=old,
                                  expected_source="frozen-source", reviewed_by="fixture")


def test_recovery_restages_a_retired_runtime_without_repeating_successful_inference(campaign, monkeypatch):
    from src.cluster import freeze

    monkeypatch.setattr(dispatch.subprocess, "run", lambda command, **kw:
                        subprocess.CompletedProcess(command, 0, stdout="1000;mindwell", stderr=""))
    execute(campaign)
    report = json.loads(campaign[0].read_text())
    for step in report["steps"]:
        if step.get("pool") == "perplexity":
            step.update(status="failed", job_id="failed-" + step["name"])
        if step["name"] in {"perplexity_scratch", "perplexity_release", "perplexity_dataset_0"}:
            step.update(status="completed", job_id="done-" + step["name"])
    write_json(campaign[0], report)
    old_freeze = campaign[0].parent / "completed.json"
    write_json(old_freeze, {"frozen_receipt": {"files": {}, "runtime_pins": {}, "source_sha256": "frozen-source"}})
    monkeypatch.setattr(dispatch, "require_idle_vsc", lambda: None)
    monkeypatch.setattr(freeze, "source_snapshot", lambda: {"files": {}})
    monkeypatch.setattr(dispatch, "scheduler_state", lambda job, **kw:
                        ("FAILED", "2:0") if job.startswith("failed-") else ("COMPLETED", "0:0"))
    dispatch.recover_campaign(previous_identity=report["identity"], previous_freeze=old_freeze,
                             expected_source="frozen-source", reviewed_by="fixture")
    steps = {s["name"]: s for s in json.loads(campaign[0].read_text())["steps"]}
    for name in ("perplexity_scratch", "perplexity_release"):
        assert steps[name].get("status", "pending") == "pending"
        assert "job_id" not in steps[name]
        assert steps[name]["previous_attempt"]["job_id"] == "done-" + name
    assert steps["perplexity_dataset_0"]["status"] == "completed"
    assert steps["perplexity_dataset_1"].get("status", "pending") == "pending"
    # Already resident, unfinished Liquid/Strands sets retain their staging receipts.
    assert steps["liquid_scratch"]["status"] == "completed"


@pytest.mark.parametrize("name", ["src/jev/self_hosted.py", "src/data/text_embeddings.py",
                                  "config/2. feature_creation/embedding_models.yaml",
                                  "config/3. experiment_0/self_hosted.yaml",
                                  "src/cluster/feature_campaign.py"])
def test_explicit_review_cannot_override_extraction_science(tmp_path, name):
    old = {"files": {name: "a"}, "source_sha256": "old"}
    new = {"files": {name: "b"}, "source_sha256": "new"}
    p = tmp_path / "review.json"
    write_json(p, {"schema": "jevpfn.extraction-source-review.v1", "reviewed_by": "reviewer",
                   "previous_source": "old", "next_source": "new", "rationale": "reviewed",
                   "changes": {name: {"before": "a", "after": "b"}}})
    with pytest.raises(ValueError, match="changed extraction science"):
        dispatch.validate_source_review(old, new, {name}, p, "reviewer")


def test_exact_review_accepts_future_benchmark_code_but_rejects_stale_delta(tmp_path):
    name = "src/evaluation/tar.py"
    old = {"files": {}, "source_sha256": "old"}
    new = {"files": {name: "a"}, "source_sha256": "new"}
    p = tmp_path / "review.json"
    write_json(p, {"schema": "jevpfn.extraction-source-review.v1", "reviewed_by": "reviewer",
                   "previous_source": "old", "next_source": "new",
                   "rationale": "Future TAR remains disabled; extraction unchanged",
                   "changes": {name: {"before": None, "after": "a"}}})
    result = dispatch.validate_source_review(old, new, {name}, p, "reviewer")
    assert len(result["sha256"]) == 64
    with pytest.raises(ValueError, match="exact reviewed transition"):
        dispatch.validate_source_review(old, new, {name}, p, "another reviewer")
    new["files"][name] = "unreviewed-edit"
    with pytest.raises(ValueError, match="exact reviewed transition"):
        dispatch.validate_source_review(old, new, {name}, p, "reviewer")


def test_timeout_retries_staging_before_consumers_and_preserves_attempts():
    steps = [
        dict(name="intern_scratch", cluster="mindwell", workspace_stage="intern",
             status="submitted", job_id="1"),
        dict(name="intern_data", cluster="mindwell", pool="intern",
             depends_on=["intern_scratch"]),
    ]
    settings = {"max_active_jobs": 2, "max_resident_workspaces": 1,
                "test_pool_limits": {"transfer": 1, "intern": 1}}
    outcomes = {"1": ("TIMEOUT", "0:0")}
    receipts = []
    calls = []

    def submit(step):
        calls.append(step["name"])
        step.update(status="submitted", job_id=str(len(calls) + 1))

    def poll():
        return advance(steps, settings=settings, state=lambda job, **_: outcomes.get(job),
                       retry=dispatch.retry_interrupted, submit=submit,
                       save=lambda: receipts.append(copy.deepcopy(steps)))

    poll()
    assert calls == ["intern_scratch"]
    assert any(record[0].get("status") == "pending" and "job_id" not in record[0]
               for record in receipts)
    assert steps[0]["scheduler_retries"][0]["job_id"] == "1"
    outcomes["2"] = ("COMPLETED", "0:0")
    poll()
    assert calls == ["intern_scratch", "intern_data"]
    poll()
    assert calls == ["intern_scratch", "intern_data"]


@pytest.mark.parametrize("state", ["FAILED", "OUT_OF_MEMORY", "CANCELLED", "TIMEOUT"])
def test_retry_exhaustion_and_permanent_failures_are_terminal(state):
    step = {"status": "failed", "job_id": "3", "name": "intern_scratch",
            "scheduler_state": state, "exit_code": "1:0",
            "scheduler_retries": [{"job_id": "1"}, {"job_id": "2"}] if state == "TIMEOUT" else []}
    before = copy.deepcopy(step)
    dispatch.retry_interrupted(step)
    assert step == before


@pytest.mark.parametrize("release", [None, "perplexity"])
@pytest.mark.parametrize("exit_code", [
    dispatch.QUEUE_CHECK_UNAVAILABLE, dispatch.PUBLICATION_LOCK_UNAVAILABLE,
])
def test_retry_transient_exit_only_for_workspace_release(release, exit_code):
    step = {"status": "failed", "job_id": "7", "name": "perplexity_release",
            "workspace_release": release, "scheduler_state": "FAILED",
            "exit_code": f"{exit_code}:0"}
    dispatch.retry_interrupted(step)
    if release:
        assert step["status"] == "pending"
        assert step["scheduler_retries"][0]["job_id"] == "7"
    else:
        assert step["status"] == "failed"


def test_full_scratch_budget_makes_staging_wait_not_fail():
    step = {"status": "failed", "job_id": "9", "name": "kev_scratch", "workspace_stage": "kev",
            "scheduler_state": "FAILED", "exit_code": f"{dispatch.SCRATCH_CAPACITY_UNAVAILABLE}:0"}
    dispatch.retry_interrupted(step)
    assert step["status"] == "pending" and step["capacity_retries"][0]["job_id"] == "9"
    assert "scheduler_retries" not in step  # Transient-failure retries stay available.
    step.update(status="failed", job_id="10", scheduler_state="FAILED",
                exit_code=f"{dispatch.SCRATCH_CAPACITY_UNAVAILABLE}:0",
                capacity_retries=[{"job_id": str(i)} for i in range(dispatch.MAX_CAPACITY_RETRIES)])
    before = copy.deepcopy(step)
    dispatch.retry_interrupted(step)
    assert step == before  # Bounded: a permanently full budget needs review.


def test_capacity_exit_code_never_retries_non_staging_steps():
    step = {"status": "failed", "job_id": "9", "name": "kev_BIN_TEXT_X", "pool": "kev",
            "scheduler_state": "FAILED", "exit_code": f"{dispatch.SCRATCH_CAPACITY_UNAVAILABLE}:0"}
    before = copy.deepcopy(step)
    dispatch.retry_interrupted(step)
    assert step == before


def test_staging_reports_a_full_budget_with_its_own_exit_status(monkeypatch):
    from src.cluster import scratch_capacity, scratch_stage

    def full(runtime):
        raise scratch_capacity.ScratchCapacityUnavailable("JEVPFN GPFS byte budget exceeded")

    monkeypatch.setattr(scratch_stage, "stage", full)
    monkeypatch.setattr("sys.argv", ["scratch_stage", "stage", "--runtime", "kev"])
    with pytest.raises(SystemExit) as stopped:
        scratch_stage.main()
    assert stopped.value.code == scratch_capacity.SCRATCH_CAPACITY_UNAVAILABLE
