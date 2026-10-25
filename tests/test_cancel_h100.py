"""Cancelling an owned project H100 workload cannot include CPU/other projects."""

from pathlib import Path

import pytest

from src.cluster import cancel_h100


def fixture(monkeypatch):
    monkeypatch.setattr(cancel_h100.paths, "on_vsc", lambda: True)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(cancel_h100.getpass, "getuser", lambda: "vsc_user")
    monkeypatch.setattr(cancel_h100.paths, "REPO_ROOT", Path("/data/user/JEVPFN"))
    monkeypatch.setattr(
        cancel_h100,
        "_inside",
        lambda v, root: bool(
            v and (v == "/data/user/JEVPFN" or v.startswith("/data/user/JEVPFN/"))
        ),
    )


def test_preview_skips_other_projects_even_with_similar_names(monkeypatch):
    fixture(monkeypatch)
    listing = "\n".join(
        [
            "1|vsc_user|gpu_h100|jevpfn-contracts",
            "2|vsc_user|gpu_h100|jevpfn-other-checkout",
            "3|vsc_user|batch|jevpfn-tokens",
            "4|other_user|gpu_h100|jevpfn-contracts",
            "5|vsc_user|gpu_b200|jevpfn-contracts",
            "6|vsc_user|gpu_h100|creditpfn",
        ]
    )
    records = {
        "1": "JobId=1 UserId=vsc_user(1) Partition=gpu_h100 WorkDir=/data/user/JEVPFN Command=/data/user/JEVPFN/scripts/x.slurm",
        "2": "JobId=2 UserId=vsc_user(1) Partition=gpu_h100 WorkDir=/data/user/JEVPFN-other Command=/data/user/JEVPFN-other/scripts/x.slurm",
        "6": "JobId=6 UserId=vsc_user(1) Partition=gpu_h100 WorkDir=/data/user/CreditPFN Command=/data/user/CreditPFN/scripts/x.slurm",
    }
    monkeypatch.setattr(
        cancel_h100.subprocess,
        "check_output",
        lambda c, **k: listing if c[0] == "squeue" else records[c[-1]],
    )
    monkeypatch.setattr(
        cancel_h100.subprocess, "run", lambda *a, **k: pytest.fail("Preview cancelled a job")
    )
    report = cancel_h100.run()
    assert [j["job_id"] for j in report["eligible"]] == ["1"]
    assert {j["job_id"] for j in report["excluded_h100"]} == {"2", "6"}
    assert report["applied"] is False


def test_apply_rechecks_identity_and_never_adds_new_jobs(monkeypatch):
    reports = iter(
        [
            {"eligible": [{"job_id": "1"}, {"job_id": "2"}]},
            {"eligible": [{"job_id": "2"}, {"job_id": "3"}]},
        ]
    )
    monkeypatch.setattr(cancel_h100, "inspect_jobs", lambda: next(reports))
    commands = []
    monkeypatch.setattr(cancel_h100.subprocess, "run", lambda c, **k: commands.append(c))
    report = cancel_h100.run(apply=True)
    assert commands == [["scancel", "--clusters=wice", "2"]]
    assert report["cancellation_requested"] == ["2"]


def test_live_owner_and_partition_changes_remove_selection(monkeypatch):
    fixture(monkeypatch)
    monkeypatch.setattr(
        cancel_h100.subprocess,
        "check_output",
        lambda c, **k: (
            "1|vsc_user|gpu_h100|jevpfn-contracts"
            if c[0] == "squeue"
            else "JobId=1 UserId=another_user(2) Partition=gpu_b200 WorkDir=/data/user/JEVPFN"
        ),
    )
    assert cancel_h100.inspect_jobs()["eligible"] == []


def test_individual_array_tasks_match_exact_parent_and_task(monkeypatch):
    fixture(monkeypatch)
    monkeypatch.setattr(
        cancel_h100.subprocess,
        "check_output",
        lambda c, **k: (
            "100_7|vsc_user|gpu_h100|jevpfn-array"
            if c[0] == "squeue"
            else "JobId=107 ArrayJobId=100 ArrayTaskId=7 UserId=vsc_user(1) Partition=gpu_h100 WorkDir=/data/user/JEVPFN"
        ),
    )
    assert [j["job_id"] for j in cancel_h100.inspect_jobs()["eligible"]] == ["100_7"]


def test_inside_path_is_component_bounded(tmp_path):
    checkout = (tmp_path / "JEVPFN").resolve()
    # POSIX scheduler paths are required; use an absolute POSIX representation
    # on the local platform only for this pure containment branch.
    if not str(checkout).startswith("/"):
        assert cancel_h100._inside("relative/path", checkout) is False
        return
    assert cancel_h100._inside(str(checkout / "scripts/job.slurm"), checkout)
    assert not cancel_h100._inside(str(tmp_path / "JEVPFN-other"), checkout)
