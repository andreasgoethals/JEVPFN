"""An accepted-but-timed-out submission must not become a duplicate job."""

import json
import shlex
import subprocess
from types import SimpleNamespace

import pytest

from src.cluster import setup_recovery as recovery
from src.utils.serialization import digest, write_json


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery.paths, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(recovery.paths, "manifests_dir", lambda *a: tmp_path)
    monkeypatch.setattr(recovery.getpass, "getuser", lambda: "vsc38338")
    monkeypatch.setattr(recovery, "require_idle_vsc", lambda: None)
    command = [
        "sbatch",
        "--parsable",
        "--clusters=mindwell",
        "--account=lp_verbekelab",
        "scripts/slurm/experiment_0_storage_gpfs.slurm",
        "liquid",
    ]
    report = {
        "identity": "original-source",
        "status": "stopped_requires_review",
        "steps": [
            {
                "name": "liquid_tokens",
                "cluster": "wice",
                "status": "completed",
                "job_id": "62431739",
            },
            {
                "name": "liquid_gpfs_storage",
                "cluster": "mindwell",
                "status": "starting",
                "command": command,
                "submission_diagnostics": {"stderr": "Socket timed out"},
            },
        ],
    }
    receipt = tmp_path / "container_setup" / "open_models_mindwell_b200.json"
    write_json(receipt, report)
    evidence = dict(
        zip(
            recovery.FIELDS,
            [
                "11739854",
                "vsc38338",
                "lp_verbekelab",
                "COMPLETED",
                "0:0",
                str(tmp_path),
                shlex.join(command),
            ],
            strict=True,
        )
    )
    calls = []

    def query(argv, **kwargs):
        calls.append(argv)
        assert argv[0] == "sacct"  # No submission or cancellation is ever permitted.
        assert "--clusters=mindwell" in argv and "--allocations" in argv
        return SimpleNamespace(stdout="|".join(evidence[field] for field in recovery.FIELDS) + "\n")

    monkeypatch.setattr(recovery.subprocess, "run", query)
    return receipt, report, evidence, calls


def test_preview_apply_and_idempotence_preserve_history(scenario):
    receipt, original, evidence, calls = scenario
    before = receipt.read_bytes()
    preview = recovery.run("liquid_gpfs_storage", "11739854")
    assert preview["status"] == "preview" and receipt.read_bytes() == before
    result = recovery.run("liquid_gpfs_storage", "11739854", apply=True)
    assert result["status"] == "recorded_requires_restart"
    report = json.loads(receipt.read_text())
    step = report["steps"][1]
    assert report["status"] == "stopped_requires_review"
    assert report["identity"] == original["identity"]
    assert report["steps"][0] == original["steps"][0]
    assert step["submission_diagnostics"] == original["steps"][1]["submission_diagnostics"]
    assert step["job_id"] == "11739854" and step["status"] == "submitted"
    assert step["submission_recovery"]["prior_receipt_sha256"] == digest(original)
    assert step["submission_recovery"]["accounting"] == evidence
    after = receipt.read_bytes()
    assert (
        recovery.run("liquid_gpfs_storage", "11739854", apply=True)["status"] == "already_recorded"
    )
    assert receipt.read_bytes() == after and len(calls) == 3


@pytest.mark.parametrize(
    "field,value",
    [
        ("User", "another-user"),
        ("Account", "another-project"),
        ("WorkDir", "/another/checkout"),
        ("SubmitLine", "sbatch scripts/slurm/other.slurm"),
        ("SubmitLine", ""),
        ("State", "RUNNING"),
        ("State", "UNKNOWN"),
        ("State", ""),
        ("ExitCode", ""),
        ("JobIDRaw", "11739854.batch"),
    ],
)
def test_mismatched_or_incomplete_accounting_never_changes_receipt(scenario, field, value):
    receipt, _, evidence, _ = scenario
    evidence[field] = value
    before = receipt.read_bytes()
    with pytest.raises(ValueError):
        recovery.run("liquid_gpfs_storage", "11739854", apply=True)
    assert receipt.read_bytes() == before


@pytest.mark.parametrize("stdout", ["", "11739854|vsc38338\n", "junk\njunk\n"])
def test_missing_or_ambiguous_records_refused(scenario, monkeypatch, stdout):
    receipt, _, _, _ = scenario
    monkeypatch.setattr(recovery.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=stdout))
    before = receipt.read_bytes()
    with pytest.raises(ValueError, match="exactly one"):
        recovery.run("liquid_gpfs_storage", "11739854", apply=True)
    assert receipt.read_bytes() == before


def test_terminal_failure_is_recorded_without_claiming_a_pass(scenario):
    receipt, _, evidence, _ = scenario
    evidence.update(State="FAILED", ExitCode="1:0")
    result = recovery.run("liquid_gpfs_storage", "11739854", apply=True)
    assert result["final_acceptance"] is False and result["jobs_submitted"] is False
    assert json.loads(receipt.read_text())["steps"][1]["status"] == "submitted"


@pytest.mark.parametrize(
    "failure", ["active_jobs", "scheduler_timeout", "duplicate_id", "different_id", "not_stopped"]
)
def test_unresolved_state_does_not_bypass_safety(scenario, monkeypatch, failure):
    receipt, report, _, _ = scenario
    if failure == "active_jobs":

        def busy():
            raise RuntimeError("Wait for queued/running JEVPFN jobs")

        monkeypatch.setattr(recovery, "require_idle_vsc", busy)
    elif failure == "scheduler_timeout":

        def timeout(*a, **k):
            raise subprocess.TimeoutExpired("sacct", 30)

        monkeypatch.setattr(recovery.subprocess, "run", timeout)
    elif failure == "duplicate_id":
        report["steps"][0].update(cluster="mindwell", job_id="11739854")
    elif failure == "different_id":
        report["steps"][1]["job_id"] = "99999"
    else:
        report["status"] = "waiting_for_job"
    write_json(receipt, report)
    before = receipt.read_bytes()
    with pytest.raises((RuntimeError, ValueError, subprocess.TimeoutExpired)):
        recovery.run("liquid_gpfs_storage", "11739854", apply=True)
    assert receipt.read_bytes() == before


def test_absolute_sbatch_binary_matches(scenario):
    _, _, evidence, _ = scenario
    evidence["SubmitLine"] = "/usr/bin/" + evidence["SubmitLine"]
    assert recovery.run("liquid_gpfs_storage", "11739854")["status"] == "preview"
