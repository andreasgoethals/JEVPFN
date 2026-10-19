import json

import pytest

from src.cluster import freeze
from src.utils import paths


def test_source_or_config_changes_stop_frozen_jobs_but_not_notebook_output(
    isolated_output, tmp_path, monkeypatch
):
    repo = tmp_path / "fixture_repo"
    for name, content in (
        ("src/example.py", "x=1\n"),
        ("config/example.yaml", "seed: 1\n"),
        (
            "notebooks/example.ipynb",
            json.dumps({"cells": [{"cell_type": "code", "source": ["print(1)"], "outputs": []}]}),
        ),
    ):
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    monkeypatch.setattr(paths, "REPO_ROOT", repo)
    assert freeze.check_if_present()["frozen"] is False
    original = freeze.record()
    nb = repo / "notebooks/example.ipynb"
    obj = json.loads(nb.read_text())
    obj["cells"][0]["outputs"] = [{"text": "new run output"}]
    nb.write_text(json.dumps(obj))
    assert freeze.check_if_present()["source_sha256"] == original["source_sha256"]
    (repo / "config/jev.local.yaml").write_text("private_settings: 1")
    assert freeze.check_if_present()["frozen"]
    (repo / "src/example.py").write_text("x=2")
    with pytest.raises(ValueError, match="src/example.py"):
        freeze.check_if_present()
    with pytest.raises(ValueError):
        freeze.record()


def test_local_vsc_revision_must_match(isolated_output, tmp_path, monkeypatch):
    fake = tmp_path / "local.json"
    fake.write_text(json.dumps({"source_sha256": "different", "git_head": "different"}))
    with pytest.raises(ValueError, match="Local and VSC"):
        freeze.record(expected_file=fake)


def test_model_runtime_receipts_cannot_change_after_freezing(isolated_output):
    marker = paths.checkpoints_dir() / "decision_models/cloudflare/_JEVPFN_CHECKPOINT.json"
    marker.parent.mkdir(parents=True)
    marker.write_text('{"checkpoint":"first"}')
    freeze.record()
    marker.write_text('{"checkpoint":"changed"}')
    with pytest.raises(ValueError, match="runtime receipts changed"):
        freeze.check_if_present()


def test_reviewed_stage_completion_retains_receipt_before_configuration_changes(
    isolated_output, monkeypatch
):
    from src.cluster import maintenance

    monkeypatch.setattr(maintenance, "require_idle_vsc", lambda: None)
    original = freeze.record()
    with pytest.raises(ValueError, match="stopped writers"):
        freeze.finish_stage(reviewed_by="Andreas")
    assert freeze.check_if_present()["frozen"]
    result = freeze.finish_stage(reviewed_by="Andreas", writers_stopped=True)
    from pathlib import Path

    completed = json.loads(Path(result["completed_receipt"]).read_text())
    assert completed["frozen_receipt"]["source_sha256"] == original["source_sha256"]
    assert completed["reviewed_by"] == "Andreas"
    assert freeze.check_if_present()["frozen"] is False
    assert freeze.record()["frozen"]  # The next stage can be frozen without cleaning output.


def test_stage_completion_cannot_hide_changed_runtime_receipts(isolated_output, monkeypatch):
    from src.cluster import maintenance

    monkeypatch.setattr(maintenance, "require_idle_vsc", lambda: None)
    freeze.record()
    marker = paths.checkpoints_dir() / "decision_models/intern/_JEVPFN_CHECKPOINT.json"
    marker.parent.mkdir(parents=True)
    marker.write_text('{"checkpoint":"unexpected addition"}')
    with pytest.raises(ValueError, match="runtime receipts changed"):
        freeze.finish_stage(reviewed_by="Andreas", writers_stopped=True)
    assert freeze.receipt_path().is_file()


def test_windows_linux_source_endings_match_but_content_changes_do_not(
    isolated_output, tmp_path, monkeypatch
):
    repo = tmp_path / "checkout"
    source = repo / "src/example.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"x = 1\n# same source\n")
    monkeypatch.setattr(paths, "REPO_ROOT", repo)
    original = freeze.source_snapshot()
    expected = tmp_path / "local_receipt.json"
    expected.write_text(json.dumps(original))
    source.write_bytes(b"x = 1\r\n# same source\r\n")
    assert freeze.source_snapshot() == original
    assert freeze.record(expected_file=expected)["frozen"]
    source.write_bytes(b"x = 2\r\n# changed source\r\n")
    with pytest.raises(ValueError, match="src/example.py"):
        freeze.check_if_present()


def test_crlf_shell_scripts_are_rejected_instead_of_normalized(tmp_path):
    shell = tmp_path / "job.slurm"
    shell.write_bytes(b"#!/bin/bash\r\nexit 0\r\n")
    with pytest.raises(ValueError, match="LF line endings"):
        freeze.source_hash(shell)
