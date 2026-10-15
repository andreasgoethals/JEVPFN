"""Question upload is exact, validated before writes, and restartable with backups."""

import json
import sqlite3
import zipfile
from contextlib import closing
from pathlib import Path

import pytest

from src.cluster import input_transfer as transfer
from src.cluster.gpu_contracts import synthetic_datasets
from src.jev.inputs import prepare_dataset_inputs
from src.jev.question_bank import build_question_bank, save_question_bank
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.serialization import file_sha256


@pytest.fixture
def bundle(isolated_output, monkeypatch):
    dataset = next(synthetic_datasets())
    bank = build_question_bank([dataset.metadata], load_yaml("question_design"))
    save_question_bank(bank)
    prepared = prepare_dataset_inputs(dataset, bank, variant="accepted")
    source = Path(prepared["path"])
    monkeypatch.setattr(transfer, "selected_ids", lambda: [dataset.metadata.dataset_id])
    monkeypatch.setattr(paths, "on_vsc", lambda: False)
    destination = paths.large_outputs_dir() / "transfers" / "questions.zip"
    transfer.pack(destination)
    assert transfer.pack(destination) == destination  # Idempotent, unchanged bundle.
    receiver = source.parent.parent / "receiver"
    monkeypatch.setattr(paths, "decision_inputs_dir", lambda: receiver)
    monkeypatch.setattr(transfer, "require_idle_vsc", lambda: None)
    return source, destination, receiver


def test_input_upload_is_byte_exact_and_resumes_without_replacement(bundle):
    source, bundle_path, receiver = bundle
    result = transfer.install(bundle_path, writers_stopped=True)
    target = receiver / source.name
    assert result["status"] == "inputs_installed"
    assert file_sha256(source) == file_sha256(target)
    assert (
        transfer.install(bundle_path, writers_stopped=True)["datasets"][0]["status"]
        == "already_current"
    )


def test_input_upload_preserves_previous_file_before_replace(bundle):
    source, bundle_path, receiver = bundle
    transfer.install(bundle_path, writers_stopped=True)
    target = receiver / source.name
    with closing(sqlite3.connect(target)) as db, db:
        db.execute("CREATE TABLE historical_note(value TEXT)")
        db.execute("INSERT INTO historical_note VALUES ('retain old file')")
    previous_sha = file_sha256(target)
    result = transfer.install(bundle_path, writers_stopped=True)
    backup = Path(result["datasets"][0]["backup"])
    assert file_sha256(backup) == previous_sha
    assert file_sha256(target) == file_sha256(source)


@pytest.mark.parametrize("damage", ["whole_bundle", "member", "traversal", "sidecar"])
def test_invalid_upload_does_not_replace_existing_inputs(bundle, damage):
    source, bundle_path, receiver = bundle
    transfer.install(bundle_path, writers_stopped=True)
    target = receiver / source.name
    before = file_sha256(target)
    if damage == "whole_bundle":
        with bundle_path.open("ab") as handle:
            handle.write(b"damage")
    elif damage in {"member", "traversal"}:
        with zipfile.ZipFile(bundle_path) as archive:
            members = {n: archive.read(n) for n in archive.namelist()}
        if damage == "traversal":
            members["../bad.sqlite3"] = b"bad"
        else:
            manifest = json.loads(members["manifest.json"])
            manifest["files"][0]["sha256"] = "0" * 64
            members["manifest.json"] = json.dumps(manifest).encode()
        with zipfile.ZipFile(bundle_path, "w") as archive:
            for name, data in members.items():
                archive.writestr(name, data)
        bundle_path.with_suffix(".zip.sha256").write_text(
            file_sha256(bundle_path) + "  questions.zip\n"
        )
    else:
        Path(str(target) + "-journal").write_bytes(b"active writer")
    with pytest.raises(ValueError):
        transfer.install(bundle_path, writers_stopped=True)
    assert file_sha256(target) == before


def test_upload_requires_explicit_stopped_writers(bundle):
    with pytest.raises(ValueError, match="writers"):
        transfer.install(bundle[1])
