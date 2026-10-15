"""Storage setup needs readable immutable inputs and writable destinations."""

import json
from pathlib import Path

import pytest

from src.cluster import setup
from src.utils import paths


@pytest.fixture
def storage(isolated_output):
    # The allocation exists before setup; setup must never create a missing mount.
    paths.staging_root().mkdir()
    return isolated_output


def test_existing_raw_inputs_are_never_write_probed(storage, monkeypatch):
    raw = paths.raw_dir()
    raw.mkdir(parents=True)
    dataset = raw / "example.csv"
    dataset.write_bytes(b"text,target\nexample,1\n")
    original = dataset.read_bytes()
    probed = []
    real_probe = paths.resolve_writable

    def reject_raw_writes(directory):
        probed.append(directory)
        if directory == raw:
            raise PermissionError("Immutable raw inputs reject temporary files")
        return real_probe(directory)

    monkeypatch.setattr(paths, "resolve_writable", reject_raw_writes)
    report = setup.setup_storage("wice")

    assert raw not in probed
    assert dataset.read_bytes() == original
    assert list(raw.iterdir()) == [dataset]
    assert report["access_verified"]["raw_datasets"] == "directory_read"
    assert report["raw_dataset_contents_verified"] is False
    writable = {Path(value) for key, value in report["paths"].items() if key != "raw_datasets"}
    assert set(probed) == writable
    assert all(
        report["access_verified"][key] == "write"
        for key in report["paths"]
        if key != "raw_datasets"
    )
    assert report["quota_verified"] is False
    assert report["backup_verified"] is False
    assert report["jobs_submitted"] is False
    receipt = paths.manifests_dir("experiment_0") / "before_extraction/storage_setup.json"
    assert json.loads(receipt.read_text(encoding="utf-8")) == report


def test_first_setup_creates_an_empty_raw_destination(storage):
    raw = paths.raw_dir()
    assert not raw.exists()
    report = setup.setup_storage("wice")
    assert raw.is_dir()
    assert list(raw.iterdir()) == []
    assert report["raw_dataset_contents_verified"] is False


def test_unreadable_raw_directory_still_stops_setup(storage, monkeypatch):
    raw = paths.raw_dir()
    raw.mkdir(parents=True)

    def deny_read(directory):
        assert directory == raw
        raise PermissionError("Synthetic raw directory read failure")

    monkeypatch.setattr(setup.os, "scandir", deny_read)
    with pytest.raises(PermissionError, match="raw_datasets.*directory_read"):
        setup.setup_storage("wice")
    assert not paths.outputs_dir().exists()


@pytest.mark.parametrize("destination", ["prepared_questions", "small_logs", "checkpoints"])
def test_unwritable_destinations_still_fail_without_fallback(storage, monkeypatch, destination):
    directory = {
        "prepared_questions": paths.decision_inputs_dir(),
        "small_logs": paths.logs_dir("experiment_0"),
        "checkpoints": paths.checkpoints_dir(),
    }[destination]
    real_probe = paths.resolve_writable

    def deny_write(preferred):
        if preferred == directory:
            raise PermissionError("Synthetic output permission failure")
        return real_probe(preferred)

    monkeypatch.setattr(paths, "resolve_writable", deny_write)
    with pytest.raises(PermissionError, match=f"{destination}.*write"):
        setup.setup_storage("wice")
    receipt = paths.manifests_dir("experiment_0") / "before_extraction/storage_setup.json"
    assert not receipt.exists()


def test_missing_allocation_stops_before_creating_any_tree(isolated_output):
    with pytest.raises(ValueError, match="allocation parent must already exist"):
        setup.setup_storage("wice")
    assert not paths.staging_root().exists()
    assert not paths.outputs_dir().exists()


def test_setup_requires_vsc():
    with pytest.raises(RuntimeError, match="Open OnDemand"):
        setup.setup_storage("wice")
