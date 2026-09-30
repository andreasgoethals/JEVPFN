"""Pinned metadata and upstream text rules; no network required in the test suite."""

import json
import subprocess
import sys
from io import BytesIO
from zipfile import ZipFile

import pandas as pd
import pytest

from src.data.loaders import download_dataset, load_catalog, load_dataset
from src.data.metadata import extract_metadata
from src.data.official_types import official_text_columns
from src.utils import paths


def test_pinned_catalog_has_only_twenty_core_text_datasets():
    catalog = load_catalog()
    assert len({d["id"] for d in catalog["datasets"]}) == 20
    assert len(catalog["upstream_commit"]) == 40
    assert all("-full-" not in d["slug"] for d in catalog["datasets"])
    assert all(d["version"] == 1 and d["text_columns"] for d in catalog["datasets"])
    assert all(len(d["sha256"]["data.csv"]) == 64 for d in catalog["datasets"])


def test_official_text_rule_not_all_string_columns():
    x = pd.DataFrame(
        {
            "prose": [f"some writing {i}" for i in range(120)],
            "category": ["A"] * 120,
            "numeric": range(120),
            "numeric_string": [str(i) for i in range(120)],
            "date": pd.date_range("2020-01-01", periods=120).astype(str),
            "empty": [None] * 120,
        }
    )
    assert official_text_columns(x) == ["prose"]


def test_metadata_rejects_unknown_task_and_target_as_text():
    frame = pd.DataFrame({"a": ["one", "two"], "y": [0, 1]})
    for task, text in [("unknown", ["a"]), ("cls", ["y"])]:
        with pytest.raises(ValueError):
            extract_metadata(
                frame,
                {"target": "y", "task_type": task},
                dataset_id="X",
                dataset_name="X",
                text_columns=text,
            )


def test_versioned_download_verifies_hashes_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    contents = BytesIO()
    with ZipFile(contents, "w") as z:
        z.writestr("data.csv", "text,num,y\nfirst,3,no\nsecond,4,yes\n")
        z.writestr(
            "metadata.json",
            json.dumps(
                {
                    "target": "y",
                    "task_type": "cls",
                    "num_rows": 2,
                    "num_features": 2,
                    "num_classes": 2,
                    "image_col": None,
                }
            ),
        )
    urls = []

    def open_url(url, timeout):
        urls.append(url)
        return BytesIO(contents.getvalue())

    monkeypatch.setattr("urllib.request.urlopen", open_url)
    entry = {
        "id": "BIN_TEXT_TEST",
        "name": "Test",
        "slug": "test",
        "version": 7,
        "text_columns": ["text"],
        "paper_summary": {"rows": 2, "text_columns": 1, "non_text_columns": 1},
    }
    dataset = load_dataset(entry, allow_download=True)
    assert dataset.metadata.classes == ("no", "yes")
    assert "datasetVersionNumber=7" in urls[0]
    assert paths.dataset_file("test", "data.csv").parent == tmp_path / "data" / "raw" / "test"
    download_dataset(entry, allow_download=False)
    assert len(urls) == 1
    with pytest.raises(ValueError, match="source/version"):
        download_dataset({**entry, "version": 8}, allow_download=False)
    assert len(urls) == 1
    paths.dataset_file("test", "data.csv").write_text("corrupt")
    with pytest.raises(ValueError, match="Corrupt"):
        download_dataset(entry, allow_download=False)


def test_response_cache_survives_disposable_cleanup_roots(isolated_output):
    from src.utils.clean_run import roots

    cache = paths.jev_cache_path()
    assert not any(cache.is_relative_to(p.resolve()) for p in roots(processed=True))
    with pytest.raises(ValueError):
        paths.jev_cache_path("processed/paid.sqlite3")
    with pytest.raises(ValueError):
        paths.jev_cache_path("raw/paid.sqlite3")
    with pytest.raises(ValueError):
        paths.jev_cache_path("../../outside.sqlite3")


def test_audit_build_lock_excludes_other_process_and_releases_after_exception(tmp_path):
    from src.data.audit import audit_lock

    code = """
import sys
from pathlib import Path
from src.data.audit import audit_lock
try:
    with audit_lock(Path(sys.argv[1]), timeout=.2):
        print('acquired')
except TimeoutError:
    print('blocked')
"""
    with pytest.raises(RuntimeError), audit_lock(tmp_path):
        blocked = subprocess.run(
            [sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, check=True
        )
        assert blocked.stdout.strip() == "blocked"
        raise RuntimeError("simulated failed audit")
    acquired = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, check=True
    )
    assert acquired.stdout.strip() == "acquired"


def test_audit_lock_is_released_when_owner_process_crashes(tmp_path):
    from src.data.audit import audit_lock

    code = """
import os,sys
from pathlib import Path
from src.data.audit import audit_lock
with audit_lock(Path(sys.argv[1])):
    os._exit(23)
"""
    crashed = subprocess.run([sys.executable, "-c", code, str(tmp_path)], check=False)
    assert crashed.returncode == 23
    with audit_lock(tmp_path, timeout=0.2):
        pass
