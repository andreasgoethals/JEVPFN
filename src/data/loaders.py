"""Load the official, already-curated MulTaBench core text releases without model packages.

The template's storage constraints are preserved:

1. **Never build a path.** Ask `src.utils.paths`, so one call works on a laptop and on both
   cluster tiers.
2. **`data/raw/` is read-only**, and a cache is valid only once its marker file exists — write
   that marker LAST, so a run killed halfway leaves a cache correctly treated as absent. A
   half-written cache that looks complete is a wrong result nobody investigates.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd
import yaml

from src.data.metadata import TaskMetadata, extract_metadata
from src.data.official_types import official_text_columns
from src.utils import paths
from src.utils.serialization import digest, file_sha256, write_json


@dataclass
class LoadedDataset:
    frame: pd.DataFrame
    metadata: TaskMetadata
    official: dict
    source: dict
    issues: list[str]

    @property
    def features(self) -> pd.DataFrame:
        return self.frame.drop(columns=[self.metadata.target])


def load_catalog(name: str = "exploration/datasets") -> dict:
    with paths.config_path(name).open(encoding="utf-8") as handle:
        catalog = yaml.safe_load(handle)
    if len(catalog["datasets"]) != 20 or any("_TEXT_" not in d["id"] for d in catalog["datasets"]):
        raise ValueError("Expected exactly the 20 core TEXT datasets.")
    if len({d["directory"] for d in catalog["datasets"]}) != 20:
        raise ValueError("Each dataset needs a unique numbered local directory.")
    return catalog


def download_dataset(entry: dict, *, allow_download: bool) -> dict:
    """Retrieve the official public Kaggle archive at the catalog's explicit version.

    No authentication, upstream Python execution, model import or curation is involved.
    A marker is written last, after member hashes. Raw completed files are never rewritten.
    """
    slug, version = entry["slug"], entry["version"]
    directory = entry.get("directory", slug)
    marker = paths.dataset_file(directory, "download.json")
    if marker.is_file():
        info = json.loads(marker.read_text(encoding="utf-8"))
        if info.get("version") != version or info.get("kaggle_ref") != f"chico89/{slug}":
            raise ValueError(f"Cached source/version differs from the configured pin for {slug}.")
        for name, expected in info["sha256"].items():
            if file_sha256(paths.dataset_file(directory, name)) != expected:
                raise ValueError(f"Corrupt raw file for {slug}: {name}; restore it before reuse.")
        return info
    if not allow_download:
        raise FileNotFoundError(
            f"{slug} v{version} is not cached; enable allow_download or prepare it."
        )
    url = f"https://www.kaggle.com/api/v1/datasets/download/chico89/{slug}?datasetVersionNumber={version}"
    folder = marker.parent
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=folder, prefix="download_") as temporary:
        # This is an input download, not an inference endpoint.
        archive = paths.Path(temporary) / "archive.zip"
        with urllib.request.urlopen(url, timeout=60) as response, archive.open("wb") as out:
            shutil.copyfileobj(response, out)
        with zipfile.ZipFile(archive) as zf:
            wanted = ("data.csv", "metadata.json")
            if not set(wanted) <= set(zf.namelist()):
                raise ValueError(f"Unexpected archive layout for {slug}.")
            hashes = {}
            for name in wanted:
                extracted = paths.Path(temporary) / name
                with zf.open(name) as incoming, extracted.open("wb") as out:
                    shutil.copyfileobj(incoming, out)
                hashes[name] = file_sha256(extracted)
                target = paths.dataset_file(directory, name)
                if target.exists():
                    if file_sha256(target) != hashes[name]:
                        raise ValueError(f"Refusing to overwrite different raw content: {target}")
                else:
                    os.replace(extracted, target)
        info = {
            "url": url,
            "kaggle_ref": f"chico89/{slug}",
            "version": version,
            "retrieved_at": datetime.now(UTC).isoformat(),
            "sha256": hashes,
            "archive_sha256": file_sha256(archive),
        }
        write_json(marker, info)
    return info


def load_dataset(entry: dict, *, allow_download: bool = False) -> LoadedDataset:
    source = download_dataset(entry, allow_download=allow_download)
    if "sha256" in entry and source["sha256"] != entry["sha256"]:
        raise ValueError(f"Released files disagree with pinned hashes for {entry['id']}.")
    official = json.loads(
        paths.dataset_file(entry.get("directory", entry["slug"]), "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    # Match the official loader's read_csv defaults; low_memory=False only stabilises inference.
    frame = pd.read_csv(
        paths.dataset_file(entry.get("directory", entry["slug"]), "data.csv"),
        low_memory=False,
    )
    if official.get("image_col") is not None:
        raise ValueError("An image dataset is outside the configured collection.")
    if len(frame) != official["num_rows"] or len(frame.columns) - 1 != official["num_features"]:
        raise ValueError(f"Shape disagrees with official metadata for {entry['id']}.")
    features = frame.drop(columns=[official["target"]])
    # The initial catalog freezes the upstream detector's output, including column order.
    # Subsequent runs use those reviewed names instead of rediscovering a split in each fold.
    text = entry.get("text_columns")
    if text is None:
        text = official_text_columns(features)
    metadata = extract_metadata(
        frame, official, dataset_id=entry["id"], dataset_name=entry["name"], text_columns=text
    )
    issues = []
    expected = entry["paper_summary"]
    for key, observed in {
        "rows": len(frame),
        "text_columns": len(text),
        "non_text_columns": len(metadata.non_text_columns),
    }.items():
        if expected[key] != observed:
            issues.append(f"Paper summary {key}={expected[key]}, observed {observed}.")
    if frame[metadata.target].isna().any():
        issues.append("Missing target values retained in audit; evaluation handling remains open.")
    source = {
        **source,
        "dataset_id": entry["id"],
        "schema_sha256": metadata.fingerprint,
        "data_identity": digest(source["sha256"]),
    }
    return LoadedDataset(frame, metadata, official, source, issues)
