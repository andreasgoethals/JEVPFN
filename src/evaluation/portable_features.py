"""Read-only notebook inputs copied from completed VSC feature publications."""

import json
import shutil
import sqlite3
import tempfile
import zipfile
from contextlib import closing
from pathlib import Path

import pandas as pd

from src.data.loaders import load_catalog
from src.data.selection import selected_entries
from src.jev.features import feature_inventory
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.locking import file_lock
from src.utils.serialization import file_sha256


def portable_path(dataset_id):
    return paths.jev_cache_path("decision_features") / f"{paths.output_component(dataset_id)}.sqlite3"


def notebook_inventory(dataset_id):
    canonical = paths.dataset_cache_path(dataset_id)
    portable = portable_path(dataset_id)
    tables = [feature_inventory(canonical).assign(feature_store=str(canonical), portable=False)]
    if portable.is_file():
        tables.append(feature_inventory(portable).assign(feature_store=str(portable), portable=True))
    return pd.concat(tables, ignore_index=True).drop_duplicates("artifact_id", keep="first")


def verified_payload_hashes(path, keys):
    """Load semantic digests verified against actual request bytes during export."""
    keys = list(keys)
    hashes = {}
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        for offset in range(0, len(keys), 400):
            batch = keys[offset:offset + 400]
            placeholders = ",".join("?" for _ in batch)
            for key, value in db.execute(
                f"SELECT request_key,payload_sha256 FROM payload_hashes WHERE request_key IN ({placeholders})",
                batch,
            ):
                if not isinstance(value, bytes) or len(value) != 32:
                    raise ValueError("Malformed portable payload digest")
                hashes[key] = value.hex()
    if hashes.keys() != set(keys):
        raise ValueError("Portable feature artifact is missing verified request digests")
    return hashes


def _include_previous_blocks(candidate, previous):
    """Merge into a staged store so a failed import leaves the installed copy intact."""
    with closing(sqlite3.connect(candidate.resolve().as_uri(), uri=True)) as db, db:
        db.execute("ATTACH DATABASE ? AS previous", (previous.resolve().as_uri() + "?mode=ro",))
        for table, key, values in (
            ("feature_artifacts", "artifact_id", ("manifest_json", "features_parquet", "requests_parquet")),
            ("payload_hashes", "request_key", ("payload_sha256",)),
        ):
            mismatch = " OR ".join(f"old.{column} IS NOT new.{column}" for column in values)
            if db.execute(
                f"SELECT 1 FROM previous.{table} old JOIN main.{table} new USING ({key}) "
                f"WHERE {mismatch} LIMIT 1"
            ).fetchone():
                raise ValueError(f"Conflicting portable records in {table}")
            db.execute(
                f"INSERT INTO main.{table} SELECT old.* FROM previous.{table} old "
                f"WHERE NOT EXISTS (SELECT 1 FROM main.{table} new WHERE new.{key}=old.{key})"
            )


def import_bundle(bundle, expected_sha256, *, add_providers=False):
    """Verify a complete snapshot before publishing its per-dataset notebook stores."""
    if paths.on_vsc():
        raise RuntimeError("Portable decision feature import is for the local notebooks")
    bundle = Path(bundle)
    if file_sha256(bundle) != expected_sha256:
        raise ValueError("Feature bundle checksum differs from the VSC export receipt")
    expected = {
        e["id"] for e in selected_entries(load_catalog(), load_yaml("feature_creation"))
    }
    root = paths.jev_cache_path("decision_features")
    root.mkdir(parents=True, exist_ok=True)
    with (
        file_lock(root / ".import.lock"),
        tempfile.TemporaryDirectory(dir=root, prefix=".import-") as staging,
        zipfile.ZipFile(bundle) as archive,
    ):
        report = json.loads(archive.read("manifest.json"))
        datasets = report["datasets"]
        providers = set(report["providers"])
        members = archive.namelist()
        if (
            report["schema"] != "jevpfn.decision-features.v1"
            or not providers
            or set(report["jobs"]) != providers
            or any(len(set(ids)) != len(expected) for ids in report["jobs"].values())
            or len(datasets) != len(expected)
            or {r["dataset_id"] for r in datasets} != expected
            or len(members) != len(datasets) + 1
            or set(members) != {"manifest.json", *(f"{d}.sqlite3" for d in expected)}
        ):
            raise ValueError("Incomplete decision feature bundle")
        checked = []
        for row in datasets:
            destination = portable_path(row["dataset_id"])
            if row["path"] != destination.name or archive.getinfo(row["path"]).file_size != row["bytes"]:
                raise ValueError("Feature bundle file identity differs")
            candidate = Path(staging) / destination.name
            with archive.open(row["path"]) as source, candidate.open("wb") as target:
                shutil.copyfileobj(source, target)
            if file_sha256(candidate) != row["sha256"]:
                raise ValueError("Portable dataset store checksum differs")
            inventory = feature_inventory(candidate)
            if (
                set(inventory.artifact_id) != set(row["artifacts"])
                or set(inventory.dataset_id) != {row["dataset_id"]}
                or len(inventory) != 2 * len(providers)
                or set(zip(inventory.provider, inventory.representation, strict=True))
                != {(p, rep) for p in providers for rep in ("joint", "per_column")}
            ):
                raise ValueError("Portable dataset does not contain the entire provider roster")
            if destination.exists():
                previous = feature_inventory(destination)
                if add_providers:
                    identities = pd.concat([previous, inventory], ignore_index=True).drop_duplicates("artifact_id")
                    if identities.duplicated(["provider", "representation"]).any():
                        raise ValueError("Incoming snapshot would mix versions of an installed provider")
                    _include_previous_blocks(candidate, destination)
                    if set(feature_inventory(candidate).artifact_id) != set(identities.artifact_id):
                        raise ValueError("Portable feature union is incomplete")
                elif not set(previous.artifact_id) <= set(inventory.artifact_id):
                    raise ValueError("Incoming snapshot would drop previously downloaded artifacts")
            checked.append((candidate, destination))
        for candidate, destination in checked:
            candidate.replace(destination)
    return report
