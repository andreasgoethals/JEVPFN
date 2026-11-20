"""One compact, checksummed ZIP per Group A experiment (A1 outputs, A2 stability, A3 geometry).

Raw responses, features and row-level data stay on VSC PROJECT. A Mindwell CPU job builds
the ZIP (Parquet tables + ``manifest.json`` with member SHA-256 values and a receipt); the
owner copies only that ZIP to the same relative path under the local ``output_JEVPFN`` and
the notebook reads nothing else. An absent ZIP is an explicit waiting state.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from src.utils import paths
from src.utils.files import atomic_replace
from src.utils.serialization import digest, file_sha256

#: Well under the 200 MB transfer target; a larger export is refused, not truncated.
MAX_EXPORT_BYTES = 150 * 1024 * 1024
AWAITING = "awaiting_export"


def require_mindwell_cpu(study: str) -> None:
    """Exports read complete PROJECT stores, so they run only inside a Mindwell CPU job."""
    if (
        not paths.on_vsc()
        or os.environ.get("SLURM_CLUSTER_NAME") != "mindwell"
        or not os.environ.get("SLURM_JOB_ID")
        or not paths.native_gpfs_project(paths.durable_project_override())
        or paths.staging_root() != paths.durable_project_override()
    ):
        raise RuntimeError(
            f"Build the {study} export on a Mindwell CPU allocation reading native GPFS PROJECT"
        )


def job_origin() -> str:
    return f"mindwell:{os.environ['SLURM_JOB_ID']}"


def write_export(target, tables: dict[str, pd.DataFrame], *, schema: str, identity: str,
                 origin: str, extra: dict | None = None, max_bytes: int = MAX_EXPORT_BYTES) -> Path:
    """Atomically publish a complete export, verified by reading it back first."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as work:
        candidate = Path(work) / target.name
        with ZipFile(candidate, "w", compression=ZIP_DEFLATED) as out:
            hashes = {}
            for name, frame in tables.items():
                data = frame.to_parquet(index=False)
                out.writestr(f"{name}.parquet", data)
                hashes[name] = hashlib.sha256(data).hexdigest()
            meta = dict(schema=schema, config_sha256=identity, origin=origin,
                        created_utc=datetime.now(UTC).isoformat(), members=hashes, **(extra or {}))
            out.writestr("manifest.json", json.dumps({**meta, "receipt_sha256": digest(meta)}))
        if candidate.stat().st_size > max_bytes:
            raise ValueError(f"{target.name} is {candidate.stat().st_size:,} bytes; the limit is {max_bytes:,}")
        read_export(candidate, schema=schema, tables=tuple(tables))
        atomic_replace(candidate, target)
    return target


def read_export(target, *, schema: str, tables, identity: str | None = None):
    """Verified tables and receipt, or an explicit waiting state when the ZIP is absent.

    Returns ``(frames, receipt)``. A ZIP with another schema is identified but not read.
    """
    target = Path(target)
    if not target.exists():
        return {}, {"status": AWAITING, "path": str(target)}
    with ZipFile(target) as archive:
        meta = json.loads(archive.read("manifest.json"))
        found = {"path": str(target), **file_identity(target), "schema": meta.get("schema"),
                 "origin": meta.get("origin"), "created_utc": meta.get("created_utc")}
        if meta.get("schema") != schema:
            return {}, {**found, "status": "superseded_schema", "expected_schema": schema}
        receipt = meta.pop("receipt_sha256")
        names = ["manifest.json", *(f"{table}.parquet" for table in tables)]
        if (
            digest(meta) != receipt
            or set(meta["members"]) != set(tables)
            or sorted(archive.namelist()) != sorted(names)
        ):
            raise ValueError(f"{target.name}: receipt or member list differs")
        frames = {}
        for name in tables:
            data = archive.read(f"{name}.parquet")
            if hashlib.sha256(data).hexdigest() != meta["members"][name]:
                raise ValueError(f"{target.name}: checksum differs for {name}")
            frames[name] = pd.read_parquet(io.BytesIO(data))
    if identity is not None and meta.get("config_sha256") != identity:
        raise ValueError(f"{target.name}: configuration differs from this checkout")
    return frames, {**meta, **found, "status": "verified_export", "receipt_sha256": receipt}


def file_identity(target) -> dict:
    target = Path(target)
    return {"sha256": file_sha256(target), "bytes": target.stat().st_size}


def describe(receipt: dict) -> str:
    """One line for the notebook: which ZIP was read, or what it is waiting for."""
    if receipt.get("status") == AWAITING:
        return f"Waiting for export: {receipt['path']} is absent. No measurements are shown."
    line = (f"{Path(receipt['path']).name}: created {receipt.get('created_utc')} by {receipt.get('origin')}, "
            f"{receipt.get('bytes', 0):,} bytes, SHA-256 {receipt.get('sha256')}")
    if receipt.get("status") == "superseded_schema":
        return (f"{line}. Schema {receipt.get('schema')} predates {receipt['expected_schema']}; "
                "waiting for a new export. No measurements are shown.")
    return line
