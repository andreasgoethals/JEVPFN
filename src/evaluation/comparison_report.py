"""Publish the RQ1A CPU analysis once and render its small verified tables locally."""

import argparse
import hashlib
import io
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from src.evaluation.decision_comparison import Comparison, compare_saved_features
from src.evaluation.decision_diagnostics import inspect_cases, peer_majority_agreement
from src.utils import paths
from src.utils.config import load_config, load_yaml
from src.utils.files import atomic_replace
from src.utils.locking import file_lock
from src.utils.serialization import digest, file_sha256

TABLES = (
    "inventory",
    "coverage",
    "pairs",
    "consensus",
    "distributions",
    "row_summary",
    "vote_patterns",
    "cases",
    "peer_majority",
)
SCHEMA = "jevpfn.rq1a.report.v1"


def report_config():
    cfg = load_config("feature_comparison")
    pins = load_yaml("experiment_1/main")["dataset_feature_artifacts"]
    for dataset, providers in pins.items():
        for provider, modes in providers.items():
            specific = (
                cfg["comparison"]["artifacts"].setdefault(dataset, {}).setdefault(provider, {})
            )
            for mode, identity in modes.items():
                if specific.setdefault(mode, identity) != identity:
                    raise ValueError("RQ1A and RQ2 select different decision features")
    return cfg


def analysis_identity(cfg):
    # API prices and notebook-only settings do not change label-free comparisons.
    return digest({key: cfg.get(key) for key in ("catalog", "dataset_selection", "comparison")})


def report_path():
    return paths.results_dir("outputs", "comparison.zip", phase="rq1")


def write_report(result, cfg, target, *, origin, joint_rows):
    """Atomically publish a complete report. Member hashes also bind scientific scope."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as work:
        candidate = Path(work) / target.name
        with ZipFile(candidate, "w", compression=ZIP_DEFLATED) as out:
            hashes = {}
            for name in TABLES:
                data = result.table(name).to_parquet(index=False)
                out.writestr(f"{name}.parquet", data)
                hashes[name] = hashlib.sha256(data).hexdigest()
            meta = dict(
                schema=SCHEMA,
                config_sha256=analysis_identity(cfg),
                origin=origin,
                created_utc=datetime.now(UTC).isoformat(),
                members=hashes,
                joint_rows=joint_rows,
            )
            out.writestr("manifest.json", json.dumps({**meta, "receipt_sha256": digest(meta)}))
        read_report(cfg, candidate)
        atomic_replace(candidate, target)


def read_report(cfg, target=None, *, allow_previous_roster=False):
    target = report_path() if target is None else Path(target)
    result = Comparison()
    if not target.exists():
        result.frames["inventory"] = [pd.DataFrame(columns=["provider", "status"])]
        return result, {"status": "awaiting_cpu_report"}
    with ZipFile(target) as archive:
        meta = json.loads(archive.read("manifest.json"))
        receipt = meta.pop("receipt_sha256")
        names = ["manifest.json", *(f"{table}.parquet" for table in TABLES)]
        if (
            meta.get("schema") != SCHEMA
            or digest(meta) != receipt
            or set(meta["members"]) != set(TABLES)
            or sorted(archive.namelist()) != sorted(names)
        ):
            raise ValueError("Comparison report configuration, receipt or member list differs")
        for name in TABLES:
            data = archive.read(f"{name}.parquet")
            if hashlib.sha256(data).hexdigest() != meta["members"][name]:
                raise ValueError(f"Comparison report checksum differs for {name}")
            result.frames[name] = [pd.read_parquet(io.BytesIO(data))]
    status = "verified_snapshot"
    pending = []
    if meta.get("config_sha256") != analysis_identity(cfg):
        from copy import deepcopy

        if not allow_previous_roster:
            raise ValueError("Comparison report configuration differs")
        previous = set(result.table("inventory").provider)
        configured = cfg["comparison"]["providers"]
        historical = deepcopy(cfg)
        historical["comparison"]["providers"] = [p for p in configured if p in previous]
        if (not previous < set(configured)
                or analysis_identity(historical) != meta.get("config_sha256")):
            raise ValueError("Comparison report configuration differs")
        pending = [p for p in configured if p not in previous]
        status = "verified_previous_roster"
    return result, {**meta, "status": status, "unassessed_additions": pending, "receipt_sha256": receipt}


def build():
    if (
        not paths.on_vsc()
        or os.environ.get("SLURM_CLUSTER_NAME") != "mindwell"
        or not os.environ.get("SLURM_JOB_ID")
        or not paths.native_gpfs_project(paths.durable_project_override())
        or paths.staging_root() != paths.durable_project_override()
    ):
        raise RuntimeError(
            "Build the RQ1A comparison on a Mindwell CPU allocation reading native GPFS PROJECT"
        )
    cfg = report_config()
    target = report_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(target.with_suffix(".lock")):
        result = compare_saved_features(cfg, workers=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
        if result.table("pairs").empty:
            raise ValueError("At least two complete published providers are required for RQ1A")
        inspect_cases(result)
        result.frames["peer_majority"] = [peer_majority_agreement(result)]
        rows = result.table("row_diagnostics")
        with tempfile.TemporaryDirectory(dir=target.parent) as work:
            candidate = Path(work) / "joint_rows.parquet"
            rows.to_parquet(candidate, index=False)
            checksum = file_sha256(candidate)
            name = f"joint_rows_{checksum}.parquet"
            atomic_replace(candidate, target.parent / name)
        write_report(
            result,
            cfg,
            target,
            origin=f"mindwell:{os.environ['SLURM_JOB_ID']}",
            joint_rows={"name": name, "rows": len(rows), "sha256": checksum},
        )
        print(f"{target}  SHA256 {file_sha256(target)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    build()
