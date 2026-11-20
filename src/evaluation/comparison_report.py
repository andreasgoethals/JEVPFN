"""A1 (exp01_outputs): publish the CPU analysis once as `comparison.zip`; render it locally.

The Mindwell CPU job (`scripts/slurm/rq1_outputs.slurm`) reads every complete published
feature store on GPFS PROJECT and writes one compact ZIP with the label-free comparison
tables, the output-shape, pairwise and leave-one-out consensus tables, the joint-request
stand-alone quality/calibration tables (labels used for scoring only) and the model profile
(compute time from the Slurm ledger). Row-level data stay on PROJECT.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from src.evaluation import study_export
from src.evaluation.decision_comparison import Comparison, compare_saved_features
from src.evaluation.decision_diagnostics import inspect_cases, peer_majority_agreement
from src.evaluation.zero_shot import direct_scores
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
    "output_shape",
    "level_histogram",
    "loo_consensus",
    "quality",
    "reliability",
    "model_profile",
)
SCHEMA = "jevpfn.rq1a.report.v2"


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
    return study_export.write_export(
        target, {name: result.table(name) for name in TABLES}, schema=SCHEMA,
        identity=analysis_identity(cfg), origin=origin, extra={"joint_rows": joint_rows})


def read_report(cfg, target=None, *, allow_previous_roster=False):
    """Verified tables, or an explicit waiting state (absent or superseded-schema ZIP)."""
    target = report_path() if target is None else Path(target)
    result = Comparison()
    frames, meta = study_export.read_export(target, schema=SCHEMA, tables=TABLES)
    if meta["status"] != "verified_export":
        result.frames["inventory"] = [pd.DataFrame(columns=["provider", "status"])]
        return result, {**meta, "status": "awaiting_cpu_report" if meta["status"] == study_export.AWAITING
                        else meta["status"]}
    result.frames.update({name: [frame] for name, frame in frames.items()})
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
    return result, {**meta, "status": status, "unassessed_additions": pending}


def model_profile(cfg):
    """Model properties plus successful GPU-hours per 1,000 distinct requests (compute only)."""
    from src.evaluation import model_properties

    def read(path):
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    ledger = read(paths.outputs_dir() / cfg["comparison"]["compute_ledger"])
    audit = read(paths.manifests_dir("experiment_0") / "before_extraction" / "token_audit.json")
    compute = model_properties.compute_per_thousand(ledger, audit, cfg["comparison"]["providers"])
    return model_properties.profile_table(compute)


def build():
    study_export.require_mindwell_cpu("A1 outputs")
    cfg = report_config()
    target = report_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(target.with_suffix(".lock")):
        result = compare_saved_features(cfg, workers=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
        if result.table("pairs").empty:
            raise ValueError("At least two complete published providers are required for RQ1A")
        inspect_cases(result)
        result.frames["peer_majority"] = [peer_majority_agreement(result)]
        quality, reliability = direct_scores(cfg=cfg)
        result.frames["quality"] = [quality]
        result.frames["reliability"] = [reliability]
        result.frames["model_profile"] = [model_profile(cfg)]
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
            origin=study_export.job_origin(),
            joint_rows={"name": name, "rows": len(rows), "sha256": checksum},
        )
        print(f"{target}  SHA256 {file_sha256(target)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    build()


# ---- Notebook views of the verified export (compact tables only; nothing is recomputed) ----

def dataset_mode_table(table, metrics):
    """Per model × dataset × request mode: distinct inputs, text fields with equal weight."""
    if table.empty:
        return pd.DataFrame(columns=["task", "dataset", "representation", "provider", *metrics])
    present = [m for m in metrics if m in table]
    frame = table[table.weighting.eq("unique_inputs")]
    return (frame.groupby(["task", "dataset", "representation", "provider"], sort=False)[present]
            .mean().reset_index())


def equal_dataset_table(table, metrics):
    """Per task, mode and model: equal dataset weight over `dataset_mode_table` rows."""
    frame = dataset_mode_table(table, metrics)
    if frame.empty:
        return frame
    present = [m for m in metrics if m in frame]
    out = frame.groupby(["task", "representation", "provider"], sort=False)[present].mean()
    out["datasets"] = frame.groupby(["task", "representation", "provider"], sort=False).size()
    return out.reset_index()


def pairwise_table(pairs):
    """Per task and mode, every model pair with equal dataset weight."""
    from src.evaluation.decision_comparison import dataset_balanced

    if pairs.empty or "top_agreement" not in pairs:
        return pd.DataFrame()
    frame = pairs[pairs.weighting.eq("unique_inputs")]
    tables = [dataset_balanced(rows, ["model_a", "model_b"], ["top_agreement", "total_variation", "earth_movers"])
              .assign(task=task, representation=mode)
              for (task, mode), rows in frame.groupby(["task", "representation"], sort=False)]
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


OUTPUT_SHAPE_METRICS = ["mean_top_probability", "confident_share", "mean_normalized_entropy",
                        "mean_level_std", "expected_level_mean", "expected_level_median",
                        "expected_level_q1", "expected_level_q3"]
CONSENSUS_METRICS = ["top_agreement", "distance", "expected_level_difference", "coverage"]


def describe_receipt(receipt):
    """The notebook's one-line export identity, or its explicit waiting state."""
    if receipt["status"] == "awaiting_cpu_report":
        return study_export.describe({"status": study_export.AWAITING,
                                      "path": receipt.get("path", str(report_path()))})
    return study_export.describe(receipt)
