"""Reusable feature artifacts built ONLY from previously cached, validated responses.

The feature build is independent of experiment numbers and CV folds. A block is
one dataset, one representation and text-only inputs. Targets never
enter its tables. No API calls, imputation, truncation or model fitting happen here.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.audit import audit_lock
from src.jev.provider import MAPPING_VERSION, response_features
from src.jev.requests import build_requests, text_groups
from src.jev.templates import make_question
from src.utils import paths
from src.utils.provenance import code_identity
from src.utils.serialization import digest, file_sha256, write_json


@dataclass
class FeatureBlock:
    values: pd.DataFrame
    requests: pd.DataFrame
    manifest: dict


def block_identity(metadata, data_identity: str, cfg: dict) -> dict:
    if not data_identity:
        raise ValueError("A pinned data identity is required for feature artifacts.")
    if cfg["jev"]["representation"] not in {"per_column", "joint"}:
        raise ValueError("Build per-column and joint blocks separately, then combine them.")
    if cfg["jev"]["include_non_text_features"] is not False:
        raise ValueError("Non-text features are excluded from Jev requests.")
    return {
        "schema_version": 2,
        "missing_text_policy": cfg["jev"]["missing_text_policy"],
        "dataset_id": metadata.dataset_id,
        "data_identity": data_identity,
        "task_schema_sha256": metadata.fingerprint,
        "question": make_question(metadata, scores=cfg["jev"]["regression_scores"]),
        "representation": cfg["jev"]["representation"],
        "include_non_text_features": cfg["jev"]["include_non_text_features"],
        "model": cfg["jev"]["model"],
        "version": cfg["jev"]["version"],
        "request_configuration": cfg["jev"]["request_configuration"],
        "template_version": cfg["jev"]["template_version"],
        "api_mapping_version": MAPPING_VERSION,
        "code_sha256": code_identity(),
    }


def materialize_cached_block(dataset, cfg: dict, cache) -> FeatureBlock:
    """Fail on a missing/invalid response; mock responses can never become model features."""
    identity = block_identity(dataset.metadata, dataset.source.get("data_identity"), cfg)
    mode = identity["representation"]
    prefix = f"jev__{mode}__text_only"
    features = dataset.features
    records, references = [], []
    metadata = dataset.metadata
    keys = (
        ["p_positive"]
        if metadata.task_type == "binary_classification"
        else (
            [f"p_c{i:03d}" for i in range(len(metadata.classes))] + ["confidence"]
            if metadata.task_type == "multiclass_classification"
            else [f"p_{i}" for i in range(9)] + ["confidence", "expected_direction"]
        )
    )
    groups = text_groups(metadata, mode)
    for position in range(len(features)):
        record = {"row_id": f"csv:{position}"}
        requests = build_requests(
            features,
            dataset.metadata,
            position,
            cfg,
            source=dataset.source,
            code_version=identity["code_sha256"],
        )
        by_slot = {json.loads(r.provenance_json)["representation_slot"]: r for r in requests}
        for slot, _columns in enumerate(groups):
            group = f"text_{slot:03d}" if mode == "per_column" else "all_text"
            request = by_slot.get(slot)
            if request is None:
                values = dict.fromkeys(keys, float("nan"))
            else:
                response = cache.get(request, kind="verified_response")
                if response is None:
                    raise ValueError(
                        f"Missing verified response for {metadata.dataset_id}, csv:{position}, slot {slot}."
                    )
                values = response_features(request, response)
            record.update({f"{prefix}__{group}__{key}": value for key, value in values.items()})
            references.append(
                {
                    "row_id": record["row_id"],
                    "group": group,
                    "request_key": request.cache_key if request else None,
                    "status": "verified_response" if request else "missing_text",
                }
            )
        records.append(record)
    values = pd.DataFrame(records)
    manifest = {
        "identity": identity,
        "artifact_id": digest(identity),
        "status": "complete",
        "namespace": "verified_response",
        "row_count": len(values),
        "row_ids_sha256": digest(values.row_id.tolist()),
        "feature_columns": list(values.columns.drop("row_id")),
        "text_columns": list(dataset.metadata.text_columns),
        "class_labels": list(dataset.metadata.classes),
        "regression_index_to_direction": {str(i): i - 4 for i in range(9)}
        if dataset.metadata.task_type == "regression"
        else None,
        "created_at": datetime.now(UTC).isoformat(),
    }
    return FeatureBlock(values, pd.DataFrame(references), manifest)


def save_block(block: FeatureBlock, directory: Path | None = None) -> Path:
    """Publish immutable Parquet tables; manifest is the final completion marker."""
    folder = directory or paths.jev_cache_path(
        f"jev_cache/features/{block.manifest['artifact_id']}"
    )
    folder.mkdir(parents=True, exist_ok=True)
    with audit_lock(folder):
        marker = folder / "manifest.json"
        if marker.exists():
            existing = load_block(folder)
            if (
                existing.manifest["identity"] != block.manifest["identity"]
                or not existing.values.equals(block.values)
                or not existing.requests.equals(block.requests)
            ):
                raise ValueError("Refusing to overwrite a different feature artifact.")
            return folder
        hashes = {}
        for name, frame in (
            ("features.parquet", block.values),
            ("requests.parquet", block.requests),
        ):
            with tempfile.NamedTemporaryFile(dir=folder, suffix=".parquet", delete=False) as f:
                temporary = Path(f.name)
            try:
                frame.to_parquet(temporary, index=False)
                hashes[name] = file_sha256(temporary)
                os.replace(temporary, folder / name)
            finally:
                temporary.unlink(missing_ok=True)
        write_json(marker, {**block.manifest, "sha256": hashes})
    return folder


def load_block(folder: Path) -> FeatureBlock:
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest["status"] != "complete"
        or manifest["namespace"] != "verified_response"
        or digest(manifest["identity"]) != manifest["artifact_id"]
    ):
        raise ValueError("Invalid feature manifest.")
    for name in ("features.parquet", "requests.parquet"):
        if file_sha256(folder / name) != manifest["sha256"][name]:
            raise ValueError("Feature artifact checksum mismatch.")
    values = pd.read_parquet(folder / "features.parquet")
    if (
        len(values) != manifest["row_count"]
        or digest(values.row_id.tolist()) != manifest["row_ids_sha256"]
        or not values.row_id.is_unique
    ):
        raise ValueError("Feature row identity mismatch.")
    if list(values.columns.drop("row_id")) != manifest["feature_columns"]:
        raise ValueError("Feature schema mismatch.")
    requests = pd.read_parquet(folder / "requests.parquet")
    if (
        list(requests.columns) != ["row_id", "group", "request_key", "status"]
        or set(requests.row_id) != set(values.row_id)
        or requests.duplicated(["row_id", "group"]).any()
    ):
        raise ValueError("Invalid row/request correspondence.")
    indexed = values.set_index("row_id")
    for group, refs in requests.groupby("group", sort=False):
        if set(refs.row_id) != set(values.row_id):
            raise ValueError("Incomplete text-group trace.")
        cells = indexed.loc[
            refs.row_id, [c for c in manifest["feature_columns"] if f"__{group}__" in c]
        ]
        missing = refs.status.eq("missing_text").to_numpy()
        if not refs.status.isin(["verified_response", "missing_text"]).all():
            raise ValueError("Invalid request status.")
        if (
            not refs.loc[missing, "request_key"].isna().all()
            or not cells.iloc[missing].isna().all().all()
        ):
            raise ValueError("Missing text must have no request key and all missing features.")
        if (
            refs.loc[~missing, "request_key"].isna().any()
            or not np.isfinite(cells.iloc[~missing].to_numpy()).all()
        ):
            raise ValueError("Verified response must have a key and complete numeric features.")
    return FeatureBlock(values, requests, manifest)


def assemble_features(dataset, blocks: list[FeatureBlock]) -> pd.DataFrame:
    """Join by stable row ID and verify dataset/schema identity; labels remain separate."""
    row_ids = [f"csv:{i}" for i in range(len(dataset.frame))]
    result = dataset.features.loc[:, list(dataset.metadata.non_text_columns)].copy()
    result.index = row_ids
    for block in blocks:
        if (
            block.manifest["status"] != "complete"
            or block.manifest["namespace"] != "verified_response"
        ):
            raise ValueError("Only complete verified-response blocks may be assembled.")
        identity = block.manifest["identity"]
        if (
            identity["dataset_id"] != dataset.metadata.dataset_id
            or identity["data_identity"] != dataset.source["data_identity"]
            or identity["task_schema_sha256"] != dataset.metadata.fingerprint
        ):
            raise ValueError("Feature block belongs to a different dataset version/schema.")
        if not block.values.row_id.is_unique or set(block.values.row_id) != set(row_ids):
            raise ValueError("Feature row IDs are missing, duplicated or unexpected.")
        if set(result.columns) & set(block.manifest["feature_columns"]):
            raise ValueError("Duplicate feature block/column names.")
        if list(block.values.columns.drop("row_id")) != block.manifest["feature_columns"]:
            raise ValueError("Feature schema mismatch.")
        result = result.join(block.values.set_index("row_id"), validate="one_to_one")
    return result


def feature_plan(audit, cfg: dict) -> pd.DataFrame:
    """Inspect the full feature-build scope without enumerating millions of requests."""
    records = []
    for row in audit.summary.itertuples():
        metadata = audit.metadata(row.dataset_id)
        width = (
            1
            if metadata.task_type == "binary_classification"
            else (11 if metadata.task_type == "regression" else len(metadata.classes) + 1)
        )
        for mode in cfg["feature_creation"]["modes"]:
            groups = len(metadata.text_columns) if mode == "per_column" else 1
            with (
                audit_lock(audit.folder),
                np.load(
                    audit.folder / f"{row.dataset_id}_lengths.npz", allow_pickle=False
                ) as lengths,
            ):
                masks = np.stack(
                    [lengths[f"present_{i}"] for i in range(len(metadata.text_columns))]
                )
            active = int(masks.sum()) if mode == "per_column" else int(masks.any(axis=0).sum())
            records.append(
                {
                    "dataset_id": row.dataset_id,
                    "dataset": row.dataset,
                    "mode": mode,
                    "include_non_text_features": False,
                    "rows": row.rows,
                    "potential_slots": row.rows * groups,
                    "requests_before_deduplication": active,
                    "unique_requests_within_mode": getattr(row, f"{mode}_unique_inputs"),
                    "new_requests_in_combined_build": (
                        row.per_column_unique_inputs
                        if mode == "per_column"
                        else row.joint_unique_inputs - row.joint_single_field_unique_overlap
                    ),
                    "empty_slots_skipped": row.rows * groups - active,
                    "retained_numeric_columns": width * groups,
                    "proposed_compact_numeric_columns": groups
                    * (
                        len(metadata.classes)
                        if metadata.task_type == "multiclass_classification"
                        else 1
                    ),
                    "status": "planned_only",
                }
            )
    frame = pd.DataFrame(records)
    folder = paths.results_dir(phase="feature_creation")
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / "feature_plan.csv", index=False)
    return frame


def table_schemas() -> pd.DataFrame:
    """Storage contracts, not fabricated data or fabricated Jev outputs."""
    return pd.DataFrame(
        [
            {
                "table": "Raw source",
                "key": "dataset hash + csv:N",
                "content": "Original CSV and metadata; target remains here",
                "scope": "One immutable dataset version",
            },
            {
                "table": "responses.sqlite3 / responses",
                "key": "request_key + namespace",
                "content": "Exact state/question/settings, raw response, UTC timestamp",
                "scope": "Shared across all experiments",
            },
            {
                "table": "responses.sqlite3 / origins",
                "key": "request_key + origin hash",
                "content": "Dataset, csv:N, text columns, code/config/source hashes",
                "scope": "Many origins may reuse one response",
            },
            {
                "table": "features.parquet",
                "key": "csv:N within artifact_id",
                "content": "One row per source row; numeric Jev columns only",
                "scope": "One dataset and text representation",
            },
            {
                "table": "requests.parquet",
                "key": "csv:N + text group",
                "content": "Request key or missing_text status for each row/group",
                "scope": "Trace feature cells to raw responses",
            },
            {
                "table": "manifest.json",
                "key": "artifact_id",
                "content": "Model, templates, source/schema/code hashes, mappings, file hashes",
                "scope": "Written last; complete artifact marker",
            },
            {
                "table": "Experiment feature matrix",
                "key": "csv:N",
                "content": "Original non-text columns joined with selected cached blocks",
                "scope": "Targets separate; no API work during experiment",
            },
        ]
    )


def feature_summary(plan, totals) -> str:
    return "\n".join(
        [
            "1. Feature-build scope",
            f"{len(plan)} dataset/representation tables planned; no features generated.",
            "2. API body preview",
            "Documentation-verified Noul/Choice/Score mapping; no HTTP client or paid call.",
            "3. Reusable tables",
            "Raw responses -> validated probability blocks + row/request map -> experiment-specific feature joins; targets stay separate.",
            "4. Workload and budget",
            totals.to_string(index=False),
            "5. Execution sequence",
            "Design and local previews -> small approved local Jev test -> full local feature creation and validation -> experiment 0 on VSC -> experiment 1.",
            "Context overflow, live API behaviour and experimental protocol still require review.",
        ]
    )
