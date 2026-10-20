"""Reusable feature artifacts built ONLY from previously cached, validated responses.

The feature build is independent of experiment numbers and CV folds. A block is
one dataset, one representation and text-only inputs. Targets never
enter its tables. No API calls, imputation, truncation or model fitting happen here.
"""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.audit import audit_lock
from src.data.loaders import load_catalog
from src.data.selection import selected_entries
from src.jev.provider import MAPPING_VERSION, response_features
from src.jev.requests import build_requests, text_groups
from src.jev.templates import make_question
from src.utils import paths
from src.utils.events import append_event
from src.utils.provenance import code_identity
from src.utils.serialization import digest, file_sha256, write_json


@dataclass
class FeatureBlock:
    values: pd.DataFrame
    requests: pd.DataFrame
    manifest: dict


def block_identity(
    metadata, data_identity: str, cfg: dict, *, question_bank=None, question_variant="accepted"
) -> dict:
    if not data_identity:
        raise ValueError("A pinned data identity is required for feature artifacts.")
    if cfg["jev"]["representation"] not in {"per_column", "joint"}:
        raise ValueError("Build per-column and joint blocks separately, then combine them.")
    if cfg["jev"]["include_non_text_features"] is not False:
        raise ValueError("Non-text features are excluded from Jev requests.")
    identity = {
        "schema_version": 2,
        "missing_text_policy": cfg["jev"]["missing_text_policy"],
        "dataset_id": metadata.dataset_id,
        "data_identity": data_identity,
        "task_schema_sha256": metadata.fingerprint,
        "question": make_question(metadata, scores=cfg["jev"]["regression_scores"]),
        "representation": cfg["jev"]["representation"],
        "include_non_text_features": cfg["jev"]["include_non_text_features"],
        "provider": "jev",
        "model": cfg["jev"]["model"],
        "version": cfg["jev"]["version"],
        "request_configuration": cfg["jev"]["request_configuration"],
        "template_version": cfg["jev"]["template_version"],
        "api_mapping_version": MAPPING_VERSION,
        "code_sha256": code_identity(
            [
                "src/jev/features.py",
                "src/jev/provider.py",
                "src/jev/requests.py",
                "src/jev/question_bank.py",
                "src/jev/templates.py",
                "src/utils/serialization.py",
            ]
        ),
    }
    if question_bank is not None:
        from src.jev.question_bank import validate_bank

        validate_bank(question_bank)
        entry = question_bank["datasets"][metadata.dataset_id]
        if entry["task_schema_sha256"] != metadata.fingerprint:
            raise ValueError("Question bank belongs to a different task schema")
        variant = entry["variants"][question_variant]
        identity.update(
            question_bank_id=question_bank["bank_id"],
            question_variant=question_variant,
            question=variant["question"],
            documentary_definition=variant["documentary_definition"],
            question_design_version=variant["version"],
        )
    return identity


def materialize_cached_block(
    dataset,
    cfg: dict,
    cache,
    *,
    question_bank=None,
    question_variant="accepted",
    provider_settings=None,
    phase="feature_creation",
) -> FeatureBlock:
    """Fail on a missing/invalid response; mock responses can never become model features."""
    identity = block_identity(
        dataset.metadata,
        dataset.source.get("data_identity"),
        cfg,
        question_bank=question_bank,
        question_variant=question_variant,
    )
    if provider_settings is not None:
        identity["settings_sha256"] = digest(provider_settings)
        identity.update(
            {
                k: provider_settings[k]
                for k in (
                    "provider",
                    "model",
                    "version",
                    "request_configuration",
                    "api_mapping_version",
                    "template_version",
                )
            }
        )
    artifact_id = digest(identity)
    event_log = paths.logs_dir(phase) / "builds" / f"{artifact_id}.log"
    started = time.perf_counter()
    append_event(
        event_log,
        "materialization_started",
        dataset_id=identity["dataset_id"],
        representation=identity["representation"],
        rows=len(dataset.frame),
    )
    try:
        block = _materialize_cached_block(
            dataset,
            cfg,
            cache,
            identity,
            event_log,
            question_bank,
            question_variant,
            provider_settings,
        )
    except Exception as exc:
        append_event(
            event_log,
            "materialization_failed",
            error_type=type(exc).__name__,
            seconds=time.perf_counter() - started,
        )
        raise
    append_event(
        event_log,
        "materialization_complete",
        rows=len(block.values),
        unique_requests=int(block.requests.request_key.nunique()),
        missing_slots=int(block.requests.status.eq("missing_text").sum()),
        seconds=time.perf_counter() - started,
    )
    block.manifest["response_store"] = str(cache.path.resolve())
    return block


def _materialize_cached_block(
    dataset,
    cfg,
    cache,
    identity,
    event_log,
    question_bank,
    question_variant,
    provider_settings=None,
):
    mode = identity["representation"]
    prefix = f"{identity['provider']}__{mode}__text_only"
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
    transform = None
    if question_bank is not None:
        from src.jev.question_bank import compile_frozen_question

        transform = compile_frozen_question(metadata, question_bank, question_variant)
    cached_batch = {}
    requests_batch = {}
    for position in range(len(features)):
        if position % 256 == 0:
            requests_batch = {
                p: build_requests(
                    features,
                    dataset.metadata,
                    p,
                    cfg,
                    source=dataset.source,
                    code_version=identity["code_sha256"],
                )
                for p in range(position, min(position + 256, len(features)))
            }
            if transform is not None:
                requests_batch = {
                    p: [transform(r) for r in batch] for p, batch in requests_batch.items()
                }
            if provider_settings is not None:
                from dataclasses import replace

                from src.utils.serialization import canonical_json

                encoded_settings = canonical_json(provider_settings)
                requests_batch = {
                    p: [replace(r, settings_json=encoded_settings) for r in batch]
                    for p, batch in requests_batch.items()
                }
            cached_batch = cache.get_many(
                (r for batch in requests_batch.values() for r in batch), kind="verified_response"
            )
        record = {"row_id": f"csv:{position}"}
        requests = requests_batch[position]
        by_slot = {json.loads(r.provenance_json)["representation_slot"]: r for r in requests}
        for slot, _columns in enumerate(groups):
            group = f"text_{slot:03d}" if mode == "per_column" else "all_text"
            request = by_slot.get(slot)
            if request is None:
                values = dict.fromkeys(keys, float("nan"))
            else:
                response = cached_batch.get(request.cache_key)
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
        if (position + 1) % 10000 == 0:
            append_event(
                event_log, "progress", completed_rows=position + 1, total_rows=len(features)
            )
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


def save_block(
    block: FeatureBlock, directory: Path | None = None, *, phase="feature_creation"
) -> Path:
    """One transactional dataset SQLite file; compressed Parquet tables are embedded."""
    from src.jev.cache import ResponseCache

    identity = block.manifest["identity"]
    artifact_id = block.manifest["artifact_id"]
    path = paths.require_project_output(
        directory or paths.dataset_cache_path(identity["dataset_id"])
    )
    if path.suffix != ".sqlite3":
        raise ValueError("Feature storage must be a dataset .sqlite3 file.")
    _validate_block(block.values, block.requests, block.manifest)
    store = ResponseCache(path, dataset_id=identity["dataset_id"])
    buffers = {}
    for name, frame in (("features.parquet", block.values), ("requests.parquet", block.requests)):
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False, compression="zstd")
        buffers[name] = buffer.getvalue()
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in buffers.items()}
    manifest = {**block.manifest, "sha256": hashes}
    manifest.pop("response_store", None)  # Portable file; no local absolute path in provenance.
    source = Path(block.manifest["response_store"]).resolve()
    with store._connect() as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS feature_artifacts (artifact_id TEXT PRIMARY KEY, manifest_json TEXT NOT NULL, features_parquet BLOB NOT NULL, requests_parquet BLOB NOT NULL)"
        )
        existing = db.execute(
            "SELECT manifest_json FROM feature_artifacts WHERE artifact_id=?", (artifact_id,)
        ).fetchone()
        if existing:
            old = load_block(path, artifact_id)
            if (
                old.manifest["identity"] != identity
                or not old.values.equals(block.values)
                or not old.requests.equals(block.requests)
            ):
                raise ValueError("Refusing to overwrite a different feature artifact.")
        else:
            if source != path.resolve():
                if not source.is_file():
                    raise ValueError(
                        "Response store is missing; cannot publish untraceable features."
                    )
                db.execute("ATTACH DATABASE ? AS incoming", (source.as_uri() + "?mode=ro",))
                db.execute("CREATE TEMP TABLE referenced_keys (request_key TEXT PRIMARY KEY)")
                db.executemany(
                    "INSERT INTO referenced_keys VALUES (?)",
                    ((key,) for key in block.requests.request_key.dropna().unique()),
                )
                conflict = db.execute(
                    "SELECT 1 FROM incoming.responses i JOIN responses r USING(request_key,kind) JOIN referenced_keys k USING(request_key) WHERE i.kind='verified_response' AND i.response_json != r.response_json LIMIT 1"
                ).fetchone()
                if conflict:
                    raise ValueError("Conflicting cached response for identical request.")
                db.execute(
                    "INSERT OR IGNORE INTO responses SELECT i.* FROM incoming.responses i JOIN referenced_keys k USING(request_key) WHERE i.kind='verified_response'"
                )
                db.execute(
                    "INSERT OR IGNORE INTO origins SELECT i.* FROM incoming.origins i JOIN referenced_keys k USING(request_key) WHERE i.kind='verified_response'"
                )
            missing = [
                key
                for key in block.requests.request_key.dropna().unique()
                if db.execute(
                    "SELECT 1 FROM responses WHERE request_key=? AND kind='verified_response'",
                    (key,),
                ).fetchone()
                is None
            ]
            if missing:
                raise ValueError("Missing raw responses for published features.")
            db.execute(
                "INSERT INTO feature_artifacts VALUES (?,?,?,?)",
                (
                    artifact_id,
                    json.dumps(manifest, ensure_ascii=False),
                    buffers["features.parquet"],
                    buffers["requests.parquet"],
                ),
            )
    _record_saved_block(
        path, json.loads(existing[0]) if existing else manifest, reused=bool(existing), phase=phase
    )
    return path


def _record_saved_block(folder, manifest, *, reused, phase="feature_creation"):
    artifact_id = manifest["artifact_id"]
    identity = manifest["identity"]
    # The embedded manifest is portable; DATA retains only a compact discovery index.
    write_json(
        paths.manifests_dir(phase) / "features" / f"{artifact_id}.json",
        {
            "artifact_id": artifact_id,
            "dataset_id": identity["dataset_id"],
            "data_identity": identity["data_identity"],
            "representation": identity["representation"],
            "provider": identity["provider"],
            "model": identity["model"],
            "version": identity["version"],
            "question_bank_id": identity.get("question_bank_id"),
            "question_variant": identity.get("question_variant"),
            "rows": manifest["row_count"],
            "numeric_columns": len(manifest["feature_columns"]),
            "artifact_path": str(folder),
            "sha256": manifest["sha256"],
            "status": "complete",
        },
    )
    append_event(
        paths.logs_dir(phase) / "builds" / f"{artifact_id}.log",
        "artifact_reused" if reused else "artifact_saved",
        artifact_path=str(folder),
        rows=manifest["row_count"],
    )


def load_block(folder: Path, artifact_id: str | None = None) -> FeatureBlock:
    """Read a selected immutable artifact; never silently choose among several versions."""
    folder = Path(folder)
    if folder.is_dir():  # Read-only compatibility for existing exported Parquet artifacts.
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        for name in ("features.parquet", "requests.parquet"):
            if file_sha256(folder / name) != manifest["sha256"][name]:
                raise ValueError("Feature artifact checksum mismatch.")
        values = pd.read_parquet(folder / "features.parquet")
        requests = pd.read_parquet(folder / "requests.parquet")
    else:
        with sqlite3.connect(folder.resolve().as_uri() + "?mode=ro", uri=True) as db:
            if (
                artifact_id is None
                and db.execute("SELECT COUNT(*) FROM feature_artifacts").fetchone()[0] != 1
            ):
                raise ValueError(
                    "Select an explicit artifact_id; file has zero or multiple matching artifacts."
                )
            rows = db.execute(
                "SELECT manifest_json, features_parquet, requests_parquet FROM feature_artifacts"
                + (" WHERE artifact_id=?" if artifact_id else ""),
                (artifact_id,) if artifact_id else (),
            ).fetchall()
        if len(rows) != 1:
            raise ValueError(
                "Select an explicit artifact_id; file has zero or multiple matching artifacts."
            )
        encoded, feature_bytes, request_bytes = rows[0]
        manifest = json.loads(encoded)
        for name, content in (
            ("features.parquet", feature_bytes),
            ("requests.parquet", request_bytes),
        ):
            if hashlib.sha256(content).hexdigest() != manifest["sha256"][name]:
                raise ValueError("Feature artifact checksum mismatch.")
        values = pd.read_parquet(io.BytesIO(feature_bytes))
        requests = pd.read_parquet(io.BytesIO(request_bytes))
    _validate_block(values, requests, manifest)
    return FeatureBlock(values, requests, manifest)


def feature_inventory(path: Path) -> pd.DataFrame:
    """Inspect all model/version/representation manifests without loading feature blobs.

    A nonexistent path is an empty inventory, not a reason to create an empty database.
    """
    path = Path(path)
    columns = [
        "artifact_id",
        "settings_sha256",
        "dataset_id",
        "provider",
        "model",
        "version",
        "question_bank_id",
        "question_variant",
        "representation",
        "row_count",
        "numeric_columns",
    ]
    if not path.is_file():
        return pd.DataFrame(columns=columns)
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        exists = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='feature_artifacts'"
        ).fetchone()
        rows = (
            db.execute(
                "SELECT manifest_json FROM feature_artifacts ORDER BY artifact_id"
            ).fetchall()
            if exists
            else []
        )
    records = []
    for (encoded,) in rows:
        manifest = json.loads(encoded)
        identity = manifest["identity"]
        if digest(identity) != manifest["artifact_id"]:
            raise ValueError("Invalid feature manifest identity")
        records.append(
            {
                **{key: identity.get(key) for key in columns},
                "artifact_id": manifest["artifact_id"],
                "row_count": manifest["row_count"],
                "numeric_columns": len(manifest["feature_columns"]),
            }
        )
    return pd.DataFrame(records, columns=columns)


def _validate_block(values, requests, manifest):
    if (
        manifest["status"] != "complete"
        or manifest["namespace"] != "verified_response"
        or digest(manifest["identity"]) != manifest["artifact_id"]
    ):
        raise ValueError("Invalid feature manifest.")
    if (
        len(values) != manifest["row_count"]
        or digest(values.row_id.tolist()) != manifest["row_ids_sha256"]
        or not values.row_id.is_unique
    ):
        raise ValueError("Feature row identity mismatch.")
    if list(values.columns.drop("row_id")) != manifest["feature_columns"]:
        raise ValueError("Feature schema mismatch.")
    if (
        list(requests.columns) != ["row_id", "group", "request_key", "status"]
        or set(requests.row_id) != set(values.row_id)
        or requests.duplicated(["row_id", "group"]).any()
    ):
        raise ValueError("Invalid row/request correspondence.")
    mode = manifest["identity"]["representation"]
    expected_groups = (
        {f"text_{i:03d}" for i in range(len(manifest["text_columns"]))}
        if mode == "per_column"
        else {"all_text"}
        if mode == "joint"
        else set()
    )
    if not expected_groups or set(requests.group) != expected_groups:
        raise ValueError("Missing or unexpected text-group trace.")
    indexed = values.set_index("row_id")
    for group, refs in requests.groupby("group", sort=False):
        if set(refs.row_id) != set(values.row_id):
            raise ValueError("Incomplete text-group trace.")
        cells = indexed.loc[
            refs.row_id, [c for c in manifest["feature_columns"] if f"__{group}__" in c]
        ]
        if cells.shape[1] == 0:
            raise ValueError("Text-group trace has no corresponding feature columns.")
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


def assemble_features(dataset, blocks: list[FeatureBlock], *, view: str = "main") -> pd.DataFrame:
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
        selected = feature_view(block, dataset.metadata.task_type, view)
        result = result.join(selected.set_index("row_id"), validate="one_to_one")
    result.attrs["feature_artifact_ids"] = [block.manifest["artifact_id"] for block in blocks]
    return result


def feature_view(block: FeatureBlock, task: str, view: str = "main") -> pd.DataFrame:
    """Select the agreed model inputs without discarding the retained raw probabilities."""
    if view not in {"main", "probabilities", "probabilities_plus_confidence", "retained"}:
        raise ValueError("Unknown feature view")
    if task not in {"binary_classification", "multiclass_classification", "regression"}:
        raise ValueError("Unknown task")
    columns = block.manifest["feature_columns"]
    if view == "retained":
        chosen = columns
    elif view == "main" and task == "regression":
        chosen = [c for c in columns if c.endswith("__expected_direction")]
    else:
        chosen = [c for c in columns if c.rsplit("__", 1)[-1].startswith("p_")]
    if not chosen:
        raise ValueError("Requested feature view is absent")
    result = block.values.loc[:, ["row_id", *chosen]].copy()
    if view == "probabilities_plus_confidence":
        # Provider confidence fields have different definitions. Compute comparable
        # summaries from probabilities; raw provider values remain in retained view.
        groups = {}
        for column in chosen:
            groups.setdefault(column.rsplit("__", 1)[0], []).append(column)
        for prefix, names in groups.items():
            probs = result[names].to_numpy(dtype=float)
            if task == "binary_classification":
                probs = np.column_stack([1 - probs[:, 0], probs[:, 0]])
            safe = np.clip(probs, np.finfo(float).tiny, 1)
            result[prefix + "__entropy_normalized"] = -(probs * np.log(safe)).sum(1) / np.log(
                probs.shape[1]
            )
            result[prefix + "__max_probability"] = probs.max(1)
            if task == "regression":
                levels = np.arange(-4, 5, dtype=float)
                mean = probs @ levels
                result[prefix + "__direction_variance"] = probs @ (levels**2) - mean**2
    return result


def feature_plan(audit, cfg: dict) -> pd.DataFrame:
    """Inspect the full feature-build scope without enumerating millions of requests."""
    records = []
    selected = {entry["id"] for entry in selected_entries(load_catalog(cfg["catalog"]), cfg)}
    for row in audit.summary.itertuples():
        if row.dataset_id not in selected:
            continue
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
                    "main_numeric_columns": groups
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
                "table": "<dataset_id>.sqlite3 / responses",
                "key": "request_key + namespace",
                "content": "Exact state/question/settings, raw response, UTC timestamp",
                "scope": "Shared across all experiments",
            },
            {
                "table": "<dataset_id>.sqlite3 / origins",
                "key": "request_key + origin hash",
                "content": "Dataset, csv:N, text columns, code/config/source hashes",
                "scope": "Many origins may reuse one response",
            },
            {
                "table": "feature_artifacts / embedded features.parquet",
                "key": "csv:N within artifact_id",
                "content": "One row per source row; numeric Jev columns only",
                "scope": "One dataset and text representation",
            },
            {
                "table": "feature_artifacts / embedded requests.parquet",
                "key": "csv:N + text group",
                "content": "Request key or missing_text status for each row/group",
                "scope": "Trace feature cells to raw responses",
            },
            {
                "table": "feature_artifacts / manifest_json",
                "key": "artifact_id",
                "content": "Embedded immutable model/template/data/code manifest and compressed table checksums",
                "scope": "Written last; complete artifact marker",
            },
            {
                "table": "Feature-build log / DATA manifest index",
                "key": "artifact_id",
                "content": "UTC progress, completion/failure, row counts, hashes and artifact location; no raw text",
                "scope": "output_JEVPFN/2. feature_creation; compact DATA records on VSC",
            },
            {
                "table": "Experiment feature matrix",
                "key": "csv:N",
                "content": "Original non-text columns joined with selected cached blocks",
                "scope": "Targets separate; no API work during experiment",
            },
        ]
    )


def feature_summary(plan, totals, review_totals=None) -> str:
    return "\n".join(
        [
            "1. Feature-build scope",
            f"{len(plan)} dataset/representation tables planned; no features generated.",
            "2. API body preview",
            "Documentation-verified Noul/Choice/Score mapping; guarded transport disabled, no paid call.",
            "3. Reusable tables",
            "Raw responses -> validated probability blocks + row/request map -> experiment-specific feature joins; targets stay separate.",
            "4. Workload and budget",
            "PRIMARY ESTIMATE: unique requests after exact reuse, documented API-body JSON, still approximate."
            if review_totals is not None
            else "Refined estimate unavailable/stale; refresh with python -m src.jev.review.",
            review_totals.to_string(index=False) if review_totals is not None else "",
            "REFERENCE ONLY: logical nonempty slots BEFORE exact reuse; intended JSON, not the expected paid-call total:",
            totals.to_string(index=False),
            "5. Execution sequence",
            "Design and local previews -> small approved local Jev test -> full local feature creation and validation -> experiment 0 on VSC -> experiment 1.",
            "Context overflow, live API behaviour and experimental protocol still require review.",
        ]
    )
