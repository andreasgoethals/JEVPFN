"""Descriptive audit only. Target statistics are never inputs to src.jev."""

from __future__ import annotations

import json
import os
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loaders import LoadedDataset, load_catalog, load_dataset
from src.data.metadata import TaskMetadata
from src.data.official_types import is_date
from src.utils import paths
from src.utils.serialization import canonical_json, digest, scalar, text_value, write_json

AUDIT_VERSION = "audit-v2-skip-empty"


@dataclass
class AuditCollection:
    summary: pd.DataFrame
    columns: pd.DataFrame
    text: pd.DataFrame
    numeric: pd.DataFrame
    categorical: pd.DataFrame
    details: dict
    folder: Path

    def metadata(self, dataset_id: str) -> TaskMetadata:
        record = dict(self.details[dataset_id]["task_metadata"])
        for key in ("classes", "text_columns", "non_text_columns"):
            record[key] = tuple(record[key])
        return TaskMetadata(**record)


def _stats(values: pd.Series) -> dict:
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {
            k: None
            for k in ("count", "mean", "std", "min", "p05", "p25", "median", "p75", "p95", "max")
        }
    return {
        "count": int(values.size),
        "mean": scalar(values.mean()),
        "std": scalar(values.std()),
        "min": scalar(values.min()),
        **{
            name: scalar(values.quantile(q))
            for name, q in (
                ("p05", 0.05),
                ("p25", 0.25),
                ("median", 0.5),
                ("p75", 0.75),
                ("p95", 0.95),
            )
        },
        "max": scalar(values.max()),
    }


def _length_stats(values: pd.Series, prefix: str) -> dict:
    return {
        f"{prefix}_mean": scalar(values.mean()),
        f"{prefix}_median": scalar(values.median()),
        f"{prefix}_p95": scalar(values.quantile(0.95)),
        f"{prefix}_max": scalar(values.max()),
    }


def _examples(series: pd.Series, count: int, limit: int) -> list[dict]:
    nonnull = series.dropna().astype(str)
    candidates = pd.DataFrame({"value": nonnull, "characters": nonnull.str.len()})
    candidates = candidates[candidates["characters"] > 0].drop_duplicates("value")
    candidates = candidates.sort_values("characters", kind="stable")
    if candidates.empty:
        return []
    positions = np.unique(np.linspace(0.1, 0.9, count) * (len(candidates) - 1)).astype(int)
    return [
        {
            "row_id": f"csv:{candidates.index[i]}",
            "characters": int(candidates.iloc[i]["characters"]),
            "text_preview": candidates.iloc[i]["value"][:limit],
            "truncated_for_display": len(candidates.iloc[i]["value"]) > limit,
        }
        for i in positions
    ]


def _value_lengths(series: pd.Series) -> np.ndarray:
    return np.fromiter(
        (len(canonical_json(scalar(v))) for v in series), dtype=np.int64, count=len(series)
    )


def request_lengths(dataset: LoadedDataset) -> dict[str, np.ndarray]:
    """JSON character counts of label-free feature values, with no raw values saved here."""
    m = dataset.metadata
    lengths = {}
    for i, col in enumerate(m.text_columns):
        lengths[f"text_{i}"] = len(canonical_json(col)) + 1 + _value_lengths(dataset.frame[col])
        lengths[f"present_{i}"] = dataset.frame[col].map(text_value).notna().to_numpy()
    return lengths


def audit_dataset(dataset: LoadedDataset, cfg: dict) -> tuple[dict, dict]:
    frame, m = dataset.frame, dataset.metadata
    ratio = cfg["token_estimation"]["characters_per_token"]
    n = len(frame)
    columns, texts, numeric, categorical, examples = [], [], [], [], {}
    combined_chars = pd.Series(np.zeros(n, dtype=np.int64), index=frame.index)
    cell_tokens_total = 0
    for col in frame:
        if col == m.target:
            continue
        series = frame[col]
        observed = series.notna().sum()
        if col in m.text_columns:
            kind = "text"
        elif pd.api.types.is_bool_dtype(series):
            kind = "categorical"
        elif pd.api.types.is_numeric_dtype(series):
            kind = "numerical"
        elif is_date(series):
            kind = "other"
        elif pd.api.types.is_string_dtype(series.dtype) or isinstance(
            series.dtype, pd.CategoricalDtype
        ):
            kind = "categorical"
        else:
            kind = "other"
        base = {
            "dataset_id": m.dataset_id,
            "feature": col,
            "dtype": str(series.dtype),
            "kind": kind,
            "missing_pct": float(100 * series.isna().mean()),
            "n_unique": int(series.nunique()),
            "unique_ratio": float(series.nunique() / observed) if observed else 0.0,
        }
        columns.append(base)
        if kind == "text":
            values = series.fillna("").astype(str)
            chars = values.str.len()
            words = values.str.split().str.len()
            tokens = np.ceil(chars / ratio).astype(np.int64)
            combined_chars += chars
            cell_tokens_total += int(tokens.sum())
            text_stats = {
                **base,
                **_length_stats(chars, "characters"),
                **_length_stats(words, "words"),
                **_length_stats(tokens, "tokens"),
                "empty_or_whitespace_pct": float(100 * values.str.strip().eq("").mean()),
            }
            texts.append(text_stats)
            examples[col] = _examples(
                series, cfg["audit"]["examples_per_column"], cfg["audit"]["example_character_limit"]
            )
        elif kind == "numerical":
            numeric.append({**base, **_stats(series)})
        elif kind == "categorical":
            counts = series.dropna().value_counts(sort=False)
            common = sorted(
                ((scalar(k), int(v)) for k, v in counts.items()),
                key=lambda kv: (-kv[1], canonical_json(kv[0])),
            )[: cfg["audit"]["top_categories"]]
            categorical.append(
                {**base, "top_values": [{"value": k, "count": v} for k, v in common]}
            )
    combined_tokens = np.ceil(combined_chars / ratio).astype(np.int64)
    summary = {
        "dataset_id": m.dataset_id,
        "dataset": m.dataset_name,
        "rows": n,
        "columns_including_target": len(frame.columns),
        "features": len(frame.columns) - 1,
        "task_type": m.task_type,
        "target": m.target,
        "target_dtype": str(frame[m.target].dtype),
        "target_missing_pct": float(100 * frame[m.target].isna().mean()),
        "n_classes": len(m.classes) if m.classes else None,
        "class_labels": list(m.classes),
        "text_columns": len(m.text_columns),
        "non_text_columns": len(m.non_text_columns),
        **_length_stats(combined_tokens, "combined_tokens"),
        "total_text_tokens": int(combined_tokens.sum()),
        "total_cell_text_tokens": cell_tokens_total,
        "total_text_characters": int(combined_chars.sum()),
        "issues": dataset.issues,
    }
    y = frame[m.target]
    if m.task_type == "regression":
        target = {"kind": "regression", "statistics": _stats(y)}
    else:
        counts = y.value_counts()
        target = {
            "kind": m.task_type,
            "distribution": [
                {
                    "class": v,
                    "count": int(counts.get(v, 0)),
                    "fraction_of_all_rows": float(counts.get(v, 0) / n),
                }
                for v in m.classes
            ],
        }
    details = {
        "task_metadata": asdict(m),
        "target_audit_only": target,
        "text_examples": examples,
        "source": dataset.source,
        "issues": dataset.issues,
    }
    return {
        "summary": summary,
        "columns": columns,
        "text": texts,
        "numeric": numeric,
        "categorical": categorical,
    }, details


def _save_lengths(path: Path, lengths: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **lengths)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def audit_lock(folder: Path, *, timeout: float = 600):
    """One builder/length-reader per signature; OS releases the lock after a crash.

    Atomic replacement alone is insufficient on Windows: an open npz reader prevents
    another process replacing that file. Both template notebooks share this audit.
    """
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".audit.lock").open("a+b") as handle:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"0")
            handle.flush()
        if os.name == "nt":
            import msvcrt

            def acquire():
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

            def release():
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            def acquire():
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            def release():
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

        started = time.monotonic()
        while True:
            try:
                acquire()
                break
            except OSError as exc:
                if time.monotonic() - started >= timeout:
                    raise TimeoutError(f"Timed out waiting for shared audit at {folder}") from exc
                time.sleep(0.1)
        try:
            yield
        finally:
            release()


def audit_collection(cfg: dict, *, refresh: bool = False) -> AuditCollection:
    """Both notebooks can run independently; identical complete audits are reused.

    Separate atomic files and a final manifest allow resuming at dataset boundaries.
    The signature includes all audit source code, data hashes, runtime and relevant config.
    """
    from src.utils.provenance import code_identity, environment

    catalog = load_catalog(cfg["catalog"])
    identity = {
        "catalog": catalog,
        "audit_config": cfg["audit"],
        "tokens": cfg["token_estimation"],
        "version": AUDIT_VERSION,
        "code": code_identity(),
        "environment": environment(),
    }
    folder = paths.results_dir("data_audit", digest(identity)[:16], phase="exploration")
    folder.mkdir(parents=True, exist_ok=True)
    with audit_lock(folder):
        return _collect_audit(cfg, catalog, identity, folder, refresh=refresh)


def _collect_audit(cfg, catalog, identity, folder, *, refresh):
    tables = {name: [] for name in ("summary", "columns", "text", "numeric", "categorical")}
    details = {}
    for entry in catalog["datasets"]:
        marker = folder / f"{entry['id']}.json"
        length_path = folder / f"{entry['id']}_lengths.npz"
        if marker.exists() and length_path.exists() and not refresh:
            cached = json.loads(marker.read_text(encoding="utf-8"))
            table, detail = cached["tables"], cached["details"]
        else:
            dataset = load_dataset(entry, allow_download=cfg["allow_download"])
            table, detail = audit_dataset(dataset, cfg)
            _save_lengths(length_path, request_lengths(dataset))
            write_json(marker, {"tables": table, "details": detail})
        details[entry["id"]] = detail
        for name in tables:
            tables[name].extend([table[name]] if name == "summary" else table[name])
    frames = {k: pd.DataFrame(v) for k, v in tables.items()}
    for name, frame in frames.items():
        # No raw examples in CSV; nested audit details have their own ignored JSON files.
        csv = frame.copy()
        for col in csv:
            if csv[col].map(lambda v: isinstance(v, (list, dict))).any():
                csv[col] = csv[col].map(
                    lambda v: canonical_json(v) if isinstance(v, (list, dict)) else v
                )
        _atomic_csv(csv, folder / f"{name}.csv")
    write_json(folder / "manifest.json", identity)
    write_json(
        paths.manifest_path("latest_data_audit", phase="exploration"),
        {"folder": str(folder), "identity": digest(identity)},
    )
    return AuditCollection(**frames, details=details, folder=folder)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def audit_summary(audit: AuditCollection) -> str:
    s = audit.summary
    lines = [
        "1. Scope and provenance",
        f"Loaded {len(s)}/20 core text datasets; {s.rows.sum():,} rows.",
        "2. Dataset overview",
        s[["dataset", "task_type", "rows", "text_columns", "non_text_columns"]].to_string(
            index=False
        ),
        "3. Column, text and target audit",
        f"Audited {len(audit.columns)} features, including {len(audit.text)} text columns.",
        "Distinct-value ratios exclude missing values. Length statistics include missing text as empty.",
        "Target distributions and regression summaries are audit-only and never enter requests.",
        "4. Text burden",
        f"Combined text: approximately {s.total_text_tokens.sum():,} tokens at the configured character ratio.",
        s[
            ["dataset", "combined_tokens_median", "combined_tokens_p95", "total_text_tokens"]
        ].to_string(index=False),
        "5. Saved outputs",
        str(audit.folder),
        "No Jev calls, model training or evaluation were performed.",
    ]
    return "\n".join(lines)
