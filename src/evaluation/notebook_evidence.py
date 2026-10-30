"""Read-only, dated evidence for preparation notebooks; never infers live VSC status.

Missing local receipts are unknown, not failed models. Diagnostic observations are
kept separate from production feature coverage and final Experiment 0 acceptance.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import paths
from src.utils.config import load_yaml
from src.utils.serialization import file_sha256


def study_grid(plan):
    """Count executable configurations and correlated fits, not independent samples."""
    frame = pd.DataFrame(plan)
    semantics = load_yaml("dataset_semantics")["datasets"]
    rows = []
    for phase, part in frame.groupby("phase", sort=True):
        binary = sum(semantics[d]["task_type"] == "binary_classification" for d in part.dataset_id)
        rows.append({"experiment": phase, "datasets": part.dataset_id.nunique(),
                     "predictors": part.model.nunique(), "configurations": len(part),
                     "outer_fits": 5 * len(part), "binary_inner_fits": 15 * binary})
    return pd.DataFrame(rows)


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _reports(folder):
    return [(p, _read(p)) for p in sorted(folder.glob("*.json"))]


def local_runtime_evidence(inventory=None):
    """Compact receipt inventory; counts do not certify current-code acceptance.

    The latest dated contract is shown per runtime, with its path/hash retained.
    No checkpoint files are hashed or loaded and no remote folders are inspected.
    """
    from src.cluster.token_audit import saved_exact_token_tables

    folder = paths.manifests_dir("experiment_0") / "before_extraction"
    token_scope, tokens = saved_exact_token_tables()
    token_providers = set(tokens.provider) if not tokens.empty else set()
    candidates = load_yaml("extractors")["candidates"]
    encoders = load_yaml("feature_creation/embedding_models")["models"]
    reports = _reports(folder / "gpu_contracts") + _reports(folder / "embeddings")
    smoke = folder / "jev_smoke.json"
    if smoke.is_file():
        reports.append((smoke, _read(smoke) | {"provider": "jev"}))
    rows = []
    for model_id, config in (list(candidates.items()) + list(encoders.items())):
        embedding = model_id in encoders
        selected = [(p, r) for p, r in reports
                    if r.get("encoder_id", r.get("provider")) == model_id]
        selected.sort(key=lambda x: (x[1].get("created_utc", ""), x[0].name))
        source, report = selected[-1] if selected else (None, {})
        status = report.get("status", "")
        runtime = ("failed" if status == "failed" else "observed"
                   if status.startswith(("gpu_contracts_passed", "embedding_contracts_passed", "jev_smoke_passed"))
                   else "unknown")
        settings = report.get("config" if embedding else "settings", {})
        reported_model = settings.get("model", report.get("model"))
        reported_pin = settings.get("revision" if embedding else "version")
        expected_pin = config.get("revision", config.get("version"))
        if reported_model and reported_model != config["model"]:
            runtime = "review"
        if reported_pin and reported_pin != expected_pin:
            runtime = "review"
        if runtime == "observed" and model_id != "jev" and (not reported_model or not reported_pin):
            runtime = "review"
        if model_id == "jev":
            marker, staged = None, "not_applicable"
        else:
            marker = paths.checkpoints_dir(
                "text_embeddings" if embedding else "decision_models",
                config["model"].split("/")[-1] if embedding else model_id,
                "_JEVPFN_EMBEDDING.json" if embedding else "_JEVPFN_CHECKPOINT.json",
            )
            staged = "observed" if marker.is_file() else "unknown"
            if marker.is_file() and _read(marker).get("revision") != expected_pin:
                staged = "review"
        if inventory is not None and model_id in set(inventory.provider):
            current = inventory[inventory.provider.eq(model_id)]
            counts = [current[current.representation.eq(m)].loc[lambda t: t.status.eq("available")].dataset.nunique()
                      for m in ("per_column", "joint")]
            coverage = f"{counts[0]}/{current.dataset.nunique()} · {counts[1]}/{current.dataset.nunique()}"
        else:
            coverage = "separate encoder store" if embedding else "not assessed here"
        acceptance = folder / ("accepted_encoders.json" if embedding else
                               "accepted_jev14_pilot.json" if model_id == "jev" else
                               f"accepted_{model_id}.json")
        accepted = _read(acceptance) if acceptance.is_file() else {}
        robust = (report.get("real_prompt_robustness") or {}).get("structurally_valid")
        hosted_probes = folder / "jev_dataset_robustness.json"
        if model_id == "jev" and hosted_probes.is_file():
            robust = _read(hosted_probes).get("status") == "jev_dataset_robustness_requires_review"
        semantic = (report.get("semantic_diagnostics") or {}).get("passed")
        gpu = report.get("gpu") if isinstance(report.get("gpu"), dict) else {}
        rows.append({"model_id": model_id, "model": config["model"], "pin": expected_pin,
                     "kind": "encoder" if embedding else "hosted" if model_id == "jev" else "open",
                     "known_answer": "observed" if semantic else "failed" if semantic is False
                     else "not_applicable" if embedding or model_id == "jev" else "unknown",
                     "robustness": "observed" if robust else "failed" if robust is False
                     else "not_applicable" if embedding else "unknown",
                     "token_audit": "observed" if accepted.get("token_audit_id")
                     else "not_applicable" if embedding or model_id == "jev" else "unknown",
                     "accepted": "observed" if accepted.get("accepted") is True else "unknown",
                     "accepted_utc": accepted.get("reviewed_utc"),
                     "gpu_name": gpu.get("name"),
                     "peak_gpu_gb": (report.get("peak_gpu_allocated_bytes") or np.nan) / 1e9,
                     "load_seconds": report.get("load_seconds", np.nan),
                     "contract_seconds": report.get("seconds", np.nan),
                     "live_requests": report.get("live_requests"),
                     "configured": "observed", "staged_receipt": staged,
                     "full_bank_tokens": "observed" if model_id in token_providers else
                     "not_applicable" if embedding or model_id == "jev" else "unknown",
                     "bounded_runtime": runtime,
                     "semantic_review": "review" if status.startswith("gpu_contracts_passed") else "unknown",
                     "complete_feature_datasets": coverage,
                     "observed_utc": report.get("created_utc", "not recorded" if source else "not found locally"),
                     "source": str(source) if source else None,
                     "source_sha256": file_sha256(source) if source else None,
                     "staged_manifest": str(marker) if marker else None})
    return pd.DataFrame(rows), token_scope


def prompt_evidence(report_paths=None):
    """Analyze explicit local receipts, retaining their bank IDs, dates and hashes.

    Defaults to the real Jev receipt and latest real GPU contract for each provider.
    Different reports are plotted separately; they are never pooled into a rate.
    """
    from src.evaluation.diagnostic_reports import analyze

    folder = paths.manifests_dir("experiment_0") / "before_extraction"
    if report_paths is None:
        jev = folder / "jev_dataset_robustness.json"
        report_paths = [jev] if jev.is_file() else []
        grouped = {}
        for p, r in _reports(folder / "gpu_contracts"):
            if r.get("status") == "gpu_contracts_passed_requires_review" and r.get("real_prompt_robustness"):
                grouped.setdefault(r["provider"], []).append((r.get("created_utc", ""), p))
        report_paths += [max(items)[1] for items in grouped.values()]
    tables, probabilities, provenance = [], [], []
    for filename in report_paths:
        source = Path(filename)
        report = _read(source)
        table = analyze([source])
        table["created_utc"] = report.get("created_utc")
        table["variant"] = table.probe.str.split("__").str[-1]
        table["dataset_id"] = table.probe.str.removeprefix("real_").str.split("__").str[0]
        tables.append(table)
        detail = report.get("real_prompt_robustness", report.get("prompt_robustness", {}))
        records = detail.get("probes", report.get("attempts", []))
        for record in records:
            features = record["features"]
            if "expected_direction" in features:
                vector = np.array([features[f"p_{i}"] for i in range(9)], dtype=float)
                mean = float(vector @ np.arange(-4, 5))
                probabilities.append({"provider": table.provider.iloc[0], "probe": record["probe"],
                                      "mean": mean, "variance": float(vector @ (np.arange(-4, 5) - mean) ** 2),
                                      "probabilities": vector.tolist(), "source_sha256": file_sha256(source)})
        provenance.append({"path": str(source), "sha256": file_sha256(source),
                           "created_utc": report.get("created_utc"), "status": report["status"],
                           "scope": "selected diagnostic inputs; not a population estimate or current-code acceptance",
                           "input_sets": sorted({r.get("provenance", {}).get("input_set_id")
                                                 for r in records if r.get("provenance", {}).get("input_set_id")})})
    return (pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(),
            pd.DataFrame(probabilities), provenance)


def measured_jev_usage(cache_folder=None):
    """Read cumulative paid-job accounting once per stored request, never mutate a cache.

    These are lifetime diagnostic records, not a sample-based extraction forecast.
    Input tokens can omit an earlier ambiguous/retried attempt; cumulative charges do not.
    """
    rows = []
    for path in sorted((cache_folder or paths.jev_cache_path()).glob("*.sqlite3")):
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "extraction_jobs" not in tables:
                continue
            jobs = []
            for request, status, attempts, reserved, spent, tokens in db.execute(
                "SELECT request_json,status,attempts,reserved_usd,spent_usd,input_tokens FROM extraction_jobs"
            ):
                if json.loads(request)["settings"].get("provider") != "jev":
                    continue
                jobs.append((status, attempts, reserved, spent, tokens))
            if jobs:
                rows.append({"dataset_id": path.stem, "stored_jobs": len(jobs),
                             "completed": sum(j[0] == "completed" for j in jobs),
                             "attempts": sum(j[1] for j in jobs),
                             "unresolved_jobs": sum(j[0] != "completed" for j in jobs),
                             "reserved_usd": sum(j[2] for j in jobs),
                             "recorded_charges_usd": sum(j[3] for j in jobs),
                             "reported_input_tokens": sum(j[4] or 0 for j in jobs),
                             "source": str(path), "scope": "cumulative stored Jev diagnostic jobs"})
    return pd.DataFrame(rows)


def pilot_decision_outputs(cache_folder=None):
    """Decode the lifetime response cache; never imply pilot/campaign membership.

    Uses the production decoder, so a response that fails validation is counted, not
    plotted. Returns one row per cached payload/settings identity. This can include
    historical questions, model settings and production responses. No target is read.
    """
    from src.jev.provider import response_features
    from src.jev.requests import IntendedRequest
    from src.utils.serialization import digest

    rows, invalid = [], 0
    for path in sorted((cache_folder or paths.jev_cache_path()).glob("*.sqlite3")):
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "responses" not in tables:
                continue
            for payload_json, settings_json, response_json in db.execute(
                "SELECT payload_json,settings_json,response_json FROM responses "
                "WHERE kind='verified_response'"
            ):
                try:
                    features = response_features(
                        IntendedRequest(payload_json, settings_json, "{}"), json.loads(response_json)
                    )
                except Exception:
                    invalid += 1
                    continue
                payload = json.loads(payload_json)
                task = payload["state"]["task"]
                probabilities = np.array(
                    [v for k, v in features.items() if k.startswith("p_") and k != "p_positive"]
                )
                settings = json.loads(settings_json)
                row = {
                    "dataset_id": path.stem,
                    "dataset": task["dataset"],
                    "task_type": task["task_type"],
                    "provider": settings.get("provider", "jev"),
                    "model": settings.get("model"),
                    "settings_sha256": digest(json.loads(settings_json)),
                    "input": digest(payload),
                    "text_fields": len(payload["state"]["text"]),
                }
                if task["task_type"] == "binary_classification":
                    row["value"] = features["p_positive"]
                elif task["task_type"] == "multiclass_classification":
                    safe = np.clip(probabilities, np.finfo(float).tiny, 1)
                    row["value"] = float(-(probabilities * np.log(safe)).sum() / np.log(len(safe)))
                    row["top_class"] = int(np.argmax(probabilities))
                    row["top_probability"] = float(probabilities.max())
                    row["class_probabilities"] = probabilities.tolist()
                else:
                    row["value"] = features["expected_direction"]
                    row["no_evidence_probability"] = features["p_4"]
                rows.append(row)
    table = pd.DataFrame(rows, columns=[
        "dataset_id", "dataset", "task_type", "provider", "model", "settings_sha256", "input",
        "text_fields", "value", "top_class", "top_probability", "class_probabilities",
        "no_evidence_probability",
    ])
    table.attrs["invalid_responses"] = invalid
    return table


def shared_pilot_outputs(pilot):
    """Inputs answered by every configured extractor at its currently pinned model.

    Historical model versions (for example Jev 1.13) are excluded, and each provider
    contributes one settings identity per input. Returns the long table restricted to
    those inputs; agreement and spread are then computed on identical payloads only.
    """
    candidates = load_yaml("extractors")["candidates"]
    current = pilot[pilot.apply(
        lambda r: r.provider in candidates and r.model == candidates[r.provider]["model"], axis=1)]
    current = current.drop_duplicates(["provider", "input"])
    counts = current.groupby("input").provider.nunique()
    shared = counts[counts.eq(len(candidates))].index
    result = current[current.input.isin(shared)].copy()
    result.attrs["providers"] = [p for p in candidates if p in set(result.provider)]
    return result


def pilot_agreement(shared):
    """Pairwise agreement on identical inputs, per task.

    Binary and regression: Spearman rank correlation of the main scalar feature.
    Multiclass: share of inputs on which both models put most probability on the same class.
    """
    providers = shared.attrs.get("providers") or sorted(shared.provider.unique())
    result = {}
    for task, part in shared.groupby("task_type"):
        column = "top_class" if task == "multiclass_classification" else "value"
        wide = part.pivot(index="input", columns="provider", values=column).reindex(columns=providers)
        if task == "multiclass_classification":
            matrix = pd.DataFrame(
                [[float((wide[a] == wide[b]).mean()) for b in providers] for a in providers],
                index=providers, columns=providers)
        else:
            matrix = wide.astype(float).corr(method="spearman")
        result[task] = (matrix, len(wide))
    return result


def saved_pca_diagnostics(folds):
    """Extract outer-training diagnostics; remove reuse across predictors/metric rows.

    No validation/test observation is fitted, projected or used to select a recipe.
    Retained records identify encoder, dataset, split, fold, scope and text group.
    """
    rows = []
    if folds.empty:
        return pd.DataFrame()
    for record_path in folds.record_path.drop_duplicates():
        # collect() serializes compact RecordRefs as database.sqlite3#record-id;
        # legacy results still point at individual JSON files. Read both without
        # creating a database or materializing its potentially large predictions.
        database, separator, key = str(record_path).rpartition("#")
        if separator and Path(database).suffix == ".sqlite3":
            from src.evaluation.store import RecordRef

            report = json.loads(RecordRef(Path(database), key).read_text())
        else:
            report = _read(Path(record_path))
        meta = report["metadata"]
        if meta["prediction_role"] != "outer_test":
            continue
        embedding = (meta.get("representation_recipe") or {}).get("embedding") or {}
        for partition in meta.get("pca_diagnostics", []):
            if partition["partition"] != "outer_train":
                continue
            for group, diagnostic in enumerate(partition["columns"]):
                rows.append({"dataset_id": meta["dataset_id"], "split_id": meta["split_id"],
                             "outer_fold": meta["outer_fold"], "encoder": embedding.get("encoder_id"),
                             "pca_scope": embedding.get("pca_scope"), "text_group": group,
                             **diagnostic})
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
