"""Read-only study audit: completeness, compute, spending and data checks (no outcomes).

Feeds the general notebooks (`general/2_feature_creation`, `general/4_study_audit`,
`general/5_synthesis`). Every reader returns an explicit empty table or a waiting status when
its source is absent locally; nothing is estimated, simulated or written.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.endpoints import HEADLINES
from src.utils import paths
from src.utils.config import load_yaml

#: The owner-authorised Jev allowance (AGENTS.md, 5 November 2026): EUR 5 under a cumulative
#: USD 5.50 ceiling. No top-up is approved.
JEV_CEILING_USD = 5.50
WAITING = "waiting for export"

#: Benchmark phase -> study label used in audit tables (folder renumbering of 23-11-2026).
PHASE_STUDY = {
    "experiment_1": "B1", "experiment_2": "B2a", "experiment_3": "B2b", "rq2": "B3",
    "data_efficiency": "B4", "experiment_4": "B5", "representations": "C1", "rq3": "C2",
}
#: Feature set -> arm family (one column group per family in the completeness matrix).
FEATURE_FAMILY = {
    "non_text": "no text",
    "non_text_plus_joint": "decision", "non_text_plus_per_column": "decision",
    "non_text_plus_both": "decision",
    "non_text_plus_frozen_e5": "frozen", "non_text_plus_both_plus_frozen_e5": "decision+frozen",
    "non_text_plus_tar": "TAR", "non_text_plus_supervised_probe": "classifier",
    "non_text_plus_per_column_without_one_text": "column refit",
    "original_features": "native",
    "non_text_plus_hidden": "hidden", "non_text_plus_per_column_plus_hidden": "hidden+decision",
    "non_text_plus_joint_plus_hidden": "hidden+decision",
    "non_text_plus_adapted_joint": "adapted",
}
ARM_KEYS = ["phase", "dataset_id", "model_id", "feature_set", "provider", "view", "encoder",
            "pca_scope", "pca_components", "probe_view", "dropped_text_column", "training_size",
            "draw", "hidden_view", "hidden_position", "hidden_layer", "adaptation_cell"]
STATUSES = ("complete", "missing", "roster_gap", "excluded", "waiting")


def _roster():
    from src.data.loaders import load_catalog
    from src.data.selection import selected_entries

    main = load_yaml("experiment_1/main")
    datasets = [e["id"] for e in selected_entries(load_catalog(), main)]
    return datasets, list(main["candidate_extractors"])


# ----------------------------------------------------------------- feature-store completeness

def _compact_manifests(folder=None) -> pd.DataFrame:
    folder = Path(folder or paths.manifests_dir("feature_creation") / "features")
    rows = []
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        body = json.loads(path.read_text(encoding="utf-8"))
        if body.get("status") == "complete":
            rows.append({"dataset_id": body["dataset_id"], "provider": body["provider"],
                         "representation": body["representation"], "rows": body.get("rows")})
    return pd.DataFrame(rows, columns=["dataset_id", "provider", "representation", "rows"])


def _local_stores(datasets) -> pd.DataFrame:
    from src.evaluation.portable_features import notebook_inventory

    frames = []
    for dataset_id in datasets:
        if paths.dataset_cache_path(dataset_id).is_file():
            frames.append(notebook_inventory(dataset_id)[["dataset_id", "provider", "representation",
                                                          "row_count"]].rename(columns={"row_count": "rows"}))
    return (pd.concat(frames, ignore_index=True) if frames
            else pd.DataFrame(columns=["dataset_id", "provider", "representation", "rows"]))


def feature_store_status(local: pd.DataFrame | None = None, manifests: pd.DataFrame | None = None,
                         *, datasets=None, providers=None, partial=()) -> pd.DataFrame:
    """One row per dataset x decision model x request view (per column, joint).

    ``observed``: a complete feature artifact is in the local store or a compact
    feature-creation manifest. ``partial``: ``(provider, dataset)`` pairs with recorded
    requests but no complete artifact (Jev under its fixed per-dataset allocation).
    ``unknown``: no local evidence (not a failure).
    """
    partial = set(partial)
    if datasets is None or providers is None:
        default_datasets, default_providers = _roster()
        datasets = datasets or default_datasets
        providers = providers or default_providers
    local = _local_stores(datasets) if local is None else local
    manifests = _compact_manifests() if manifests is None else manifests
    sources = {}
    for frame, label in ((manifests, "compact manifest"), (local, "local store")):
        for row in frame.itertuples():
            sources.setdefault((row.dataset_id, row.provider, row.representation), set()).add(label)
    rows = []
    for dataset_id in datasets:
        for provider in providers:
            for view in ("per_column", "joint"):
                found = sorted(sources.get((dataset_id, provider, view), ()))
                status = ("observed" if found else "partial" if (provider, dataset_id) in partial
                          else "unknown")
                rows.append({"dataset_id": dataset_id, "provider": provider, "view": view,
                             "status": status,
                             "source": " + ".join(found) or None})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- extraction compute

def inference_timings(folder=None) -> pd.DataFrame:
    """Measured inference wall time per model and dataset from the timing manifests.

    Decision models: ``decision_inference`` (one request per observation, cache reuse is
    excluded); frozen encoders: ``embedding_batch`` items (distinct texts). One allocated GPU
    per job, so wall seconds are GPU seconds. Loading, staging and publication are excluded.
    """
    folder = Path(folder or paths.manifests_dir("feature_creation") / "timings")
    rows = []
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        body = json.loads(path.read_text(encoding="utf-8"))
        context = body.get("context") or {}
        name = body.get("name")
        model = name if name != "embeddings" else _encoder_id(context.get("model"))
        for stage in body.get("stages", []):
            if stage["stage"] not in ("decision_inference", "embedding_batch") or not stage.get("dataset_id"):
                continue
            rows.append({"model_id": model, "kind": "decision" if name != "embeddings" else "encoder",
                         "dataset_id": stage["dataset_id"],
                         "requests": stage["items"] if name == "embeddings" else stage["observations"],
                         "gpu_seconds": stage.get("wall_seconds"), "job_id": context.get("job_id"),
                         "source": path.name})
    return pd.DataFrame(rows, columns=["model_id", "kind", "dataset_id", "requests", "gpu_seconds",
                                       "job_id", "source"])


def _encoder_id(model):
    names = {"intfloat/e5-small-v2": "e5_small_v2", "intfloat/e5-large-v2": "e5_large_v2",
             "Qwen/Qwen3-Embedding-0.6B": "qwen3_embedding_0_6b"}
    return names.get(model, model)


def read_compute_ledger(path=None):
    """The Slurm compute ledger (`src.cluster.compute_ledger`), or ``None`` when absent."""
    path = Path(path or paths.outputs_dir() / "compute_ledger.json")
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def extraction_workload(timings: pd.DataFrame, *, token_audit=None, ledger=None,
                        models=None, encoders=("e5_small_v2", "e5_large_v2", "qwen3_embedding_0_6b"),
                        distinct_texts=None) -> pd.DataFrame:
    """Per decision model and encoder: distinct requests and GPU-hours per 1,000 requests.

    ``inference`` rates come from the timing manifests (inference only); ``ledger`` rates
    from successful Slurm decision-feature jobs (all job time) over audited distinct requests.
    """
    from src.evaluation.model_properties import compute_per_thousand

    models = list(models if models is not None else _roster()[1])
    audit = {r["provider"]: r["unique_inputs"] for r in (token_audit or {}).get("self_hosted", [])}
    ledger_rates = (compute_per_thousand(ledger, token_audit, models).set_index("provider")
                    if ledger is not None else None)
    rows = []
    for model in [*models, *encoders]:
        part = timings[timings.model_id.eq(model) & timings.gpu_seconds.notna()]
        timed, seconds = float(part.requests.sum()), float(part.gpu_seconds.sum())
        kind = "encoder" if model in encoders else "hosted" if model == "jev" else "decision"
        distinct = distinct_texts if kind == "encoder" else audit.get(model)
        row = {"model_id": model, "kind": kind, "distinct_requests": distinct,
               "timed_requests": timed or None, "timed_datasets": part.dataset_id.nunique(),
               "inference_gpu_hours": seconds / 3600 if timed else None,
               "inference_gpu_hours_per_1000": seconds / timed * 1000 / 3600 if timed else None,
               "ledger_gpu_hours_per_1000": None}
        if ledger_rates is not None and model in ledger_rates.index:
            row["ledger_gpu_hours_per_1000"] = ledger_rates.loc[model, "gpu_hours_per_1000_requests"]
        rows.append(row)
    frame = pd.DataFrame(rows)
    for column in ("distinct_requests", "timed_requests", "inference_gpu_hours",
                   "inference_gpu_hours_per_1000", "ledger_gpu_hours_per_1000"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def dataset_rates(timings: pd.DataFrame) -> pd.DataFrame:
    """GPU-hours per 1,000 requests per model and dataset (the spread behind each pooled rate)."""
    part = timings[timings.gpu_seconds.notna() & timings.requests.gt(0)]
    out = part.groupby(["model_id", "dataset_id"], as_index=False)[["requests", "gpu_seconds"]].sum()
    out["gpu_hours_per_1000"] = out.gpu_seconds / out.requests * 1000 / 3600
    return out


# ----------------------------------------------------------------------- Jev spending

def jev_spend(cache_folder=None) -> pd.DataFrame:
    """Recorded Jev charges per dataset from the stored job ledgers (read-only).

    Reads the live ``extraction_jobs`` table and imported snapshots; one record per request
    key (the largest recorded charge), so repeated snapshots are never double counted.
    """
    rows = []
    for path in sorted(Path(cache_folder or paths.jev_cache_path()).glob("*.sqlite3")):
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            jobs, allocation = {}, None
            if "imported_extraction_records" in tables:
                for table, key, body in db.execute(
                        "SELECT table_name,record_key,record_json FROM imported_extraction_records"):
                    record = json.loads(body)
                    if table == "extraction_policy":
                        allocation = json.loads(record.get("settings_json", "{}")).get("budget", allocation)
                    elif table == "extraction_jobs":
                        _keep_job(jobs, key, record)
            if "extraction_jobs" in tables:
                columns = [r[1] for r in db.execute("PRAGMA table_info(extraction_jobs)")]
                for values in db.execute("SELECT * FROM extraction_jobs"):
                    record = dict(zip(columns, values))
                    _keep_job(jobs, record.get("request_key"), record)
        jev = [j for j in jobs.values() if _provider(j) == "jev"]
        if jev:
            rows.append({"dataset_id": path.stem, "jobs": len(jev),
                         "completed": sum(j.get("status") == "completed" for j in jev),
                         "unresolved": sum(j.get("status") != "completed" for j in jev),
                         "spent_usd": sum(j.get("spent_usd") or 0 for j in jev),
                         "reserved_usd": sum(j.get("reserved_usd") or 0 for j in jev),
                         "allocation_usd": allocation})
    return pd.DataFrame(rows, columns=["dataset_id", "jobs", "completed", "unresolved", "spent_usd",
                                       "reserved_usd", "allocation_usd"])


def _provider(job):
    try:
        return json.loads(job.get("request_json") or "{}").get("settings", {}).get("provider")
    except (TypeError, ValueError):
        return None


def _keep_job(jobs, key, record):
    if key is not None and (key not in jobs or (record.get("spent_usd") or 0) > (jobs[key].get("spent_usd") or 0)):
        jobs[key] = record


def jev_budget(spend: pd.DataFrame) -> dict:
    """Recorded charges and open reservations against the owner-approved ceiling."""
    eur = load_yaml("feature_creation")["live_extraction"]["owner_budget_eur"]
    spent, reserved = float(spend.spent_usd.sum()), float(spend.reserved_usd.sum())
    return {"recorded_charges_usd": spent, "open_reservations_usd": reserved,
            "committed_usd": spent + reserved, "ceiling_usd": JEV_CEILING_USD, "allowance_eur": eur,
            "remaining_usd": JEV_CEILING_USD - spent - reserved,
            "datasets_with_jobs": int(len(spend)), "unresolved_jobs": int(spend.unresolved.sum())}


# -------------------------------------------------------------------- compute ledger views

def ledger_tables(ledger: dict | None):
    """(jobs, by_study) from the compute ledger; empty frames when it is absent."""
    from src.cluster.compute_ledger import STAGE_STUDY, stage_of

    if not ledger:
        return pd.DataFrame(), pd.DataFrame()
    jobs = pd.DataFrame(ledger.get("jobs", []))
    if jobs.empty:
        return jobs, pd.DataFrame()
    # Re-derive the stage so jobs from before a STAGES addition are classified the same way.
    jobs["stage"] = jobs.job_name.map(stage_of)
    jobs["study"] = jobs.stage.map(STAGE_STUDY).fillna("Other")
    jobs["successful"] = jobs.state.eq("COMPLETED")
    by_study = jobs.groupby("study").apply(lambda g: pd.Series({
        "jobs": len(g), "successful_jobs": int(g.successful.sum()),
        "gpu_hours": g.gpu_hours.sum(), "successful_gpu_hours": g.gpu_hours[g.successful].sum(),
        "cpu_core_hours": g.cpu_core_hours.sum(),
        "successful_cpu_core_hours": g.cpu_core_hours[g.successful].sum()}), include_groups=False)
    return jobs, by_study.reset_index()


def failure_register(jobs: pd.DataFrame) -> pd.DataFrame:
    """Terminal attempts per stage and state (failed, cancelled and timed-out kept)."""
    if jobs.empty:
        return pd.DataFrame(columns=["stage", "state", "jobs", "gpu_hours", "cpu_core_hours"])
    return (jobs.groupby(["stage", "state"], as_index=False)
            .agg(jobs=("job_id", "size"), gpu_hours=("gpu_hours", "sum"),
                 cpu_core_hours=("cpu_core_hours", "sum")))


# ------------------------------------------------------------------------- exports

def export_status() -> pd.DataFrame:
    """Every result export the notebooks read: verified, waiting or superseded."""
    from src.evaluation import (
        benchmark_export,
        comparison_report,
        geometry_export,
        group_c,
        robustness_export,
    )

    readers = [
        ("A1", "A/exp01_outputs", lambda: comparison_report.read_report(comparison_report.report_config())),
        ("A2", "A/exp02_robustness", robustness_export.read_export),
        ("A3", "A/exp03_geometry", geometry_export.read_export),
        *[(s, n, (lambda s=s: benchmark_export.read(s))) for s, n in (
            ("B1", "B/exp04_decision_features"), ("B2", "B/exp05_frozen_embeddings"),
            ("B3", "B/exp08_when_text_helps"), ("B4", "B/exp09_data_efficiency"),
            ("B5", "B/exp07_native_text"))],
        ("C1", "C/exp10_representations", lambda: group_c.read("C1")),
        ("C2", "C/exp11_adaptation", lambda: group_c.read("C2")),
    ]
    rows, frames = [], {}
    for study, notebook, read in readers:
        try:
            tables, receipt = read()
        except (ValueError, KeyError, OSError) as error:
            tables, receipt = {}, {"status": "unreadable", "error": str(error)[:160]}
        frames[study] = tables
        path = receipt.get("path")
        rows.append({"study": study, "notebook": notebook, "status": receipt.get("status"),
                     "schema": receipt.get("schema"), "created_utc": receipt.get("created_utc"),
                     "export": Path(path).relative_to(paths.outputs_dir()).as_posix()
                     if path and Path(path).is_relative_to(paths.outputs_dir()) else path,
                     "note": receipt.get("error") or receipt.get("expected_schema")})
    status = pd.DataFrame(rows)
    status.attrs["frames"] = frames
    return status


# ------------------------------------------------------------- arm completeness and checks

def _arm_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for name in ARM_KEYS:
        if name not in frame:
            frame[name] = None
    frame = frame.astype({c: object for c in frame.columns if isinstance(frame[c].dtype, pd.CategoricalDtype)})
    for name in ARM_KEYS:
        frame[name] = frame[name].map(_key_value)
    # Exports write view "main" for arms without a decision view; the plan leaves it empty.
    frame["view"] = frame["view"].where(frame["view"].ne("main"), None)
    return frame


def _key_value(value):
    """One comparable string per recipe field (None for missing; 32.0 and 32 are equal)."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
        return None
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def arm_completeness(folds: pd.DataFrame, plan: pd.DataFrame, *, roster) -> pd.DataFrame:
    """Status per study, dataset, arm family and predictor from the export fold tables.

    complete: every planned arm has five defined headline folds; roster_gap: the missing arms
    are whole decision models; missing: anything else incomplete; excluded: planned zero arms
    (by design). Only phases present in ``plan`` are reported.
    """
    plan = _arm_frame(plan.rename(columns={"model": "model_id"}))
    plan["family"] = plan.feature_set.map(FEATURE_FAMILY)
    done = pd.DataFrame(columns=ARM_KEYS)
    if not folds.empty:
        head = folds[folds.metric.astype(str).eq(folds.task_type.astype(str).map(HEADLINES))]
        head = _arm_frame(head)
        counts = head.groupby(ARM_KEYS, dropna=False).agg(
            folds=("outer_fold", "nunique"), defined=("value", lambda v: int(np.isfinite(v.astype(float)).sum())))
        done = _arm_frame(counts[counts.folds.eq(5) & counts.defined.eq(5)].reset_index()[ARM_KEYS])
    done_keys = set(map(tuple, done[ARM_KEYS].itertuples(index=False)))
    plan["complete"] = [tuple(r) in done_keys for r in plan[ARM_KEYS].itertuples(index=False)]
    datasets = sorted(plan.dataset_id.unique())
    rows = []
    for (phase, family, model), group in plan.groupby(["phase", "family", "model_id"]):
        for dataset_id in datasets:
            arms = group[group.dataset_id.eq(dataset_id)]
            if arms.empty:
                status = "excluded"
            elif arms.complete.all():
                status = "complete"
            else:
                missing = arms[~arms.complete]
                whole = (missing.provider.notna().all() and all(
                    not arms[arms.provider.eq(p)].complete.any() for p in missing.provider.unique()))
                status = "roster_gap" if whole and arms.complete.any() and set(missing.provider) <= set(roster) else "missing"
            rows.append({"study": PHASE_STUDY.get(phase, phase), "phase": phase, "family": family,
                         "model_id": model, "dataset_id": dataset_id, "planned": len(arms),
                         "complete": int(arms.complete.sum()), "status": status})
    return pd.DataFrame(rows)


def fold_checks(folds: pd.DataFrame, study: str) -> dict:
    """Duplicate records, non-finite headline values and arms without five folds."""
    if folds.empty:
        return {"study": study, "rows": 0, "duplicate_records": 0, "non_finite_headline": 0,
                "arms_without_five_folds": 0}
    frame = _arm_frame(folds)
    keys = [*ARM_KEYS, "outer_fold", "metric"]
    head = frame[frame.metric.eq(frame.task_type.astype(str).map(HEADLINES))]
    values = pd.to_numeric(head.value, errors="coerce")
    per_arm = head.groupby(ARM_KEYS, dropna=False).outer_fold.nunique()
    return {"study": study, "rows": len(frame),
            "duplicate_records": int(frame.duplicated(keys).sum()),
            "non_finite_headline": int((~np.isfinite(values)).sum()),
            "arms_without_five_folds": int(per_arm.ne(5).sum())}


def fit_seconds(folds: pd.DataFrame) -> pd.DataFrame:
    """Mean outer-fold fit seconds and input columns per arm (from export fold tables)."""
    if folds.empty or not {"seconds.fit", "columns.representation"} & set(folds.metric.astype(str)):
        return pd.DataFrame(columns=["phase", "model_id", "family", "dataset_id", "columns", "fit_seconds"])
    frame = _arm_frame(folds)
    frame = frame[frame.metric.isin(["seconds.fit", "columns.representation"])]
    wide = frame.pivot_table(index=ARM_KEYS, columns="metric", values="value", aggfunc="mean",
                             dropna=False).reset_index().dropna(subset=["seconds.fit", "columns.representation"])
    wide["family"] = wide.feature_set.map(FEATURE_FAMILY)
    return wide.rename(columns={"seconds.fit": "fit_seconds", "columns.representation": "columns"})[
        ["phase", "model_id", "family", "dataset_id", "columns", "fit_seconds"]]
