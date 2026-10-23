"""Experiment 0A: exact, resumable CPU token counting over every selected prepared input.

python -m src.cluster.token_audit --stage-tokenizers  # small public files, then count
python -m src.cluster.token_audit                     # offline; resume saved counts

Ten open-weight tokenizers are available. Jev's hosted counts remain explicitly
unverified. This never removes rows, shortens text, loads weights or performs inference.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack, closing, contextmanager
from datetime import UTC, datetime
from itertools import islice
from uuid import uuid4

import numpy as np

from src.data.loaders import load_catalog
from src.data.selection import selected_entries
from src.jev.inputs import _read_set, input_origins, iter_prepared_inputs
from src.jev.registry import (
    HOSTED_UNAVAILABLE,
    SELF_HOSTED,
    ExactCounter,
    stage_tokenizers,
)
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.locking import file_lock
from src.utils.serialization import canonical_json, digest, file_sha256, write_json

# Keep the fingerprinted tokenizer implementations stable while filtering the active roster.
HOSTED_UNAVAILABLE = {
    provider: reason
    for provider, reason in HOSTED_UNAVAILABLE.items()
    if provider in load_yaml("extractors")["candidates"]
}


@contextmanager
def _exclusive_audit(path):
    """Reject duplicate writers; the OS releases the lock even after a killed job."""
    with ExitStack() as stack:
        try:
            stack.enter_context(file_lock(path, timeout=1))
        except TimeoutError as exc:
            raise RuntimeError(
                f"A token audit is already running for {path}. Wait for that job; "
                "do not delete its lock or saved counts. Rerunning later resumes them."
            ) from exc
        yield


def selected_inputs():
    entries = selected_entries(load_catalog(), load_yaml("feature_creation"))
    result = []
    for entry in entries:
        path = paths.dataset_question_path(entry["id"])
        if not path.is_file():
            raise FileNotFoundError(
                f"Prepared inputs missing for {entry['id']}; run src.jev.preparation first"
            )
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            identity, recipe, summary, _ = _read_set(db, None)
        result.append(
            {
                "dataset_id": entry["id"],
                "path": str(path),
                "input_set_id": identity,
                "input_file_sha256": file_sha256(path),
                "source_rows": recipe["row_count"],
                "unique_inputs": summary["unique_inputs"],
            }
        )
    return result


def saved_exact_token_tables():
    """Notebook view of verified saved counts; no tokenizer dependency or recounting."""
    import pandas as pd

    path = paths.manifests_dir("experiment_0") / "before_extraction" / "token_audit.json"
    if not path.is_file():
        return {
            "status": "not_counted",
            "command": "python -m src.cluster.token_audit",
        }, pd.DataFrame()
    combined = json.loads(path.read_text(encoding="utf-8"))
    current = {
        i["dataset_id"]: (i["input_set_id"], i["input_file_sha256"]) for i in selected_inputs()
    }
    rows = []
    for report in combined["self_hosted"]:
        observed = {
            i["dataset_id"]: (i["input_set_id"], i["input_file_sha256"]) for i in report["datasets"]
        }
        if observed != current or report["status"] != "counted_all_inputs":
            return {
                "status": "historical_inputs_changed",
                "command": "python -m src.cluster.token_audit",
                "note": "Saved counts are preserved, but do not validate the active prepared inputs.",
            }, pd.DataFrame()
        rows.append(
            {
                "provider": report["provider"],
                "unique_inputs": report["unique_inputs"],
                "total_input_tokens": report["total_input_tokens"],
                "maximum_input_tokens": max(d["max_tokens"] for d in report["datasets"]),
                "context_limit": report["context_limit"],
                "overflowing_unique_inputs": report["overflow_unique_inputs"],
                "affected_source_rows": report["overflow_source_rows"],
                "audit_id": report["audit_id"],
            }
        )
    return {
        "status": "recorded_exact_counts_for_current_inputs",
        "created_utc": combined["created_utc"],
        "datasets": len(current),
        "complete_publisher_prompt_included": True,
        "hosted_unverified": HOSTED_UNAVAILABLE,
        "pending_self_hosted": sorted(
            set(SELF_HOSTED) - {r["provider"] for r in combined["self_hosted"]}
        ),
        "scope": "Pinned publisher tokenizers and rendering; real GPU runtime parity still requires the pilot.",
        "rows_removed": False,
        "inference_executed": False,
    }, pd.DataFrame(rows)


def audit_provider(provider, inputs=None, *, counter=None, batch_size=256):
    """Transactional batches and immutable input/counter identities allow safe restart."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if counter is None:
        counter = ExactCounter(provider)
        if counter.limit != load_yaml("extractors")["candidates"][provider]["context_tokens"]:
            raise ValueError("Configured context window differs from the pinned tokenizer")
    inputs = selected_inputs() if inputs is None else inputs
    recipe = {
        "schema": "jevpfn.exact_tokens.v1",
        "provider": provider,
        "counter": counter.identity,
        "inputs": [{k: v for k, v in i.items() if k != "path"} for i in inputs],
    }
    audit_id = digest(recipe)
    destination = (
        paths.require_project_output(paths.token_audit_dir()) / f"{provider}_{audit_id}.sqlite3"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    log = (
        paths.logs_dir("experiment_0")
        / "before_extraction"
        / f"tokens_{provider}_{audit_id[:12]}.log"
    )
    log.parent.mkdir(parents=True, exist_ok=True)

    def progress(message):
        print(message, flush=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"{datetime.now(UTC).isoformat()} {message}\n")

    summaries = []
    started = time.perf_counter()
    # Hold the lock before reading the saved prefix, not just during INSERT.
    # SQLite transactions alone allow two jobs to count the same next batch.
    with (
        _exclusive_audit(destination.with_suffix(".lock")),
        closing(sqlite3.connect(destination, timeout=60)) as db,
    ):
        db.executescript("""
            CREATE TABLE IF NOT EXISTS identity (id INTEGER PRIMARY KEY CHECK(id=1), recipe_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS counts (
                dataset_id TEXT NOT NULL, input_no INTEGER NOT NULL, payload_sha256 TEXT NOT NULL,
                input_tokens INTEGER NOT NULL CHECK(input_tokens>0),
                PRIMARY KEY(dataset_id,input_no)) WITHOUT ROWID;
        """)
        stored = db.execute("SELECT recipe_json FROM identity WHERE id=1").fetchone()
        if stored and stored[0] != canonical_json(recipe):
            raise ValueError("Token audit identity differs from its saved counts")
        db.execute("INSERT OR IGNORE INTO identity VALUES (1,?)", (canonical_json(recipe),))
        db.commit()
        if hasattr(counter, "attach_cache"):
            counter.attach_cache(db, writable=True)
        for item in inputs:
            dataset_id = item["dataset_id"]
            n, last = db.execute(
                "SELECT COUNT(*),MAX(input_no) FROM counts WHERE dataset_id=?", (dataset_id,)
            ).fetchone()
            if n and n != last + 1 or n > item["unique_inputs"]:
                raise ValueError(
                    "Saved token audit is not a contiguous prefix of the prepared inputs"
                )
            progress(
                f"{provider} / {dataset_id}: {n:,}/{item['unique_inputs']:,} counted; resuming"
            )
            records = iter_prepared_inputs(
                item["path"], input_set_id=item["input_set_id"], after_input_no=last if n else -1
            )
            while batch := list(islice(records, batch_size)):
                lengths = counter.count_many([r["payload"] for r in batch])
                if len(lengths) != len(batch) or any(
                    isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in lengths
                ):
                    raise ValueError(
                        "Tokenizer must return one positive integer per nonempty input"
                    )
                with db:
                    db.executemany(
                        "INSERT INTO counts VALUES (?,?,?,?)",
                        [
                            (dataset_id, r["input_no"], r["payload_sha256"], count)
                            for r, count in zip(batch, lengths, strict=True)
                        ],
                    )
                old, n = n, n + len(batch)
                if n // 25000 != old // 25000:
                    progress(
                        f"{provider} / {dataset_id}: {n:,}/{item['unique_inputs']:,} exact counts"
                    )
            if n != item["unique_inputs"]:
                raise ValueError("Prepared input count differs from the saved input-set summary")
            lengths = np.array(
                [
                    r[0]
                    for r in db.execute(
                        "SELECT input_tokens FROM counts WHERE dataset_id=?", (dataset_id,)
                    )
                ],
                dtype=np.int64,
            )
            overflows = []
            affected_rows = set()
            for number, sha, count in db.execute(
                "SELECT input_no,payload_sha256,input_tokens FROM counts WHERE dataset_id=? AND input_tokens>? ORDER BY input_no",
                (dataset_id, counter.limit),
            ):
                origins = list(
                    input_origins(item["path"], number, input_set_id=item["input_set_id"])
                )
                affected_rows.update(o["row_id"] for o in origins)
                overflows.append(
                    {
                        "input_no": number,
                        "payload_sha256": sha,
                        "input_tokens": count,
                        "origins": origins,
                    }
                )
            # Detailed origins may grow large; they belong inside the PROJECT database.
            db.execute(
                "CREATE TABLE IF NOT EXISTS dataset_summaries (dataset_id TEXT PRIMARY KEY, summary_json TEXT NOT NULL)"
            )
            summary = {
                **{k: v for k, v in item.items() if k != "path"},
                "total_input_tokens": int(lengths.sum()),
                "mean_tokens": float(lengths.mean()) if n else 0,
                "median_tokens": float(np.median(lengths)) if n else 0,
                "p99_tokens": float(np.quantile(lengths, 0.99)) if n else 0,
                "max_tokens": int(lengths.max()) if n else 0,
                "overflow_unique_inputs": len(overflows),
                "overflow_source_rows": len(affected_rows),
            }
            with db:
                db.execute(
                    "INSERT OR REPLACE INTO dataset_summaries VALUES (?,?)",
                    (dataset_id, canonical_json({**summary, "overflows": overflows})),
                )
            summaries.append(summary)
            progress(
                f"{provider} / {dataset_id}: complete; max={summary['max_tokens']:,}; {len(overflows):,} overflowing inputs / {len(affected_rows):,} rows"
            )
        if provider in {"h2o", "sii", "kev", "decision3", "nimble"}:
            from src.jev.registry import require_prepared_coverage
            require_prepared_coverage(db)
    report = {
        "schema": recipe["schema"],
        "audit_id": audit_id,
        "provider": provider,
        "status": "counted_all_inputs",
        "exact_token_counts": True,
        "complete_publisher_prompt_included": True,
        "model_weights_loaded": False,
        "inference_executed": False,
        "rows_removed": False,
        "counter_id": counter.identity["counter_id"],
        "context_limit": counter.limit,
        "created_utc": datetime.now(UTC).isoformat(),
        "seconds_this_run": time.perf_counter() - started,
        "counts_database": str(destination),
        "datasets": summaries,
        "unique_inputs": sum(s["unique_inputs"] for s in summaries),
        "total_input_tokens": sum(s["total_input_tokens"] for s in summaries),
        "overflow_unique_inputs": sum(s["overflow_unique_inputs"] for s in summaries),
        "overflow_source_rows": sum(s["overflow_source_rows"] for s in summaries),
    }
    write_json(
        paths.manifests_dir("experiment_0")
        / "before_extraction"
        / f"tokens_{provider}_{audit_id[:12]}.json",
        report,
    )
    return report


def check_tokenizers(providers):
    """Render all task types before scheduling expensive counts; no weights/data writes."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    reports, failures = [], []
    questions = [
        {"kind": "binary_probability", "text": "Does the supplied text express approval?"},
        {"kind": "choice", "text": "Which topic is supported?", "allowed_answers": ["a", "b", "c"]},
        {
            "kind": "ordered_choice",
            "text": "Does the text suggest a lower or higher value?",
            "allowed_answers": [{"score": i, "meaning": str(i)} for i in range(-4, 5)],
        },
    ]
    for provider in providers:
        try:
            counter = ExactCounter(provider)
            counts = counter.count_many(
                [
                    {"state": {"text": {"body": "Résumé: 世界. A short test."}}, "question": q}
                    for q in questions
                ]
            )
            if len(counts) != 3 or any(type(n) is not int or n <= 0 for n in counts):
                raise ValueError("Expected positive counts for all three task types")
            reports.append(
                {
                    "provider": provider,
                    "task_counts": counts,
                    "counter_id": counter.identity["counter_id"],
                }
            )
            print(f"{provider}: tokenizer and all three request formats OK", flush=True)
        except Exception as exc:
            failures.append(f"{provider}: {type(exc).__name__}: {exc}")
    if failures:
        raise RuntimeError(
            "Tokenizer precheck failed before full counting. Install the token-audit extra "
            "(including Jinja2), then stage this provider's tokenizer files.\n"
            + "\n".join(failures)
        )
    return reports


def save_reports(reports):
    """Publish completed providers even if a different worker subsequently fails."""
    report_path = paths.manifests_dir("experiment_0") / "before_extraction" / "token_audit.json"
    # Atomic replacement prevents torn JSON; the lock also prevents lost updates.
    with file_lock(report_path.with_suffix(".lock"), timeout=60):
        retained = {}
        if report_path.exists():
            retained = {
                r["provider"]: r
                for r in json.loads(report_path.read_text(encoding="utf-8"))["self_hosted"]
            }
        retained.update({r["provider"]: r for r in reports})
        write_json(
            report_path,
            {
                "created_utc": datetime.now(UTC).isoformat(),
                "stage": "experiment_0_before_extraction",
                "self_hosted": list(retained.values()),
                "hosted_unverified": HOSTED_UNAVAILABLE,
                "all_selected_models_verified": False,
                "rows_removed": False,
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage-tokenizers",
        action="store_true",
        help="Download only small pinned public tokenizer/source files before counting",
    )
    parser.add_argument(
        "--stage-only",
        action="store_true",
        help="Download tokenizer files on a connected setup host, without counting",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Quick offline tokenizer/request-format checks; allowed on the setup host",
    )
    parser.add_argument(
        "--provider",
        choices=SELF_HOSTED,
        action="append",
        help="Default: all ten self-hosted tokenizers",
    )
    parser.add_argument("--independent", action="store_true", help="One provider per CPU job; provider database and report locks still apply")
    args = parser.parse_args()
    from src.cluster.preflight import storage_policy_failures

    failures = storage_policy_failures(os.environ.get("SLURM_CLUSTER_NAME"))
    if failures:
        raise ValueError("; ".join(failures))
    providers = list(dict.fromkeys(args.provider or SELF_HOSTED))
    if args.independent and len(providers) != 1:
        parser.error("--independent requires exactly one --provider")
    if args.stage_only:
        stage_tokenizers(providers)
        return 0
    if paths.on_vsc() and not os.environ.get("SLURM_JOB_ID") and not args.check_only:
        raise RuntimeError(
            "Full token counting on VSC requires a CPU allocation; stage files on the login host separately"
        )
    if args.check_only:
        if args.stage_tokenizers:
            stage_tokenizers(providers)
        check_tokenizers(providers)
        return 0
    if args.independent:
        return _count_selected(providers)
    lock = paths.require_project_output(paths.token_audit_dir()) / ".token_audit.lock"
    with _exclusive_audit(lock):
        if args.stage_tokenizers:
            stage_tokenizers(providers)
        return _count_selected(providers)


def _count_selected(providers):
    """Run one coordinator; independent provider workers still execute in parallel."""
    check_tokenizers(providers)
    inputs = selected_inputs()
    reports, failures = [], []
    with ProcessPoolExecutor(
        max_workers=min(
            len(providers), int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
        )
    ) as pool:
        futures = {pool.submit(audit_provider, p, inputs): p for p in providers}
        for future in as_completed(futures):
            try:
                reports.append(future.result())
                save_reports(reports)
            except Exception as exc:
                failure = {
                    "provider": futures[future],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                failures.append(failure)
                print(f"Tokenizer worker failed: {failure}", flush=True)
    if failures:
        write_json(
            paths.manifests_dir("experiment_0")
            / "before_extraction"
            / f"token_audit_failure_{uuid4().hex}.json",
            {
                "created_utc": datetime.now(UTC).isoformat(),
                "failures": failures,
                "completed_providers": [r["provider"] for r in reports],
                "committed_counts_preserved": True,
            },
        )
        raise RuntimeError(
            "Token audit failed; completed providers and committed count batches were preserved"
        )
    print(
        json.dumps(
            {
                "providers": [
                    {
                        k: r[k]
                        for k in (
                            "provider",
                            "unique_inputs",
                            "total_input_tokens",
                            "overflow_unique_inputs",
                            "overflow_source_rows",
                        )
                    }
                    for r in reports
                ],
                "hosted_unverified": HOSTED_UNAVAILABLE,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
