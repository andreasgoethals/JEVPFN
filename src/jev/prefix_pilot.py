"""Bounded local Jev validation on the first 100 rows; never full extraction.

Reuse the production dispatcher and response keys. Prefix feature artifacts have a
distinct data identity, so full-dataset evaluation cannot accidentally consume them.
"""

import argparse
import copy
import json
import sqlite3
from contextlib import closing
from dataclasses import replace

from src.jev.inputs import _read_set, iter_prepared_inputs
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.serialization import digest


def prefix_numbers(path, input_set_id, rows):
    if type(rows) is not int or not 1 <= rows <= 100:
        raise ValueError("Prefix validation requires 1 to 100 source rows")
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        _, recipe, _, _ = _read_set(db, input_set_id)
        count = min(rows, recipe["row_count"])
        positions = db.execute(
            "SELECT COUNT(DISTINCT row_position) FROM row_map WHERE input_set_id=? AND row_position<?",
            (input_set_id, count),
        ).fetchone()[0]
        if positions != count:
            raise ValueError("Prepared row mapping does not cover the prefix")
        return {
            r[0] for r in db.execute(
                "SELECT DISTINCT input_no FROM row_map WHERE input_set_id=? AND row_position<? AND input_no IS NOT NULL",
                (input_set_id, count),
            )
        }


def selected_records(path, input_set_id, numbers):
    """Respect the input reader's 1,000-ID ceiling without omitting large prefixes."""
    ordered = sorted(numbers)
    for start in range(0, len(ordered), 1000):
        yield from iter_prepared_inputs(
            path, input_set_id=input_set_id, input_numbers=ordered[start:start + 1000]
        )


def ledger_total(path):
    """Read cumulative charges plus reservations across all model versions."""
    if not path.exists():
        return 0.0
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='extraction_jobs'").fetchone():
            return 0.0
        return db.execute(
            "SELECT COALESCE(SUM(spent_usd+reserved_usd),0) FROM extraction_jobs"
        ).fetchone()[0]


def subset(dataset, rows):
    count = min(rows, len(dataset.frame))
    identity = {
        "schema": "jevpfn.validation-prefix.v1",
        "parent_data_identity": dataset.source["data_identity"],
        "rows": count,
        "selection": "first_source_rows",
    }
    return replace(
        dataset, frame=dataset.frame.iloc[:count].copy(),
        source={**dataset.source, "data_identity": digest(identity)},
    ), identity


def materialize(dataset_id, settings, cache, input_set_id, rows):
    from src.data.loaders import load_catalog, load_dataset
    from src.jev.features import materialize_cached_block, save_block
    from src.jev.question_bank import load_question_bank

    entry = next(e for e in load_catalog()["datasets"] if e["id"] == dataset_id)
    dataset = load_dataset(entry, allow_download=False)
    with closing(sqlite3.connect(paths.dataset_question_path(dataset_id).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        _, recipe, _, _ = _read_set(db, input_set_id)
    if (dataset.source["data_identity"] != recipe["data_identity"]
            or dataset.metadata.fingerprint != recipe["task_schema_sha256"]):
        raise ValueError("Raw data differs from the prepared input identity")
    selected, identity = subset(dataset, rows)
    bank = load_question_bank(paths.decision_inputs_dir(), bank_id=recipe["bank_id"])
    artifacts = []
    for mode in ("per_column", "joint"):
        cfg = load_yaml("feature_creation")
        cfg["jev"]["representation"] = mode
        block = materialize_cached_block(
            selected, cfg, cache, question_bank=bank,
            question_variant=recipe["question_variant"], provider_settings=settings,
            phase="experiment_0",
        )
        block.manifest.update(purpose="validation_only", row_selection=identity)
        save_block(block)
        artifacts.append({"artifact_id": block.manifest["artifact_id"], "mode": mode, "rows": len(block.values)})
    return artifacts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--config", default="feature_creation/jev.local")
    args = parser.parse_args(argv)
    policy = load_yaml("experiment_0/jev_prefix_pilot")
    from src.cluster.token_audit import selected_inputs

    plan = [{"dataset_id": i["dataset_id"], "rows": policy["rows"], "unique_requests": len(prefix_numbers(
        paths.dataset_question_path(i["dataset_id"]), i["input_set_id"], policy["rows"]
    ))} for i in selected_inputs()]
    print(json.dumps({"scope": "validation_only", "datasets": plan, "policy": policy}), flush=True)
    if not args.execute:
        return 0
    cfg = copy.deepcopy(load_yaml(args.config))
    if cfg["live_extraction"].get("full_feature_creation_enabled"):
        raise ValueError("Use the bounded pilot with full extraction disabled")
    cfg["live_extraction"]["requests_per_second"] = policy["requests_per_second"]
    cfg["live_extraction"]["workers"] = 1
    from src.jev.hosted import run

    report = run(cfg, execute=True, prefix_policy=policy)
    print(json.dumps({"run_id": report["run_id"], "status": report["status"]}))
    return int(report["status"] != "completed_prefix_pilot_requires_review")


if __name__ == "__main__":
    raise SystemExit(main())
