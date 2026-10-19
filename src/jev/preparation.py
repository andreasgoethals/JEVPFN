"""Offline question preparation: freeze questions, deduplicate text, retain every row link."""

import argparse
import json
from datetime import UTC, datetime
from uuid import uuid4

import pandas as pd

from src.data.loaders import load_catalog, load_dataset
from src.data.selection import selected_entries
from src.jev.inputs import activate_input_set, prepare_dataset_inputs, prepared_row
from src.jev.question_bank import build_question_bank, save_question_bank
from src.utils import paths
from src.utils.config import load_config, load_yaml
from src.utils.serialization import write_json


def prepare_questions(cfg=None, *, progress=print, rebuild=False):
    """Reads only existing pinned datasets; no downloads, provider access or model imports."""
    cfg = cfg or load_config("question_preparation")
    settings = cfg.get(
        "question_preparation", load_yaml("question_preparation")["question_preparation"]
    )
    required = {
        "enabled": True,
        "canonical_storage": "data/decision_inputs",
        "storage_format": "one_sqlite_file_per_dataset",
        "input_modes": ["per_column", "joint"],
        "equality": "exact_question_task_named_text_values",
        "missing_text_policy": "skip_empty_store_nan",
        "text_policy": "verbatim_no_truncation",
        "keep_prior_versions_in_dataset_file": True,
    }
    if any(settings.get(key) != value for key, value in required.items()):
        raise ValueError(
            "Unsupported question-preparation policy; review before changing saved inputs"
        )
    entries = selected_entries(load_catalog(cfg["catalog"]), cfg)
    metadata = [load_dataset(e, allow_download=False).metadata for e in entries]
    bank = build_question_bank(metadata, load_yaml("question_design"))
    if any("documented" not in e["variants"] for e in bank["datasets"].values()):
        raise ValueError("Every selected dataset needs a source-backed question before preparation")
    path = save_question_bank(bank)
    run_id = uuid4().hex
    log_path = paths.ensure(paths.logs_dir("question_preparation") / f"prepare_{run_id}.log")
    summaries = []
    with log_path.open("w", encoding="utf-8") as log:

        def report(message):
            log.write(f"{datetime.now(UTC).isoformat()} {message}\n")
            log.flush()
            if progress:
                progress(message)

        try:
            for i, entry in enumerate(entries, 1):
                report(f"[{i}/{len(entries)}] {entry['name']}: preparing exact text inputs")
                dataset = load_dataset(entry, allow_download=False)
                summary = prepare_dataset_inputs(
                    dataset,
                    bank,
                    variant=settings["variant"],
                    progress=report,
                    rebuild=rebuild,
                )
                summaries.append(summary)
                report(
                    f"  {'REUSED' if summary['reused'] else 'SAVED'}: "
                    f"{summary['unique_inputs']:,} unique inputs; "
                    f"{summary['reused_slots']:,} nonempty slots avoided "
                    f"({summary['avoided_percent']:.1f}%)"
                )
        except BaseException as exc:
            report(f"FAILED: {type(exc).__name__}: {exc}")
            raise
    # Preparation is the explicit activation step. Consumers never choose the latest
    # timestamp; historical versions remain addressable by their immutable IDs.
    for summary in summaries:
        activate_input_set(summary["path"], summary["input_set_id"])
    write_json(
        paths.manifests_dir("question_preparation") / f"prepare_{run_id}.json",
        {
            "bank_id": bank["bank_id"],
            "log": str(log_path),
            "datasets": summaries,
            "execution": "offline_no_inference",
            "input_files": len(summaries),
        },
    )
    return bank, path, summaries


def preparation_tables(bank):
    """Readable complete metadata: raw class codes remain alongside their descriptions."""
    datasets, classes, fields, sources, notes = [], [], [], [], []
    for key, entry in bank["datasets"].items():
        facts = entry["semantics"]
        datasets.append(
            {
                "dataset_id": key,
                "dataset": entry["task"]["dataset"],
                "meaning": facts["dataset_display_name"],
                "task": entry["task"]["task_type"],
                "raw_target": facts["raw_target"],
                "target_meaning": facts["target_definition"],
                "classes": len(facts["classes"]),
                "text_columns": len(entry["text_columns"]),
                "review_notes": len(facts["review_notes"]),
                "question": entry["variants"]["documented"]["question"]["text"],
            }
        )
        classes.extend(
            {
                "dataset": entry["task"]["dataset"],
                "option_id": f"c{i:03d}",
                "raw_value": item["value"],
                "meaning": item["description"],
            }
            for i, item in enumerate(facts["classes"])
        )
        fields.extend(
            {"dataset": entry["task"]["dataset"], "raw_column": col, "meaning": meaning}
            for col, meaning in facts["text_column_definitions"].items()
        )
        sources.extend(
            {"dataset": entry["task"]["dataset"], **source} for source in facts["sources"]
        )
        notes.extend(
            {"dataset": entry["task"]["dataset"], "review_note": note}
            for note in facts["review_notes"]
        )
    return {
        "datasets": pd.DataFrame(datasets),
        "classes": pd.DataFrame(classes),
        "text_fields": pd.DataFrame(fields),
        "sources": pd.DataFrame(sources),
        "review_notes": pd.DataFrame(notes),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preview",
        metavar="DATASET_ID",
        help="Print an existing exact row input; do not regenerate",
    )
    parser.add_argument(
        "--row", type=int, default=0, help="Original zero-based CSV row for --preview"
    )
    parser.add_argument("--input-set-id", help="Explicit version when a dataset file holds several")
    parser.add_argument("--rebuild", action="store_true", help="Recompute active inputs transactionally; preserve prior versions and all features")
    args = parser.parse_args(argv)
    if args.preview and args.rebuild:
        parser.error("--preview and --rebuild are mutually exclusive")
    if args.preview:
        print(
            json.dumps(
                prepared_row(
                    paths.dataset_question_path(args.preview),
                    args.row,
                    input_set_id=args.input_set_id,
                ),
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    bank, path, summaries = prepare_questions(rebuild=args.rebuild)
    print(f"Prepared {len(summaries)} dataset files directly in {path}")
    print(f"Question bank: {bank['bank_id']}")
    print(f"Unique inputs per extractor: {sum(s['unique_inputs'] for s in summaries):,}")
    print(
        "No API calls or models. Files contain text inputs and row links, never true row targets."
    )


if __name__ == "__main__":
    main()
