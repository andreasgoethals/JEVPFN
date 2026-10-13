"""Offline question preparation: validate meanings, freeze inputs, export review copies."""

import pandas as pd

from src.data.loaders import load_catalog, load_dataset
from src.data.selection import selected_entries
from src.jev.question_bank import build_question_bank, export_dataset_questions, save_question_bank
from src.utils.config import load_config, load_yaml


def prepare_questions(cfg=None):
    """Reads only existing pinned datasets; no downloads, provider access or model imports."""
    cfg = cfg or load_config("question_preparation")
    metadata = [
        load_dataset(e, allow_download=False).metadata
        for e in selected_entries(load_catalog(cfg["catalog"]), cfg)
    ]
    bank = build_question_bank(metadata, load_yaml("question_design"))
    if any("documented" not in e["variants"] for e in bank["datasets"].values()):
        raise ValueError("Every selected dataset needs a source-backed question before preparation")
    path = save_question_bank(bank)
    exports = export_dataset_questions(bank)
    return bank, path, exports


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


def main():
    bank, path, exports = prepare_questions()
    print(f"Prepared {len(exports)} dataset question files; bank {bank['bank_id']}")
    print(f"Reusable inputs: {path}")
    print(f"Review exports: {exports[0].parent}")
    print("No API calls, model loads, raw text, row targets or label statistics.")


if __name__ == "__main__":
    main()
