"""All-selected-dataset request gallery. Local display only; no inference transports."""

import copy
import html
import json

import pandas as pd

from src.data.loaders import load_catalog, load_dataset
from src.data.selection import selected_entries
from src.jev.alternatives import preview
from src.jev.identity import bind_provider
from src.jev.provider import semantic_question
from src.jev.question_bank import apply_frozen_question
from src.jev.requests import build_requests
from src.utils.notebook_report import record
from src.utils.provenance import code_identity
from src.utils.serialization import text_value


def question_overview(bank):
    """One applicable question per dataset; never invent three targets per dataset."""
    return pd.DataFrame(
        {
            "dataset": entry["task"]["dataset"],
            "task": entry["task"]["task_type"],
            "target": entry["task"]["target"],
            "text_columns": len(entry["text_columns"]),
            "main_numeric_features_per_input": len(entry["task"]["class_labels"])
            if entry["task"]["task_type"] == "multiclass_classification"
            else 1,
            "documented_question": entry["variants"].get(
                "documented", entry["variants"]["proposed_direct"]
            )["question"]["text"],
        }
        for entry in bank["datasets"].values()
    )


def dataset_example(dataset, cfg, bank, catalog):
    """Choose the first maximally complete text row without looking at its target.

    Preserve every original text value exactly; displayed JSON is the full value.
    All per-column slots plus the joint slot are shown, with skips/reuse explicit.
    """
    features, meta = dataset.features, dataset.metadata
    counts = pd.concat(
        [features[col].map(lambda x: text_value(x) is not None) for col in meta.text_columns],
        axis=1,
    ).sum(axis=1)
    position = int(counts.to_numpy().argmax())
    config = copy.deepcopy(cfg)
    config["jev"]["representation"] = "per_column_and_joint"
    requests = build_requests(
        features, meta, position, config, source=dataset.source, code_version=code_identity()
    )
    groups = []
    for accepted in requests:
        trace = json.loads(accepted.provenance_json)
        variants = {}
        for variant in bank["datasets"][meta.dataset_id]["variants"]:
            request = apply_frozen_question(accepted, meta, bank, variant)
            variants[variant] = {
                "semantic_payload": json.loads(request.payload_json),
                "semantic_question": semantic_question(
                    json.loads(request.payload_json)["question"]
                ),
                "provider_previews": {
                    provider: {
                        **preview(request, provider, deployment_name="<azure-deployment-name>"),
                        "cache_identity": json.loads(
                            bind_provider(request, provider, catalog).settings_json
                        ),
                        "request_key": bind_provider(request, provider, catalog).cache_key,
                    }
                    for provider in catalog["candidates"]
                },
            }
        groups.append(
            {
                "source_text_columns": trace["source_text_columns"],
                "included_text_columns": trace["text_columns"],
                "mode": "joint"
                if trace["representation_slot"] == len(meta.text_columns)
                else "per_column",
                "variants": variants,
            }
        )
    return {
        "dataset_id": meta.dataset_id,
        "dataset": meta.dataset_name,
        "task": meta.task_type,
        "row_id": f"csv:{position}",
        "data_identity": dataset.source["data_identity"],
        "question_bank_id": bank["bank_id"],
        "selection": "first row with the most nonempty official text columns; no row target used",
        "empty_text_columns": [
            c for c in meta.text_columns if text_value(features.iloc[position][c]) is None
        ],
        "groups": groups,
        "runtime_verified": False,
        "notice": "Exact constructed bodies/arguments, not live-verified requests. Angle-bracket account/deployment/credential placeholders require real configuration. No actual credentials stored.",
    }


def selected_examples(cfg, bank, catalog):
    for entry in selected_entries(load_catalog(cfg["catalog"]), cfg):
        yield dataset_example(load_dataset(entry, allow_download=False), cfg, bank, catalog)


def show_dataset_example(example):
    """Nested details keep 18 datasets and seven providers readable without truncation."""
    from IPython.display import HTML, display

    escape = html.escape
    details = []
    for group in example["groups"]:
        variants = []
        for name, variant in group["variants"].items():
            body = json.dumps(variant["semantic_payload"], ensure_ascii=False, indent=2)
            providers = "".join(
                f"<details><summary>{escape(provider)}: complete call specification</summary><pre>"
                + escape(json.dumps(mapping, ensure_ascii=False, indent=2))
                + "</pre></details>"
                for provider, mapping in variant["provider_previews"].items()
            )
            variants.append(
                f"<details><summary>{escape(name)}: question, actual text and all seven calls</summary>"
                f"<pre>{escape(body)}</pre>{providers}</details>"
            )
        title = group["mode"] + ": " + ", ".join(group["source_text_columns"])
        details.append(f"<details><summary>{escape(title)}</summary>{''.join(variants)}</details>")
    variants = example["groups"][0]["variants"] if example["groups"] else {}
    first = (
        variants.get("documented", variants["proposed_direct"])["semantic_payload"]
        if example["groups"]
        else None
    )
    question = first["question"]["text"] if first else "All text is empty; skip inference."
    display(
        HTML(
            f"<h3>{escape(example['dataset'])} — {escape(example['task'])}</h3>"
            f"<p><b>Question for review:</b> {escape(question)}</p>"
            f"<p>Source {escape(example['row_id'])}, kept only in local provenance. "
            "The text in every request below describes this same row. No target value or non-text feature is sent.</p>"
            f"<p>{escape(example['notice'])}</p>" + "".join(details)
        )
    )
    record(example, f"All exact request previews — {example['dataset']}")
