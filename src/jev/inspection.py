"""Notebook presentation helpers; deterministic dataset and row selection, no labels."""

from __future__ import annotations

import copy

from src.data.loaders import load_catalog, load_dataset
from src.data.selection import selected_entries
from src.jev.question_bank import build_question_bank, compile_frozen_question
from src.jev.requests import build_requests
from src.utils.config import load_yaml
from src.utils.provenance import code_identity


def representative_requests(audit, cfg: dict) -> list:
    # Prefer small displays. Sorting uses task and feature metadata, never target outcomes.
    catalog = {d["id"]: d for d in selected_entries(load_catalog(cfg["catalog"]), cfg)}
    selected = (
        audit.summary[audit.summary.dataset_id.isin(catalog)]
        .sort_values(["text_columns", "rows", "dataset_id"])
        .groupby("task_type", sort=True)
        .head(1)
    )
    requests = []
    bank = build_question_bank([audit.metadata(i) for i in catalog], load_yaml("question_design"))
    version = code_identity()
    for dataset_id in selected.dataset_id:
        dataset = load_dataset(catalog[dataset_id], allow_download=cfg["allow_download"])
        features = dataset.features
        apply = compile_frozen_question(dataset.metadata, bank, "documented")
        for mode in ("per_column", "joint"):
            for structured in (False,):
                variant = copy.deepcopy(cfg)
                variant["jev"]["representation"] = mode
                variant["jev"]["include_non_text_features"] = structured
                requests.extend(
                    apply(request)
                    for request in build_requests(
                        features,
                        dataset.metadata,
                        0,
                        variant,
                        source=dataset.source,
                        code_version=version,
                    )
                )
    return requests
