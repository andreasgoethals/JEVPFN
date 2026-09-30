"""Notebook presentation helpers; deterministic dataset and row selection, no labels."""

from __future__ import annotations

import copy

from src.data.loaders import load_catalog, load_dataset
from src.jev.requests import build_requests
from src.utils import paths
from src.utils.provenance import code_identity
from src.utils.serialization import write_json


def representative_requests(audit, cfg: dict) -> list:
    # Prefer small displays. Sorting uses task and feature metadata, never target outcomes.
    selected = (
        audit.summary.sort_values(["text_columns", "rows", "dataset_id"])
        .groupby("task_type", sort=True)
        .head(1)
    )
    catalog = {d["id"]: d for d in load_catalog(cfg["catalog"])["datasets"]}
    requests = []
    for dataset_id in selected.dataset_id:
        dataset = load_dataset(catalog[dataset_id], allow_download=cfg["allow_download"])
        features = dataset.features
        for mode in ("per_column", "joint"):
            for structured in (False,):
                variant = copy.deepcopy(cfg)
                variant["jev"]["representation"] = mode
                variant["jev"]["include_non_text_features"] = structured
                requests.extend(
                    build_requests(
                        features,
                        dataset.metadata,
                        0,
                        variant,
                        source=dataset.source,
                        code_version=code_identity(),
                    )
                )
    write_json(
        paths.results_dir("jev_design", "representative_requests.json"),
        [r.to_dict() for r in requests],
    )
    return requests


def design_summary(audit, requests, totals, cfg) -> str:
    tasks = {r.to_dict()["payload"]["state"]["task"]["task_type"] for r in requests}
    return "\n".join(
        [
            "1. Inputs and scope",
            f"The actual full audit covers {len(audit.summary)}/20 datasets. No Jev API is enabled.",
            "2. Deterministic task templates",
            "Binary: canonical last class is the positive class; full class vocabulary is metadata-only.",
            "Multiclass: complete ordered class choices. Regression: fixed -4 to +4 directional scale.",
            "3. Representative request states",
            f"{len(requests)} intended requests cover {', '.join(sorted(tasks))}.",
            "Each text column separately and all available text columns together; no non-text features.",
            "No row targets, label frequencies, target quantiles, fitted models or metrics enter the builder.",
            "4. Workload and configurable cost",
            totals.to_string(index=False),
            f"Price assumption: {cfg['cost']['input_price_per_million']} {cfg['cost']['currency']} per million input tokens.",
            "Price is a dated, configurable documentation reference; null means unpriced. Verify before paid execution.",
            f"Characters/{cfg['token_estimation']['characters_per_token']:g} approximation includes full intended task/question/state JSON. No truncation.",
            "Empty inputs skipped; remaining request counts precede deduplication; output tokens, unknown wrappers and retries are excluded.",
            "5. Cache and open choices",
            "The mock acknowledgement uses a temporary cache, removed when the demonstration finishes; no persistent cache is created.",
            "Current API documentation is mapped in Notebook 3; live validation, final feature views, context policy and evaluation remain open.",
            f"Audit and workload: {audit.folder}",
            f"Request previews: {paths.results_dir('jev_design', 'representative_requests.json')}",
        ]
    )
