"""Offline budget review over real feature values; never sends requests.

Run ``python -m src.jev.review`` after preparing the pinned datasets. Count exact
duplicates of the current semantic inputs, and estimate the documented JSON body's
length at configurable character/token ratios. These are not billing-token counts.
"""

from __future__ import annotations

import copy
import json
import sqlite3
from contextlib import closing
from dataclasses import replace

import numpy as np
import pandas as pd

from src.data.audit import request_lengths
from src.data.loaders import load_catalog, load_dataset
from src.data.selection import selected_entries
from src.data.text_reuse import semantic_codes
from src.jev.cost import group_lengths
from src.jev.inputs import _read_set
from src.jev.provider import api_preview
from src.jev.question_bank import payload_from_text
from src.jev.requests import build_requests, text_groups
from src.utils import paths
from src.utils.config import load_config, load_yaml
from src.utils.serialization import canonical_json


def _wire_body(request, entry, question_variant):
    if entry is not None:
        text = json.loads(request.payload_json)["state"]["text"]
        request = replace(
            request, payload_json=canonical_json(payload_from_text(entry, question_variant, text))
        )
    return api_preview(request)


def _group_sizes(metadata, lengths, group, body):
    """Count the exact wire characters, including definitions of present fields only."""
    body["state"]["text"] = {}
    definitions = (
        body["state"]["task"].get("documentary_definition", {}).get("text_column_definitions")
    )
    field_sizes = (
        {c: len(canonical_json({c: v})) - 2 for c, v in definitions.items()}
        if definitions is not None
        else {}
    )
    if definitions is not None:
        definitions.clear()
    chars, present = group_lengths(metadata, lengths, group, len(canonical_json(body)))
    for column, size in field_sizes.items():
        chars += np.where(lengths[f"present_{metadata.text_columns.index(column)}"], size, 0)
    if field_sizes:
        chars += np.maximum(present - 1, 0)  # Commas between supplied field definitions.
    return chars, present


def review_dataset(
    dataset, cfg: dict, *, question_entry=None, question_variant="accepted"
) -> tuple[list[dict], list[dict]]:
    features, metadata = dataset.features, dataset.metadata
    lengths = request_lengths(dataset)
    codes = semantic_codes(features.loc[:, list(metadata.text_columns)])
    positions = sorted({0, len(features) // 2, len(features) - 1})
    ratios = cfg["budget_review"]["characters_per_token_scenarios"]
    price = cfg["cost"]["input_price_per_million"]
    windows = sorted(
        {
            4096,
            8192,
            16384,
            24576,
            32000,
            32768,
            *(
                v["context_tokens"]
                for v in load_yaml("extractors")["candidates"].values()
                if v["context_tokens"] is not None
            ),
        }
    )
    estimates, pilot = [], []
    for context in (False,):
        for mode in ("per_column", "joint"):
            variant = copy.deepcopy(cfg)
            variant["jev"].update(representation=mode, include_non_text_features=context)
            for slot, group in enumerate(text_groups(metadata, mode)):
                # Obtain a documentation-mapped question from a synthetic nonempty text value.
                # It is discarded before measuring real inputs and never sent to a provider.
                preview_features = features.iloc[:1].copy()
                for col in metadata.text_columns:
                    preview_features[col] = "preview"
                preview = build_requests(preview_features, metadata, 0, variant)[slot]
                body = _wire_body(preview, question_entry, question_variant)
                chars, present = _group_sizes(metadata, lengths, group, body)
                active = present > 0
                keep = active & ~codes[list(group)].duplicated().to_numpy()
                # A joint request with only one available field equals that field's per-column request.
                combined_keep = keep & (present > 1) if mode == "joint" else keep
                for ratio in ratios:
                    tokens = np.ceil(chars / ratio).astype(np.int64)
                    unique_tokens = int(tokens[keep].sum())
                    estimates.append(
                        {
                            "dataset": metadata.dataset_name,
                            "dataset_id": metadata.dataset_id,
                            "question_variant": question_variant,
                            "mode": mode,
                            "include_non_text_features": context,
                            "text_columns": canonical_json(list(group)),
                            "characters_per_token": ratio,
                            "logical_requests": int(active.sum()),
                            "skipped_empty_slots": int((~active).sum()),
                            "unique_requests_within_group": int(keep.sum()),
                            "requests_for_combined_build": int(combined_keep.sum()),
                            "wire_tokens_logical_approx": int(tokens[active].sum()),
                            "wire_tokens_unique_approx": unique_tokens,
                            "wire_tokens_combined_approx": int(tokens[combined_keep].sum()),
                            "wire_tokens_nonempty_max_approx": int(tokens[active].max())
                            if active.any()
                            else 0,
                            "wire_tokens_nonempty_p99_approx": float(
                                np.quantile(tokens[active], 0.99)
                            )
                            if active.any()
                            else 0,
                            **{
                                f"combined_requests_above_{limit}_approx": int(
                                    (combined_keep & (tokens > limit)).sum()
                                )
                                for limit in windows
                            },
                            "cost_unique_usd": None
                            if price is None
                            else unique_tokens * price / 1e6,
                        }
                    )
            for position in positions:
                for request in build_requests(features, metadata, position, variant):
                    pilot.append(
                        {
                            "dataset": metadata.dataset_name,
                            "row_id": f"csv:{position}",
                            "mode": mode,
                            "include_non_text_features": context,
                            "request_key": request.cache_key,
                            "wire_tokens_approx": int(
                                np.ceil(
                                    len(
                                        canonical_json(
                                            _wire_body(request, question_entry, question_variant)
                                        )
                                    )
                                    / 4
                                )
                            ),
                        }
                    )
    return estimates, pilot


def review_collection(
    cfg: dict, *, include_excluded: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recalculate selected-dataset costs from real values, retaining nothing on disk."""
    records, pilot = [], []
    catalog = load_catalog(cfg["catalog"])
    selected = {e["id"] for e in selected_entries(catalog, cfg)}
    for entry in catalog["datasets"] if include_excluded else selected_entries(catalog, cfg):
        dataset = load_dataset(entry, allow_download=False)
        question_entry, variant = None, "accepted_audit_only"
        if entry["id"] in selected:
            path = paths.dataset_question_path(entry["id"])
            if not path.is_file():
                raise FileNotFoundError(
                    "Run src.jev.preparation before reviewing the extraction workload"
                )
            with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
                _, recipe, _, question_entry = _read_set(db, None)
            if (
                recipe["data_identity"] != dataset.source["data_identity"]
                or recipe["task_schema_sha256"] != dataset.metadata.fingerprint
            ):
                raise ValueError("Prepared questions and loaded dataset identities differ")
            variant = recipe["question_variant"]
        part, sample = review_dataset(
            dataset, cfg, question_entry=question_entry, question_variant=variant
        )
        records.extend(part)
        pilot.extend(sample)
        print(f"Reviewed {entry['name']}", flush=True)
    return pd.DataFrame(records), pd.DataFrame(pilot)


def context_screen_examples(dataset, cfg, *, limit=4096, characters_per_token=3):
    """Trace distinct combined-build inputs above an approximate size threshold.

    Each record is a complete text group (column or joint), not an output class.
    Do not equate this screen with the provider tokenizer or a live API rejection.
    """
    if limit <= 0 or characters_per_token <= 0:
        raise ValueError("Context limit and approximation ratio must be positive")
    metadata, features = dataset.metadata, dataset.features
    lengths = request_lengths(dataset)
    codes = semantic_codes(features.loc[:, list(metadata.text_columns)])
    records = []
    for mode in ("per_column", "joint"):
        variant = copy.deepcopy(cfg)
        variant["jev"]["representation"] = mode
        for slot, group in enumerate(text_groups(metadata, mode)):
            sample = features.iloc[:1].copy()
            for column in metadata.text_columns:
                sample[column] = "preview"
            body = api_preview(build_requests(sample, metadata, 0, variant)[slot])
            body["state"]["text"] = {}
            chars, present = group_lengths(metadata, lengths, group, len(canonical_json(body)))
            keep = (present > 0) & ~codes[list(group)].duplicated().to_numpy()
            if mode == "joint":
                keep &= present > 1
            tokens = np.ceil(chars / characters_per_token).astype(np.int64)
            for position in np.flatnonzero(keep & (tokens > limit)):
                request = next(
                    r
                    for r in build_requests(features, metadata, int(position), variant)
                    if json.loads(r.provenance_json)["source_text_columns"] == list(group)
                )
                trace = request.to_dict()["provenance"]
                records.append(
                    {
                        "dataset": metadata.dataset_name,
                        "dataset_id": metadata.dataset_id,
                        "row_id": trace["row_id"],
                        "mode": mode,
                        "text_columns": list(group),
                        "request_key": request.cache_key,
                        "input_tokens_proxy": int(tokens[position]),
                        "context_reference": limit,
                        "characters_per_token": characters_per_token,
                    }
                )
    return pd.DataFrame(records)


def review_totals(estimates: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    totals = (
        estimates.groupby("characters_per_token")[
            [
                "logical_requests",
                "skipped_empty_slots",
                "requests_for_combined_build",
                "wire_tokens_logical_approx",
                "wire_tokens_combined_approx",
            ]
        ]
        .sum()
        .reset_index()
    )
    price = cfg["cost"]["input_price_per_million"]
    totals["cost_usd"] = None if price is None else totals.wire_tokens_combined_approx * price / 1e6
    return totals


def main() -> None:
    cfg = load_config("feature_creation")
    estimates, pilot = review_collection(cfg)
    totals = review_totals(estimates, cfg)
    sensitivity = []
    price = cfg["cost"]["input_price_per_million"]
    for row in totals.to_dict("records"):
        for overhead in cfg["budget_review"]["extra_tokens_per_unique_request_scenarios"]:
            for retry_fraction in cfg["budget_review"]["billed_retry_fraction_scenarios"]:
                tokens = (
                    row["wire_tokens_combined_approx"]
                    + row["requests_for_combined_build"] * overhead
                ) * (1 + retry_fraction)
                sensitivity.append(
                    {
                        **row,
                        "extra_tokens_per_unique_request": overhead,
                        "billed_retry_fraction": retry_fraction,
                        "input_tokens_scenario": tokens,
                        "cost_usd": None if price is None else tokens * price / 1e6,
                    }
                )
    print(totals.to_string(index=False))
    print(pd.DataFrame(sensitivity).to_string(index=False))
    unique_pilot = pilot.drop_duplicates("request_key")
    print(
        f"Illustrative pilot: {len(unique_pilot):,} unique requests; {int(unique_pilot.wire_tokens_approx.sum()):,} approximate input tokens."
    )
    print("Offline review only; no budget files retained and no requests sent.")


if __name__ == "__main__":
    main()
