"""Offline budget review over real feature values; never sends requests.

Run ``python -m src.jev.review`` after preparing the pinned datasets. Count exact
duplicates of the current semantic inputs, and estimate the documented JSON body's
length at configurable character/token ratios. These are not billing-token counts.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from src.data.audit import request_lengths
from src.data.loaders import load_catalog, load_dataset
from src.data.text_reuse import semantic_codes
from src.jev.cost import group_lengths
from src.jev.provider import api_preview
from src.jev.requests import build_requests, text_groups
from src.utils import paths
from src.utils.config import load_config
from src.utils.provenance import code_identity
from src.utils.serialization import canonical_json, write_json


def review_dataset(dataset, cfg: dict) -> tuple[list[dict], list[dict]]:
    features, metadata = dataset.features, dataset.metadata
    lengths = request_lengths(dataset)
    codes = semantic_codes(features.loc[:, list(metadata.text_columns)])
    positions = sorted({0, len(features) // 2, len(features) - 1})
    ratios = cfg["budget_review"]["characters_per_token_scenarios"]
    price = cfg["cost"]["input_price_per_million"]
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
                body = api_preview(preview)
                body["state"]["text"] = {}
                chars, present = group_lengths(metadata, lengths, group, len(canonical_json(body)))
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
                                np.ceil(len(canonical_json(api_preview(request))) / 4)
                            ),
                        }
                    )
    return estimates, pilot


def main() -> None:
    cfg = load_config("feature_creation/default")
    records, pilot = [], []
    for entry in load_catalog(cfg["catalog"])["datasets"]:
        dataset = load_dataset(entry, allow_download=False)
        part, sample = review_dataset(dataset, cfg)
        records.extend(part)
        pilot.extend(sample)
        print(f"Reviewed {entry['name']}", flush=True)
    estimates = pd.DataFrame(records)
    pilot = pd.DataFrame(pilot)
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
    folder = paths.results_dir("review", phase="feature_creation")
    folder.mkdir(parents=True, exist_ok=True)
    estimates.to_csv(folder / "dataset_group_budget.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(folder / "sensitivity.csv", index=False)
    pilot.to_csv(folder / "pilot_preview_budget.csv", index=False)
    unique_pilot = pilot.drop_duplicates("request_key")
    write_json(
        folder / "manifest.json",
        {
            "status": "offline_review_only",
            "code_sha256": code_identity(),
            "config": cfg,
            "dataset_hashes": {
                e["id"]: e["sha256"] for e in load_catalog(cfg["catalog"])["datasets"]
            },
            "caveats": [
                "Character approximation, not provider billing tokenizer.",
                "Empty inputs excluded; inputs over the context limit still included as a budget scenario.",
                "Identical canonical field values deduplicated within groups; joint reuses per-column whenever only one field is available.",
                "No change to prompt wording or long-text policy; no requests sent.",
            ],
            "totals": totals.to_dict("records"),
            "pilot": {
                "positions": "first, middle, last CSV rows per dataset; illustrative only",
                "logical_requests": len(pilot),
                "unique_requests": len(unique_pilot),
                "wire_tokens_approx": int(unique_pilot.wire_tokens_approx.sum()),
                "cost_usd": None
                if price is None
                else float(unique_pilot.wire_tokens_approx.sum() * price / 1e6),
            },
        },
    )
    print(totals.to_string(index=False))
    print(f"Saved review to {folder}")


if __name__ == "__main__":
    main()
