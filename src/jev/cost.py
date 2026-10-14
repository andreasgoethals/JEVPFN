"""Unpriced or scenario-priced workload estimates from the full dataset audit.

Empty text groups are skipped; missing text fields are omitted from joint requests. Deduplication, provider
wrappers, output tokens, retries, billing minimums and rate limits are not assumed.
The character approximation is applied to the exact intended JSON, not API wire syntax.
"""

from __future__ import annotations

import numpy as np

from src.jev.requests import make_payload, text_groups
from src.utils.serialization import canonical_json

MODES = ("per_column", "joint", "per_column_and_joint")


def estimate_dataset(metadata, lengths: dict, cfg: dict) -> list[dict]:
    ratio = cfg["token_estimation"]["characters_per_token"]
    price = cfg["cost"]["input_price_per_million"]
    provider_overhead = cfg["cost"]["provider_overhead_tokens_per_request"]
    limit = cfg.get("provider_reference", {}).get("state_plus_longest_question_tokens")
    estimates = []
    for include_structured in (False,):
        base = len(
            canonical_json(
                make_payload(
                    metadata,
                    {},
                    {} if include_structured else None,
                    scores=cfg["jev"]["regression_scores"],
                )
            )
        )
        for mode in MODES:
            total, total_chars, peak, count, above_context = 0, 0, 0, 0, 0
            for group in text_groups(metadata, mode):
                chars, present = group_lengths(metadata, lengths, group, base)
                chars = chars[present > 0]
                tokens = np.ceil(chars / ratio).astype(np.int64)
                total += int(tokens.sum())
                total_chars += int(chars.sum())
                peak = max(peak, int(tokens.max(initial=0)))
                count += len(chars)
                if limit is not None:
                    above_context += int((tokens > limit).sum())
            billed_scenario = total + count * provider_overhead
            estimates.append(
                {
                    "dataset_id": metadata.dataset_id,
                    "dataset": metadata.dataset_name,
                    "mode": mode,
                    "include_non_text_features": include_structured,
                    "requests": count,
                    "intended_json_characters": total_chars,
                    "input_tokens_approx": total,
                    "max_request_tokens_approx": peak,
                    "requests_above_context_screen_approx": above_context
                    if limit is not None
                    else None,
                    "hypothetical_provider_overhead_tokens": count * provider_overhead,
                    "input_cost_estimate": None if price is None else billed_scenario * price / 1e6,
                    "cost_per_unit_price": billed_scenario / 1e6,
                    "deduplication": "not deducted; empty inputs excluded",
                }
            )
    return estimates


def group_lengths(metadata, lengths, group, base):
    """Exact encoded input size, omitting empty fields without copying long strings."""
    n = len(lengths["text_0"])
    chars = np.full(n, base, dtype=np.int64)
    present = np.zeros(n, dtype=np.int64)
    for col in group:
        index = metadata.text_columns.index(col)
        mask = lengths[f"present_{index}"]
        chars += np.where(mask, lengths[f"text_{index}"], 0)
        present += mask
    chars += np.maximum(present - 1, 0)
    return chars, present
