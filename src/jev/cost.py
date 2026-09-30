"""Unpriced or scenario-priced workload estimates from the full dataset audit.

Empty text groups are skipped; missing text fields are omitted from joint requests. Deduplication, provider
wrappers, output tokens, retries, billing minimums and rate limits are not assumed.
The character approximation is applied to the exact intended JSON, not API wire syntax.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.audit import AuditCollection, audit_lock
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


def estimate_workload(audit: AuditCollection, cfg: dict) -> pd.DataFrame:
    records = []
    with audit_lock(audit.folder):
        for dataset_id in audit.summary.dataset_id:
            with np.load(audit.folder / f"{dataset_id}_lengths.npz", allow_pickle=False) as lengths:
                records.extend(estimate_dataset(audit.metadata(dataset_id), lengths, cfg))
    result = pd.DataFrame(records)
    with audit_lock(audit.folder):
        result.to_csv(audit.folder / "jev_workload.csv", index=False)
    return result


def workload_totals(estimates: pd.DataFrame) -> pd.DataFrame:
    return (
        estimates.groupby(["mode", "include_non_text_features"], sort=False)[
            [
                "requests",
                "input_tokens_approx",
                "hypothetical_provider_overhead_tokens",
                "cost_per_unit_price",
                "input_cost_estimate",
            ]
        ]
        .sum(min_count=1)
        .reset_index()
    )


def budget_scenarios(estimates: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One price estimate per input mode; combined already includes per-column and joint."""
    result = workload_totals(estimates)
    rate = cfg.get("provider_reference", {}).get("requests_per_minute_reference")
    if rate:
        result["ideal_hours_at_reference_rpm"] = result.requests / rate / 60
    return result


def full_feature_budget(totals: pd.DataFrame) -> dict:
    """Count the text-only combined mode once; never double-count primitive modes."""
    selected = totals.loc[totals["mode"] == "per_column_and_joint"]
    if len(selected) != 1 or set(selected.include_non_text_features) != {False}:
        raise ValueError("Expected the text-only combined mode.")
    return {
        "logical_requests": int(selected.requests.sum()),
        "input_tokens_approx": int(selected.input_tokens_approx.sum()),
        "input_cost_estimate": None
        if selected.input_cost_estimate.isna().any()
        else float(selected.input_cost_estimate.sum()),
        "deduplication": "not deducted",
        "context_overflow_policy": "unresolved; not an executable budget",
    }


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
