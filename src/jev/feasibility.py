"""Offline cost/context screens, never provider-tokenizer guarantees or live access tests."""

import pandas as pd


def comparison_feasibility(estimates, catalog, cfg=None):
    """Use the exact deduplicated input workload, approximated at 3/4/5 chars per token.

    Provider tokenizers/serialization differ. The 3-character sensitivity scenario
    uses the Jev JSON body as a proxy; it is not an upper bound on any tokenizer.
    """
    central = estimates[estimates.characters_per_token.eq(4)]
    conservative = estimates[estimates.characters_per_token.eq(3)]
    records = []
    for name, candidate in catalog["candidates"].items():
        price = candidate.get("input_price_per_million_usd")
        if name == "jev" and cfg is not None:
            price = cfg["cost"]["input_price_per_million"]
        ceiling = candidate["context_tokens"]
        above = (
            None
            if ceiling is None
            else int(conservative[f"combined_requests_above_{ceiling}_approx"].sum())
        )
        records.append(
            {
                "extractor": name,
                "enabled": False,
                "unique_requests": int(central.requests_for_combined_build.sum()),
                "input_tokens_central_proxy": int(central.wire_tokens_combined_approx.sum()),
                "input_price_usd_per_million": price,
                "hosted_cost_central_proxy_usd": None
                if price is None
                else float(central.wire_tokens_combined_approx.sum() * price / 1e6),
                "context_tokens_reference": ceiling,
                "unique_requests_above_context_screen": above,
                "above_context_screen_pct": None
                if above is None
                else 100 * above / max(1, central.requests_for_combined_build.sum()),
                "readiness": "disabled; account/checkpoint, context and live schema pilot required",
                "cost_caveat": "self-hosted compute/storage unpriced"
                if price is None
                else "proxy only; provider tokenizer, billing overhead/retries differ",
            }
        )
    return pd.DataFrame(records)
