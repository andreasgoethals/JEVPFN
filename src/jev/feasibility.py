"""Offline cost/context screens, never provider-tokenizer guarantees or live access tests."""

import pandas as pd


def comparison_feasibility(estimates, catalog):
    """Use the exact deduplicated input workload, approximated at 3/4/5 chars per token.

    Provider tokenizers/serialization differ. Context screens use conservative 3-char
    estimates and include the current Jev JSON body as a common proxy. Microsoft's
    ceiling is the OpenRouter listing; deployed Foundry accounting is unverified.
    """
    central = estimates[estimates.characters_per_token.eq(4)]
    conservative = estimates[estimates.characters_per_token.eq(3)]
    limits = {
        "jev": 32000,
        "liquid": 32768,
        "perplexity": 8192,
        "cloudflare": 24576,
        "strands": 4096,
        "microsoft": catalog["candidates"]["microsoft"].get("context_tokens_reference"),
    }
    records = []
    for name, candidate in catalog["candidates"].items():
        price = candidate.get("input_price_per_million_usd")
        if name == "perplexity":
            price = candidate["announced_hosted_input_price_per_million_usd"]
        ceiling = limits[name]
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
