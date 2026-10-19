"""Check authentication without inference, or prepare a private, explicitly reviewed pilot.

Loads the process environment or the owner's ignored local credential file, with a
hidden prompt only if neither exists. Never writes keys. No automatic top-up,
full extraction, or inference is performed by this module.
"""

import argparse
import getpass
import json
import os
from decimal import ROUND_DOWN, Decimal

import yaml

from src.data.loaders import load_catalog
from src.data.selection import selected_entries
from src.jev.auth import api_key, credentials_available, error_hint
from src.jev.extraction import campaign_limits
from src.jev.hosted import ReviewedContextEnvelope
from src.utils import paths
from src.utils.config import _merge, load_yaml
from src.utils.serialization import write_json


def account_check(client_factory=None):
    if paths.on_vsc():
        raise RuntimeError("The hosted account check runs locally")
    key = api_key()
    from typesafe_sdk import RetryPolicy, TypeSafeClient

    factory = client_factory or TypeSafeClient
    with factory(
        api_key=key,
        base_url="https://api.typesafe.ai",
        model="jev-1.13.0",
        retry=RetryPolicy(max_retries=0),
        timeout=30,
    ) as client:
        result = client.models.list()
    report = {
        "authenticated": True,
        "inference_performed": False,
        "models": [m.name for m in result.models],
        "rates_verified": False,
        "balance_verified": False,
        "pinned_version_access_verified": False,
    }
    write_json(paths.manifests_dir("experiment_0") / "jev_account.json", report)
    return report


def pilot_overlay(
    base,
    *,
    usd_per_eur,
    exchange_evidence,
    rps,
    tps,
    max_bytes,
    max_tokens,
    context_evidence,
    account_evidence,
):
    """Only verified user-supplied bounds; no inferred tokenizer, rates or currency parity."""
    if (
        not exchange_evidence.strip()
        or not context_evidence.strip()
        or not account_evidence.strip()
    ):
        raise ValueError("Rate, currency and context/billing evidence are required")
    ceiling = (
        Decimal(str(base["live_extraction"]["owner_budget_eur"])) * Decimal(str(usd_per_eur))
    ).quantize(Decimal(".01"), rounding=ROUND_DOWN)
    datasets = selected_entries(load_catalog(), base)
    each = (ceiling / len(datasets)).quantize(Decimal(".000001"), rounding=ROUND_DOWN)
    live = {
        "enabled": True,
        "pilot_authorized": True,
        "pilot_accepted": False,
        "full_feature_creation_enabled": False,
        "accepted_pilot_manifest": None,
        "campaign_spending_limit_usd": float(ceiling),
        "dataset_budget_allocations_usd": {e["id"]: float(each) for e in datasets},
        "usd_per_eur": usd_per_eur,
        "exchange_rate_evidence": exchange_evidence,
        "requests_per_second": rps,
        "input_tokens_per_second": tps,
        "workers": 1,
        "account_limits_verified": True,
        "account_limits_evidence": account_evidence,
        "billing_bound_verified": True,
        "context_policy_reviewed": True,
        "price_verified": True,
        "max_billed_input_tokens_per_attempt": max_tokens,
        "context_envelope": {
            "verified": True,
            "model": base["jev"]["model"],
            "max_request_utf8_bytes": max_bytes,
            "max_billed_input_tokens": max_tokens,
            "evidence": context_evidence,
        },
    }
    overlay = {"extends": "feature_creation", "live_extraction": live}
    resolved = _merge(base, overlay)
    campaign_limits(resolved)
    ReviewedContextEnvelope(resolved)
    return overlay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check-account", action="store_true")
    action.add_argument("--configure", action="store_true")
    args = parser.parse_args()
    if args.check_account:
        prompted = not credentials_available()
        try:
            if prompted:
                os.environ["TYPESAFE_API_KEY"] = getpass.getpass("TypeSafe API key (hidden): ")
            print(json.dumps(account_check(), indent=2))
        except Exception as exc:
            # SDK exceptions may contain request bodies; expose only safe diagnostics.
            print(
                json.dumps(
                    {
                        "authenticated": False,
                        "error_type": type(exc).__name__,
                        "http_status": getattr(exc, "status", None),
                        "hint": error_hint(exc),
                    }
                )
            )
            raise SystemExit(1) from None
        finally:
            if prompted:
                os.environ.pop("TYPESAFE_API_KEY", None)
        return
    target = paths.config_path("feature_creation/jev.local")
    if target.exists():
        raise ValueError(
            "A pilot config already exists; review it rather than silently replacing budgets"
        )
    print("Enter verified values only. Do not guess. Ctrl+C stops without writing anything.")
    base = load_yaml("feature_creation")
    price = float(input("Verified USD price per million input tokens: "))
    if price != base["cost"]["input_price_per_million"]:
        raise ValueError("Published price differs; review the pinned price before continuing")
    overlay = pilot_overlay(
        base,
        usd_per_eur=float(input("Conservative USD per EUR, allowing payment fees: ")),
        exchange_evidence=input("Exchange-rate source/date and fee allowance: "),
        rps=float(input("Verified organization requests per second: ")),
        tps=float(input("Verified organization input tokens per second: ")),
        account_evidence=input("Account limits source/date (no credentials): "),
        max_bytes=int(input("Provider-confirmed safe complete-request UTF-8 byte ceiling: ")),
        max_tokens=int(input("Provider-confirmed maximum billed input tokens per attempt: ")),
        context_evidence=input("Context/billing-bound source or confirmation/date: "),
    )
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(overlay, handle, sort_keys=False)
    print(f"Saved {target}; no inference performed. Full extraction remains disabled.")


if __name__ == "__main__":
    main()
