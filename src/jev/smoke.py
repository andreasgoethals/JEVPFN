"""Three sequential, synthetic Jev requests; separate from the dataset extraction pilot.

No inference without --execute. Model/price/context bounds come from the official
16 October model reference. Account throughput remains unverified; HTTP errors stop.
The maximum three attempted requests reserve $0.008064 at the documented 64k
per-request context ceiling and $0.042/M input. No SDK or application retries.
"""

import argparse
import getpass
import json
import os
import time
from datetime import UTC, datetime

from src.cluster.semantic_contracts import probes
from src.jev.provider import MAPPING_VERSION, api_preview, response_features
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.locking import file_lock
from src.utils.serialization import digest, write_json

POLICY = load_yaml("experiment_0/jev_smoke")
MODEL = POLICY["model"]
PRICE = POLICY["input_price_per_million_usd"]
TOKEN_BOUND = POLICY["documented_request_context_tokens"]
MAX_CALLS = POLICY["maximum_requests"]
REFERENCE = POLICY["reference"]


def requests():
    settings = dict(
        provider="jev",
        model=MODEL,
        version="1.13.0",
        api_mapping_version=MAPPING_VERSION,
        request_configuration={},
    )
    selected = {"binary_True", "choice_red_green_blue", "score_4"}
    return [
        (name, request, expected)
        for name, request, expected in probes(settings)
        if name in selected
    ]


def run(*, execute=False, client_factory=None, sleep=time.sleep):
    if not execute or paths.on_vsc():
        raise RuntimeError("The bounded Jev smoke test requires --execute locally")
    if MODEL != "jev-1.13.0" or TOKEN_BOUND != 64000 or MAX_CALLS != 3 or not 0 < PRICE <= 0.042:
        raise ValueError("Smoke policy differs from the reviewed model/context/three-call budget")
    if not os.environ.get("TYPESAFE_API_KEY"):
        raise ValueError("Supply the key through the hidden prompt or process environment")
    from src.cluster.freeze import check_if_present

    check_if_present()
    directory = paths.results_dir("jev_smoke", phase="experiment_0")
    report_path = paths.manifests_dir("experiment_0") / "before_extraction" / "jev_smoke.json"
    planned = requests()
    identity = digest([request.cache_key for _, request, _ in planned])
    report = {
        "status": "running",
        "model": MODEL,
        "request_identity": identity,
        "scope": "three synthetic task contracts; no dataset extraction",
        "account_limits_verified": False,
        "full_extraction_enabled": False,
        "documented_input_token_bound_per_attempt": TOKEN_BOUND,
        "input_price_per_million_usd": PRICE,
        "reference": REFERENCE,
        "maximum_new_call_cost_usd": MAX_CALLS * TOKEN_BOUND * PRICE / 1e6,
        "client_minimum_interval_seconds": POLICY["minimum_interval_seconds"],
        "automatic_retries": 0,
        "created_utc": datetime.now(UTC).isoformat(),
        "attempts": [],
    }
    client = None
    with file_lock(report_path.with_suffix(".lock"), timeout=1):
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("request_identity") != identity:
                raise ValueError("Stored smoke test has different requests; preserve and review")
        try:
            for name, request, expected in planned:
                prior = next((a for a in report["attempts"] if a["probe"] == name), None)
                response_file = directory / f"{name}.json"
                if prior and not response_file.exists():
                    raise RuntimeError(
                        "An earlier attempt has no saved response; review it before another call"
                    )
                if response_file.exists():
                    if prior is None:
                        raise ValueError("Response has no matching attempt record")
                    raw = json.loads(response_file.read_text(encoding="utf-8"))
                    if prior.get("response_sha256") != digest(raw):
                        raise ValueError("Stored smoke response checksum differs")
                else:
                    if len(report["attempts"]) >= MAX_CALLS:
                        raise RuntimeError("Three-call attempt ceiling reached")
                    if client is None:
                        if client_factory is None:
                            from typesafe_sdk import RetryPolicy, TypeSafeClient

                            client = TypeSafeClient(
                                api_key=os.environ["TYPESAFE_API_KEY"],
                                base_url="https://api.typesafe.ai",
                                model=MODEL,
                                retry=RetryPolicy(max_retries=0),
                                timeout=30,
                            )
                        else:
                            client = client_factory()
                    if report["attempts"]:
                        sleep(POLICY["minimum_interval_seconds"])
                    prior = {
                        "probe": name,
                        "request_key": request.cache_key,
                        "status": "attempting",
                        "reserved_usd": TOKEN_BOUND * PRICE / 1e6,
                    }
                    report["attempts"].append(prior)
                    write_json(report_path, report)  # Persist before a possibly billed request.
                    wire = api_preview(request)
                    raw = client.system_one(
                        wire["state"], wire["questions"]
                    ).raw_http_response.json()
                    write_json(response_file, raw)
                    prior["response_sha256"] = digest(raw)
                    write_json(report_path, report)
                usage = raw.get("usage", {}).get("input_tokens")
                if type(usage) is not int or not 0 <= usage <= TOKEN_BOUND:
                    raise ValueError("Provider usage differs from the documented billing envelope")
                features = response_features(request, raw)
                if name.startswith("binary"):
                    passed = features["p_positive"] > 0.5
                else:
                    probabilities = {k: v for k, v in features.items() if k.startswith("p_")}
                    best = [
                        k
                        for k, value in probabilities.items()
                        if value == max(probabilities.values())
                    ]
                    passed = best == [
                        f"p_c{expected:03d}" if name.startswith("choice") else f"p_{expected + 4}"
                    ]
                prior.update(
                    status="passed" if passed else "semantic_review_required",
                    features=features,
                    input_tokens=usage,
                    charged_usd=usage * PRICE / 1e6,
                    reserved_usd=0,
                    response_path=str(response_file),
                )
                write_json(report_path, report)
                if not passed:
                    raise ValueError("Known-answer probe needs response/wording review")
            report.update(
                status="jev_smoke_passed_requires_dataset_pilot",
                pinned_version_access_verified=True,
                total_cost_usd=sum(a["charged_usd"] for a in report["attempts"]),
            )
        except Exception as exc:
            report.update(
                status="stopped_requires_review",
                error_type=type(exc).__name__,
                http_status=getattr(exc, "status", None),
            )
            raise
        finally:
            write_json(report_path, report)
            if client is not None:
                client.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "execution_enabled": False,
                    "model": MODEL,
                    "maximum_requests": MAX_CALLS,
                    "maximum_cost_usd": MAX_CALLS * TOKEN_BOUND * PRICE / 1e6,
                },
                indent=2,
            )
        )
        return
    prompted = not os.environ.get("TYPESAFE_API_KEY")
    try:
        if prompted:
            os.environ["TYPESAFE_API_KEY"] = getpass.getpass("TypeSafe API key (hidden): ")
        print(json.dumps(run(execute=True), indent=2))
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "stopped_requires_review",
                    "error_type": type(exc).__name__,
                    "http_status": getattr(exc, "status", None),
                }
            )
        )
        raise SystemExit(1) from None
    finally:
        if prompted:
            os.environ.pop("TYPESAFE_API_KEY", None)


if __name__ == "__main__":
    main()
