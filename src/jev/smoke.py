"""Three sequential, synthetic Jev requests; separate from the dataset extraction pilot.

No inference without --execute. Model/price/context bounds come from the official
30 October model reference. These slow probes do not establish sustained account throughput.
The maximum three attempted requests reserve $0.008064 at the documented 64k
per-request context ceiling and $0.042/M input. No SDK or application retries.
"""

import argparse
import getpass
import json
import os
import re
import time
from datetime import UTC, datetime

from src.cluster.semantic_contracts import probes
from src.jev.auth import api_key, credentials_available, error_hint
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


def preserve_smoke_state():
    """Migrate legacy paid-call evidence before output cleanup; never dispatch a call.

    The durable control record lives outside output, beside permanent feature caches.
    A response without its ledger is refused rather than silently treated as reusable.
    """
    directory = paths.jev_cache_path("jev_cache/_control/jev_smoke")
    state = directory / "state.json"
    legacy = paths.manifests_dir("experiment_0") / "before_extraction" / "jev_smoke.json"
    with file_lock(directory / "state.lock", timeout=1):
        if legacy.exists():
            latest = json.loads(legacy.read_text(encoding="utf-8"))
            if latest.get("round_id") is not None:
                round_id = latest["round_id"]
                if not isinstance(round_id, str) or not re.fullmatch(r"[0-9a-f]{64}", round_id):
                    raise ValueError("Latest Jev diagnostic has an invalid round identity")
                saved_round = directory / "rounds" / round_id / "state.json"
                if not saved_round.exists() or digest(latest) != digest(json.loads(saved_round.read_text(encoding="utf-8"))):
                    raise ValueError("Latest Jev diagnostic differs from its permanent round ledger")
                # The canonical output describes a new round. Its paid ledger is
                # already durable; it must never replace the original ledger.
                return directory, state
            if latest.get("model", "jev-1.13.0") != "jev-1.13.0":
                model = latest.get("model")
                if model != "jev-1.14.0":
                    raise ValueError("Latest Jev diagnostic has an unreviewed model identity")
                model_state = directory / "models" / model / "state.json"
                if not model_state.exists() or digest(latest) != digest(json.loads(model_state.read_text(encoding="utf-8"))):
                    raise ValueError("Latest Jev diagnostic differs from its permanent model ledger")
                return directory, state  # Preserve the original 1.13 ledger in place.
        if state.exists():
            saved = json.loads(state.read_text(encoding="utf-8"))
            if legacy.exists() and digest(saved) != digest(
                json.loads(legacy.read_text(encoding="utf-8"))
            ):
                raise ValueError(
                    "Permanent and output Jev ledgers differ; review before cleanup or inference"
                )
            return directory, state
        if not legacy.exists():
            return directory, state
        report = json.loads(legacy.read_text(encoding="utf-8"))
        if not report.get("request_identity") or not isinstance(report.get("attempts"), list):
            raise ValueError("Legacy Jev smoke ledger is invalid; output cleanup refused")
        responses = []
        for attempt in report["attempts"]:
            name = paths.output_component(attempt["probe"]) + ".json"
            old = paths.results_dir("jev_smoke", phase="experiment_0") / name
            if old.exists():
                raw = json.loads(old.read_text(encoding="utf-8"))
                if digest(raw) != attempt.get("response_sha256"):
                    raise ValueError("Legacy Jev smoke response checksum differs")
                responses.append((directory / name, raw))
        for target, raw in responses:
            write_json(target, raw)
        write_json(state, report)  # Missing responses retain their unresolved attempt reservation.
    return directory, state


def requests(*, robustness=False, real=False):
    settings = dict(
        provider="jev",
        model=MODEL,
        version=POLICY["version"],
        api_mapping_version=MAPPING_VERSION,
        request_configuration={},
    )
    if real:
        from src.cluster.semantic_contracts import real_robustness_probes

        return list(real_robustness_probes(settings))
    if robustness:
        from src.cluster.semantic_contracts import robustness_probes

        return list(robustness_probes(settings))
    selected = {"binary_True", "choice_red_green_blue", "score_4"}
    return [
        (name, request, expected)
        for name, request, expected in probes(settings)
        if name in selected
    ]


def run(
    *,
    execute=False,
    client_factory=None,
    sleep=time.sleep,
    retry_auth_rejected=False,
    robustness=False,
    revalidate_only=False,
    dataset_robustness=False,
    config="feature_creation/jev.local",
    round_id=None,
):
    if not (execute or revalidate_only) or paths.on_vsc():
        raise RuntimeError("The bounded Jev smoke test requires --execute locally")
    if round_id is not None and not re.fullmatch(r"[0-9a-f]{64}", round_id):
        raise ValueError("Diagnostic round must be a frozen source SHA-256")
    if MODEL != "jev-1.14.0" or TOKEN_BOUND != 64000 or MAX_CALLS != 3 or not 0 < PRICE <= 0.042:
        raise ValueError("Smoke policy differs from the reviewed model/context/three-call budget")
    key = None if revalidate_only else api_key()
    from src.cluster.freeze import check_if_present

    check_if_present()
    diagnostic = robustness or dataset_robustness
    test_name = (
        "jev_dataset_robustness"
        if dataset_robustness
        else "jev_robustness"
        if robustness
        else "jev_smoke"
    )
    if diagnostic or round_id is not None or MODEL != "jev-1.13.0":
        directory = paths.jev_cache_path("jev_cache/_control/" + test_name)
        if round_id is not None:
            directory = directory / "rounds" / round_id
        else:
            directory = directory / "models" / MODEL
        directory.mkdir(parents=True, exist_ok=True)
        state_path = directory / "state.json"
    else:
        directory, state_path = preserve_smoke_state()
    output_directory = paths.results_dir(test_name, phase="experiment_0")
    report_path = paths.manifests_dir("experiment_0") / "before_extraction" / f"{test_name}.json"
    planned = (
        requests(real=True)
        if dataset_robustness
        else requests(robustness=True)
        if robustness
        else requests()
    )
    maximum = 13 if dataset_robustness else 15 if robustness else MAX_CALLS
    if dataset_robustness and not revalidate_only:
        from src.jev.hosted import ReviewedContextEnvelope

        envelope = ReviewedContextEnvelope(load_yaml(config))
        if envelope.contract["max_billed_input_tokens"] > TOKEN_BOUND:
            raise ValueError("Diagnostic context exceeds its fixed billing reservation")
        for _, request, _ in planned:
            envelope(request)
    identity = digest([request.cache_key for _, request, _ in planned])
    report = {
        "status": "running",
        "model": MODEL,
        "request_identity": identity,
        "round_id": round_id,
        "scope": "label-blind real-text sensitivity sample; no feature materialization"
        if dataset_robustness
        else "synthetic prompt robustness; no dataset extraction"
        if robustness
        else "three synthetic task contracts; no dataset extraction",
        "account_limits_verified": False,
        "full_extraction_enabled": False,
        "documented_input_token_bound_per_attempt": TOKEN_BOUND,
        "input_price_per_million_usd": PRICE,
        "reference": REFERENCE,
        "maximum_new_call_cost_usd": maximum * TOKEN_BOUND * PRICE / 1e6,
        "client_minimum_interval_seconds": POLICY["minimum_interval_seconds"],
        "automatic_retries": 0,
        "created_utc": datetime.now(UTC).isoformat(),
        "attempts": [],
    }
    client = None

    def persist():
        write_json(state_path, report)
        write_json(report_path, report)

    with file_lock(state_path.with_suffix(".lock"), timeout=1):
        if state_path.exists():
            report = json.loads(state_path.read_text(encoding="utf-8"))
            if report.get("request_identity") != identity:
                raise ValueError("Stored smoke test has different requests; preserve and review")
        if report.get("status") == "stopped_requires_review" and report["attempts"]:
            report["attempts"][-1].setdefault(
                "failure",
                {"error_type": report.get("error_type"), "http_status": report.get("http_status")},
            )
        if retry_auth_rejected and report.get("status") == "stopped_requires_review":
            pending = [
                a for a in report["attempts"] if not (directory / f"{a['probe']}.json").exists()
            ]
            failure = pending[0].get("failure", {}) if len(pending) == 1 else {}
            if (
                failure.get("http_status") != 401
                or failure.get("error_type") != "TypeSafeAuthenticationError"
                or len(pending) != 1
                or pending[0] != report["attempts"][-1]
                or pending[0].get("status") != "attempting"
                or report.get("authentication_rejections")
            ):
                raise ValueError("Only one explicitly recorded HTTP 401 can be retried this way")
            failed = report["attempts"].pop()
            report["authentication_rejections"] = [
                {
                    **failed,
                    "http_status": 401,
                    "review": "explicit retry after credential correction",
                }
            ]
            # Preserve the old conservative reservation; do not assert a billing refund.
            report["historical_reserved_usd"] = failed["reserved_usd"]
        report["status"] = "running"
        report.pop("error_type", None)
        report.pop("http_status", None)
        persist()
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
                    if revalidate_only:
                        raise RuntimeError(
                            "Saved response missing; offline revalidation cannot call Jev"
                        )
                    if len(report["attempts"]) >= maximum:
                        raise RuntimeError("Synthetic-test attempt ceiling reached")
                    if client is None:
                        if client_factory is None:
                            from typesafe_sdk import RetryPolicy, TypeSafeClient

                            client = TypeSafeClient(
                                api_key=key,
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
                        "provenance": json.loads(request.provenance_json),
                    }
                    report["attempts"].append(prior)
                    persist()  # Persist before a possibly billed request.
                    wire = api_preview(request)
                    raw = client.system_one(
                        wire["state"], wire["questions"]
                    ).raw_http_response.json()
                    write_json(response_file, raw)
                    prior["response_sha256"] = digest(raw)
                    persist()
                write_json(output_directory / f"{name}.json", raw)
                usage = raw.get("usage", {}).get("input_tokens")
                if type(usage) is not int or not 0 <= usage <= TOKEN_BOUND:
                    raise ValueError("Provider usage differs from the documented billing envelope")
                features = response_features(request, raw)
                prior["canonical_probability_order"] = json.loads(request.provenance_json).get(
                    "canonical_probability_order"
                )
                answer = raw["answers"]["semantic"]
                if answer["type"] in {"choice", "score"}:
                    from src.jev.provider import probability_distribution

                    _, prior["probability_validation"] = probability_distribution(
                        list(answer["probabilities"].values()), "jev"
                    )
                if answer["type"] == "score":
                    from src.jev.provider import score_consistency

                    prior["score_validation"] = score_consistency(
                        answer["score"], [answer["probabilities"][str(i)] for i in range(9)], "jev"
                    )
                if dataset_robustness:
                    passed = True  # No correct target is known or used in this diagnostic.
                elif name.startswith("binary"):
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
                    status="observed"
                    if dataset_robustness
                    else "passed"
                    if passed
                    else "semantic_review_required",
                    features=features,
                    input_tokens=usage,
                    charged_usd=usage * PRICE / 1e6,
                    reserved_usd=0,
                    response_path=str(response_file),
                )
                prior.pop("failure", None)
                persist()
                if not passed and not diagnostic:
                    raise ValueError("Known-answer probe needs response/wording review")
            report.update(
                status="jev_smoke_passed_requires_dataset_pilot",
                pinned_version_access_verified=True,
                total_cost_usd=sum(a["charged_usd"] for a in report["attempts"]),
                conservative_total_with_prior_reservations_usd=(
                    sum(a["charged_usd"] for a in report["attempts"])
                    + report.get("historical_reserved_usd", 0)
                ),
            )
            if diagnostic:
                from src.cluster.semantic_contracts import robustness_summary

                report["prompt_robustness"] = robustness_summary(report["attempts"])
                report["status"] = test_name + "_requires_review"
        except Exception as exc:
            if report["attempts"] and report["attempts"][-1].get("status") == "attempting":
                report["attempts"][-1].setdefault(
                    "failure",
                    {"error_type": type(exc).__name__, "http_status": getattr(exc, "status", None)},
                )
            report.update(
                status="stopped_requires_review",
                error_type=type(exc).__name__,
                http_status=getattr(exc, "status", None),
            )
            raise
        finally:
            persist()
            if client is not None:
                client.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--revalidate-only",
        action="store_true",
        help="Reprocess saved responses and accounting without credentials or network calls",
    )
    parser.add_argument(
        "--robustness",
        action="store_true",
        help="Separate resumable 15-call synthetic diagnostic, at most $0.04032 plus any explicit auth retry reservation",
    )
    parser.add_argument(
        "--dataset-robustness",
        action="store_true",
        help="Separate 13-call real-text diagnostic; requires the reviewed local context envelope",
    )
    parser.add_argument("--config", default="feature_creation/jev.local")
    parser.add_argument("--round-id", help="Explicit new frozen-source diagnostic round; retries reuse that round's saved responses")
    parser.add_argument(
        "--retry-auth-rejected",
        action="store_true",
        help="After correcting credentials, explicitly retry one saved 401; retain its reservation/history",
    )
    args = parser.parse_args()
    if args.robustness and args.dataset_robustness:
        parser.error("Select synthetic or real-text robustness separately")
    maximum = 13 if args.dataset_robustness else 15 if args.robustness else MAX_CALLS
    if not args.execute and not args.revalidate_only:
        print(
            json.dumps(
                {
                    "execution_enabled": False,
                    "model": MODEL,
                    "maximum_requests": maximum,
                    "maximum_cost_usd": maximum * TOKEN_BOUND * PRICE / 1e6,
                },
                indent=2,
            )
        )
        return
    prompted = not args.revalidate_only and not credentials_available()
    try:
        if prompted:
            os.environ["TYPESAFE_API_KEY"] = getpass.getpass("TypeSafe API key (hidden): ")
        print(
            json.dumps(
                run(
                    execute=True,
                    retry_auth_rejected=args.retry_auth_rejected,
                    robustness=args.robustness,
                    revalidate_only=args.revalidate_only,
                    dataset_robustness=args.dataset_robustness,
                    config=args.config,
                    round_id=args.round_id,
                ),
                indent=2,
            )
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "stopped_requires_review",
                    "error_type": type(exc).__name__,
                    "http_status": getattr(exc, "status", None),
                    "hint": error_hint(exc)
                    or "Inspect the experiment_0 before_extraction Jev report and permanent _control ledger; saved responses can be revalidated without new calls.",
                }
            )
        )
        raise SystemExit(1) from None
    finally:
        if prompted:
            os.environ.pop("TYPESAFE_API_KEY", None)


if __name__ == "__main__":
    main()
