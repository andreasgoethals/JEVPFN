"""Guarded hosted client for Liquid's d1.2-ultra decision model. Disabled; nothing runs on import.

Preview (no network, no credential access): python -m src.jev.liquid_hosted

Sources checked 2026-11-19: https://docs.liquid.ai/lfm/models/d1,
https://docs.liquid.ai/api/reference/decision-models/make-decisions.md,
https://docs.liquid.ai/api/reference/rate-limits.md and
https://docs.liquid.ai/api/reference/error-codes.md. No live call has validated this
mapping. The semantic request is the shared Noul/Choice/Score question used by every
decision model (`provider.semantic_question`), with the same label-free state. Responses
are decoded by `provider.response_features`, with strict sums because Liquid documents no
rounding. Dispatch reuses `DatasetExtractor`: persistent reservations, the spending ledger,
ambiguous-usage stops and raw-response storage before validation. The response store is
separate from Jev's, so neither provider's spending counts against the other's allocation.

Every gate in `readiness` must pass, and `--execute` is required. Full extraction also
requires an owner-signed pilot acceptance bound to these settings and input sets.
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.error
import urllib.request
from datetime import UTC, date, datetime
from uuid import uuid4

from src.jev.auth import credential_present, read_api_key
from src.jev.extraction import (
    AccountCreditError,
    RateGate,
    RejectedRetryable,
    ReleaseNotDispatched,
    campaign_limits,
)
from src.jev.provider import LIQUID_ULTRA, QUESTION_ID, semantic_question
from src.jev.requests import IntendedRequest, require_nonempty_text
from src.jev.templates import TEMPLATE_VERSION
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.serialization import canonical_json, digest, write_json

PROVIDER = LIQUID_ULTRA
MAPPING_VERSION = "liquid-decisions-systemone-v2"
LEGACY_MAPPING_VERSION = "liquid-decisions-systemone-v1"
DOCUMENTED_MODELS = frozenset({"d1.2-ultra"})
CREDENTIAL_ENV = "LIQUID_API_KEY"
LOCAL_CREDENTIAL_PATH = paths.REPO_ROOT / "secrets/liquid.local.json"
PILOT_AUTHORIZATION_STAGE = "liquid_ultra_pilot_authorization"
PILOT_ACCEPTANCE_STAGE = "liquid_ultra_pilot"
READINESS_FLAGS = (
    "pilot_authorized",
    "account_limits_verified",
    "billing_bound_verified",
    "context_policy_reviewed",
    "price_verified",
)
# 429 (organisation limit) and 503 (shared capacity) are documented rejections with
# Retry-After. Liquid does not document whether they are billed, so they stay ambiguous
# stops until a pilot verifies `overload_rejections_unbilled_verified`.
OVERLOAD_STATUSES = frozenset({429, 503})
CREDIT_CODES = frozenset({"insufficient_quota"})


class LiquidAPIError(RuntimeError):
    """A non-success response. The provider's free-form message is never retained."""

    def __init__(self, status, code=None):
        super().__init__(f"Liquid API returned HTTP {status}")
        self.status = status if type(status) is int else None
        self.code = code if isinstance(code, str) and code.isidentifier() else None


def candidate(catalog=None):
    catalog = load_yaml("extractors") if catalog is None else catalog
    return catalog["hosted_candidates"][PROVIDER]


def review_config(cand):
    """The `campaign_limits` shape, so Liquid reuses the shared allocation checks."""
    return {
        "live_extraction": cand["live_extraction"],
        "cost": {"input_price_per_million": cand["input_price_per_million_usd"]},
    }


def settings(cand):
    return {
        "provider": PROVIDER,
        "model": cand["model"],
        "version": cand["version"],
        "template_version": TEMPLATE_VERSION,
        "api_mapping_version": MAPPING_VERSION,
        "request_configuration": {},
    }


def bind(request, cand):
    """The identical label-free payload under Liquid's own settings and cache identity."""
    require_nonempty_text(json.loads(request.payload_json))
    return IntendedRequest(request.payload_json, canonical_json(settings(cand)), request.provenance_json)


def pinned_version_ok(model, version):
    if not isinstance(model, str) or not isinstance(version, str) or not version.startswith(f"{model}@"):
        return False
    release = version[len(model) + 1:]
    try:
        return date.fromisoformat(release).isoformat() == release
    except ValueError:
        return False


def catalog_matches(settings, evidence):
    if not isinstance(evidence, dict) or not pinned_version_ok(settings.get("model"), settings.get("version")):
        return False
    release = settings["version"].removeprefix(settings["model"] + "@")
    return (evidence.get("source") == "models_endpoint_release_date" and
            evidence.get("model") == settings["model"] and evidence.get("requested_version") == settings["version"] and
            evidence.get("before_release") == evidence.get("after_release") == release)


def api_preview(request):
    """Build the JSON body without sending it or reading a credential."""
    payload = json.loads(request.payload_json)
    require_nonempty_text(payload)
    s = json.loads(request.settings_json)
    if s.get("provider") != PROVIDER:
        raise ValueError("This adapter supports hosted Liquid d1.2-ultra only")
    if s.get("model") not in DOCUMENTED_MODELS:
        raise ValueError("Only the documented exact hosted model id is allowed; no aliases")
    if not pinned_version_ok(s["model"], s.get("version")):
        raise ValueError("Pin '<model>@<release_date>' at campaign start; no moving version")
    if s.get("api_mapping_version") not in {MAPPING_VERSION, LEGACY_MAPPING_VERSION}:
        raise ValueError("Unknown provider mapping version.")
    if s.get("request_configuration"):
        raise ValueError("Additional provider parameters have not been verified.")
    question = semantic_question(payload["question"])
    if question["type"] == "score" and not 2 <= len(question["criteria"]) <= 10:
        raise ValueError("The documented Score limit is 2-10 levels.")
    return {"model": s["model"], "state": payload["state"], "questions": {QUESTION_ID: question}}


def credentials_available():
    return credential_present(CREDENTIAL_ENV, LOCAL_CREDENTIAL_PATH)


def readiness(cand, *, full=False):
    """Every gate and whether it passes; reading this never touches the network or a key."""
    review = cand["live_extraction"]
    gates = {
        "candidate_enabled": cand.get("enabled") is True,
        "live_extraction_enabled": review.get("enabled") is True,
        "account_created": cand.get("account_created") is True,
        "budget_authorised": cand.get("budget_authorised") is True,
        "version_pinned": pinned_version_ok(cand["model"], cand.get("version")),
        "credential_present": credentials_available(),
        "dataset_allocations": bool(review.get("dataset_budget_allocations_usd")),
        "pilot_authorization_manifest": bool(review.get("pilot_authorization_manifest")),
        **{flag: review.get(flag) is True for flag in READINESS_FLAGS},
    }
    if full:
        gates.update(
            pilot_accepted=review.get("pilot_accepted") is True,
            accepted_pilot_manifest=bool(review.get("accepted_pilot_manifest")),
            full_feature_creation_enabled=review.get("full_feature_creation_enabled") is True,
        )
    return gates


def require_ready(cand, *, full=False):
    missing = [name for name, ok in readiness(cand, full=full).items() if not ok]
    if missing:
        raise RuntimeError("Liquid d1.2-ultra is not authorised to run; unmet gates: " + ", ".join(missing))
    return campaign_limits(review_config(cand))


class LiquidContextEnvelope:
    """Documented byte ceilings only; the server enforces 65,536 tokens with HTTP 422.

    Liquid publishes no tokenizer, so no local token count is claimed. A 422 is stored
    as an ambiguous stop: never truncated, shortened or retried.
    """

    def __init__(self, cand):
        if cand["live_extraction"].get("context_policy_reviewed") is not True:
            raise ValueError("Review the provider-enforced context policy first")
        self.model = cand["model"]
        self.max_body = cand["max_request_body_bytes"]
        self.max_text = cand["max_text_bytes_per_question"]

    def __call__(self, request):
        wire = api_preview(request)
        if wire["model"] != self.model:
            raise ValueError("Context envelope belongs to a different model")
        if len(canonical_json(wire).encode("utf-8")) > self.max_body:
            raise ValueError("Request exceeds the documented body-size ceiling; no truncation")
        # JSON escaping only adds bytes, so this over-counts text and stays conservative.
        text = canonical_json([wire["state"], wire["questions"]]).encode("utf-8")
        if len(text) > self.max_text:
            raise ValueError("Request exceeds the documented per-question text ceiling; no truncation")


class UrllibHttp:
    """Minimal stdlib transport; tests inject a fake with the same `request` method."""

    def __init__(self, timeout=60):
        self.timeout = timeout

    def request(self, method, url, headers, body=None):
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:  # noqa: S310
                return response.status, {k.lower(): v for k, v in response.headers.items()}, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, {k.lower(): v for k, v in (exc.headers or {}).items()}, exc.read()

    def close(self):
        pass


def _error_code(body):
    try:
        error = json.loads(body).get("error")
    except (ValueError, AttributeError, TypeError):
        return None
    code = error.get("code") if isinstance(error, dict) else None
    return code.lower() if isinstance(code, str) else None


def _retry_after(headers):
    try:
        delay = float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None
    return delay if math.isfinite(delay) and delay >= 0 else None


class CatalogCheckedReply(dict):
    """The unmodified decision response with separately retained model-list observations."""

    def __init__(self, response, release_evidence):
        super().__init__(response)
        self.release_evidence = release_evidence


class LiquidTransport:
    """Refuses before reading a key unless every gate passes; never retries internally.

    Unbilled model-list reads bracket each new decision. The decision response names
    the model but does not identify a release or immutable weights.
    """

    def __init__(self, cand, *, allow_network=False, http=None):
        if allow_network is not True:
            raise RuntimeError("Network extraction is disabled")
        require_ready(cand)
        if paths.on_vsc():
            raise RuntimeError("Hosted Liquid runs locally; credentials never go to VSC")
        self.model, self.version = cand["model"], cand["version"]
        self.endpoint, self.models_endpoint = cand["endpoint"], cand["models_endpoint"]
        self.overload_unbilled = cand["live_extraction"].get("overload_rejections_unbilled_verified") is True
        self.http = http or UrllibHttp()
        self._headers = {
            "Authorization": "Bearer " + read_api_key(CREDENTIAL_ENV, LOCAL_CREDENTIAL_PATH),
            "Content-Type": "application/json",
        }
        self.release = self._served_release()
        if f"{self.model}@{self.release}" != self.version:
            raise ValueError("The served release differs from the pinned version; review before any call")

    def _served_release(self):
        status, _, body = self.http.request("GET", self.models_endpoint, dict(self._headers))
        if status != 200:
            raise LiquidAPIError(status, _error_code(body))
        models = json.loads(body).get("models")
        matches = [m for m in models or [] if isinstance(m, dict) and m.get("name") == self.model]
        if len(matches) != 1 or not isinstance(matches[0].get("release_date"), str):
            raise ValueError("The pinned model is not listed exactly once with a release date")
        return matches[0]["release_date"]

    def evaluate(self, request):
        body = api_preview(request)
        pinned = json.loads(request.settings_json)
        if (body["model"] != self.model or pinned["version"] != self.version
                or pinned.get("api_mapping_version") != MAPPING_VERSION):
            raise ReleaseNotDispatched("Transport and request release policy differ")
        try:
            before = self._served_release()
        except Exception as exc:
            raise ReleaseNotDispatched("Could not verify the catalog release before dispatch") from exc
        if before != self.release:
            raise ReleaseNotDispatched("Catalog release changed before dispatch")
        status, headers, raw = self.http.request(
            "POST", self.endpoint, dict(self._headers), canonical_json(body).encode("utf-8")
        )
        if status == 200:
            # Parse errors propagate as ambiguous: the call may have been billed.
            response = json.loads(raw)
            try:
                after = self._served_release()
                failure = None
            except Exception as exc:
                # Keep the original paid reply and usage even if the catalog is unavailable.
                after, failure = None, type(exc).__name__
            return CatalogCheckedReply(response, {
                "source": "models_endpoint_release_date", "model": self.model, "requested_version": self.version,
                "before_release": before, "after_release": after, "after_error": failure,
                "matches_pin": before == after == self.release,
            })
        code = _error_code(raw)
        if status == 402 or code in CREDIT_CODES:
            raise AccountCreditError(http_status=status)
        if status in OVERLOAD_STATUSES and self.overload_unbilled:
            raise RejectedRetryable("Rate or capacity rejection", retry_after_seconds=_retry_after(headers))
        raise LiquidAPIError(status, code)

    def close(self):
        if callable(getattr(self.http, "close", None)):
            self.http.close()


def dataset_store(dataset_id):
    return paths.jev_cache_path(f"liquid_ultra_cache/{paths.output_component(dataset_id)}.sqlite3")


def run(catalog=None, *, execute=False, full=False, dataset_ids=None, transport_factory=LiquidTransport):
    """Bounded pilot (first/middle/last/longest rows) or, after signed acceptance, full extraction.

    Verified responses stay in the Liquid store. Turning them into feature blocks needs
    roster registration and a signed native 0A review, which this module does not perform.
    """
    cand = candidate(catalog)
    if not execute:
        raise RuntimeError("Liquid execution requires explicit --execute")
    limits = require_ready(cand, full=full)
    if paths.on_vsc():
        raise RuntimeError("Hosted Liquid runs locally")
    from src.cluster.freeze import check_if_present
    from src.cluster.token_audit import selected_inputs
    from src.jev.cache import ResponseCache
    from src.jev.extraction import DatasetExtractor
    from src.jev.hosted import pilot_input_numbers
    from src.jev.inputs import iter_prepared_inputs, prepared_request
    from src.jev.prefix_pilot import ledger_total
    from src.utils.acceptance import reviewed_acceptance
    from src.utils.locking import file_lock
    from src.utils.timing import TimingJournal

    check_if_present()
    review = cand["live_extraction"]
    pinned = settings(cand)
    items = selected_inputs()
    if dataset_ids:
        if set(dataset_ids) - {i["dataset_id"] for i in items}:
            raise ValueError("Unselected dataset")
        items = [i for i in items if i["dataset_id"] in dataset_ids]
    if {i["dataset_id"] for i in items} - limits.keys():
        raise ValueError("Every selected dataset needs an approved Liquid spending allocation")
    input_sets = {i["dataset_id"]: i["input_set_id"] for i in items}
    reviewed_acceptance(
        review["pilot_authorization_manifest"],
        stage=PILOT_AUTHORIZATION_STAGE,
        expected={"settings_sha256": digest(pinned),
                  "campaign_spending_limit_usd": review["campaign_spending_limit_usd"]},
    )
    if full:
        reviewed_acceptance(
            review["accepted_pilot_manifest"],
            stage=PILOT_ACCEPTANCE_STAGE,
            expected={"settings_sha256": digest(pinned), "input_sets": input_sets},
        )
    selection = {
        i["dataset_id"]: None if full
        else pilot_input_numbers(paths.dataset_question_path(i["dataset_id"]), i["input_set_id"])
        for i in items
    }
    if not full and sum(map(len, selection.values())) > review["max_pilot_requests"]:
        raise ValueError("Pilot exceeds its approved request ceiling")
    phase, run_id = ("feature_creation" if full else "experiment_0"), uuid4().hex
    timings = TimingJournal(PROVIDER, phase=phase, run_id=run_id,
                            context={"model": cand["model"], "settings_sha256": digest(pinned)})
    report = {
        "run_id": run_id, "provider": PROVIDER, "status": "running",
        "mode": "full" if full else "pilot", "settings": pinned,
        "settings_sha256": digest(pinned), "input_sets": input_sets,
        "configuration_sha256": digest(cand), "datasets": [],
        "created_utc": datetime.now(UTC).isoformat(),
    }
    manifest = paths.manifests_dir(phase) / "hosted" / PROVIDER / f"{run_id}.json"
    write_json(manifest, report)
    gate = RateGate(next(iter(limits.values())))
    validator = LiquidContextEnvelope(cand)
    client = None
    try:
        with file_lock(paths.jev_cache_path("liquid_ultra_cache") / ".campaign.lock", timeout=1):
            client = transport_factory(cand, allow_network=True)
            report["served_release"] = client.release
            for item in items:
                dataset_id = item["dataset_id"]
                store = dataset_store(dataset_id)
                with file_lock(store.with_suffix(".extraction.lock"), timeout=1):
                    cache = ResponseCache(store, dataset_id=dataset_id)
                    extractor = DatasetExtractor(
                        cache, client, limits[dataset_id], gate=gate, validate_context=validator,
                        phase=phase, policy_revision_manifest=review.get("spending_policy_revision_manifest"),
                        timings=timings,
                        log=paths.logs_dir(phase) / "extraction" / PROVIDER / f"{dataset_id}.log",
                    )
                    records = iter_prepared_inputs(
                        paths.dataset_question_path(dataset_id), input_set_id=item["input_set_id"],
                        input_numbers=selection[dataset_id],
                    )
                    counts = extractor.run(
                        prepared_request(r, settings=pinned, provenance={"dataset_id": dataset_id})
                        for r in records
                    )
                    report["datasets"].append({
                        "dataset_id": dataset_id, "input_set_id": item["input_set_id"],
                        "spending_policy_sha256": extractor.policy_sha256, "counts": counts,
                        "ledger_usd": ledger_total(cache.path),
                        "observed_response_models": extractor.observed_response_models(),
                    })
                    write_json(manifest, report)
                    if set(counts) - {"completed", "cache_hit", "recovered_response"}:
                        report["status"] = "stopped_requires_review"
                        if extractor.stop_details:
                            report["stop_details"] = {"dataset_id": dataset_id, **extractor.stop_details}
                        break
            else:
                report["status"] = "completed_requires_review"
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        report["timing"] = timings.write_summary()
        write_json(manifest, report)
        if client is not None:
            client.close()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--dataset", action="append")
    args = parser.parse_args(argv)
    if not args.execute:
        gates = readiness(candidate(), full=args.full)
        print(json.dumps({"provider": PROVIDER, "network": False, "gates": gates}, indent=2))
        return 0
    result = run(execute=True, full=args.full, dataset_ids=args.dataset)
    return int(result["status"] != "completed_requires_review")


if __name__ == "__main__":
    raise SystemExit(main())
