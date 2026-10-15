"""Opt-in extraction, persistent reservations and bounded dispatch. Nothing runs on import.

Each dataset receives a cumulative spending allocation. Reviewed policy revisions
preserve prior spending and never clear reservations. The campaign allocation sum must
be <= the owner's spending limit. Reservations and actual usage survive in the
dataset's single SQLite file; logs contain IDs/counters, never credentials or text.
A lost/ambiguous response is NOT automatically retried: billing may already have occurred.
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from threading import Lock

from src.jev.provider import api_preview, response_features
from src.jev.spending import register_policy
from src.utils import paths
from src.utils.events import append_event
from src.utils.serialization import canonical_json, digest


class RejectedRetryable(RuntimeError):
    """A verified, unbilled rejection (e.g. HTTP 429); safe to retry within the attempt budget."""

    def __init__(self, message="Rate-limit rejection", retry_after_seconds=None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


@dataclass(frozen=True)
class ExtractionLimits:
    dataset_budget_usd: float
    input_price_per_million: float
    max_billed_input_tokens: int
    requests_per_second: float
    tokens_per_second: float
    workers: int = 1
    max_attempts: int = 3
    backoff_initial_seconds: float = 1.0
    backoff_max_seconds: float = 60.0

    def __post_init__(self):
        positive = (
            self.dataset_budget_usd,
            self.input_price_per_million,
            self.requests_per_second,
            self.tokens_per_second,
            self.backoff_initial_seconds,
            self.backoff_max_seconds,
        )
        if any(not math.isfinite(v) or v <= 0 for v in positive):
            raise ValueError("Explicit positive budget, verified price and account rates required")
        if any(
            not isinstance(v, int) or isinstance(v, bool) or v < 1
            for v in (self.max_billed_input_tokens, self.workers, self.max_attempts)
        ):
            raise ValueError("Token bound, workers and attempt budget must be positive integers")


def campaign_limits(cfg):
    """Validate the aggregate approved allocation before opening any transport."""
    review = cfg["live_extraction"]
    allocation = review["dataset_budget_allocations_usd"]
    ceiling = review["campaign_spending_limit_usd"]
    if not allocation or ceiling is None or not math.isfinite(ceiling) or ceiling <= 0:
        raise ValueError("An explicit campaign limit and dataset allocations are required")
    if any(not math.isfinite(v) or v <= 0 for v in allocation.values()):
        raise ValueError("Each dataset allocation must be positive")
    if sum(allocation.values()) > ceiling + 1e-12:
        raise ValueError("Dataset allocations exceed the campaign spending limit")
    return {
        dataset: ExtractionLimits(
            budget,
            cfg["cost"]["input_price_per_million"],
            review["max_billed_input_tokens_per_attempt"],
            review["requests_per_second"],
            review["input_tokens_per_second"],
            workers=review["workers"],
        )
        for dataset, budget in allocation.items()
    }


class RateGate:
    """Conservative non-bursting gate; share ONE instance across concurrent datasets.

    Reserve the full verified billing-token upper bound, not the char/4 approximation.
    Server account limits must be supplied, never taken from hypothetical review config.
    """

    def __init__(self, limits, *, sleep=time.sleep, clock=time.monotonic):
        self.interval = max(
            1 / limits.requests_per_second,
            limits.max_billed_input_tokens / limits.tokens_per_second,
        )
        self.sleep, self.clock = sleep, clock
        self.lock, self.next_start = Lock(), 0.0

    def acquire(self):
        with self.lock:
            now = self.clock()
            delay = max(0.0, self.next_start - now)
            if delay:
                self.sleep(delay)
            self.next_start = self.clock() + self.interval


class TypeSafeTransport:
    """Documentation-verified SDK wrapper, still requiring an authorized live pilot.

    Internal SDK retries are off. Imports/key access occur ONLY after the opt-in
    and readiness checks. Auth/model setup may itself contact the provider.
    """

    def __init__(self, cfg, *, allow_network=False):
        review = cfg.get("live_extraction", {})
        required = (
            "pilot_authorized",
            "account_limits_verified",
            "billing_bound_verified",
            "context_policy_reviewed",
            "price_verified",
        )
        if allow_network is not True or review.get("enabled") is not True:
            raise RuntimeError("Network extraction is disabled")
        if not all(review.get(k) is True for k in required):
            raise ValueError("Live extraction readiness checks are incomplete")
        campaign_limits(cfg)
        if cfg["jev"]["model"] != f"jev-{cfg['jev']['version']}":
            raise ValueError("A fixed matching Jev model/version is required")
        from typesafe_sdk import RetryPolicy, TypeSafeClient

        self.model = cfg["jev"]["model"]
        self.client = TypeSafeClient(
            api_key=os.environ["TYPESAFE_API_KEY"],
            model=self.model,
            retry=RetryPolicy(max_retries=0, timeout=60),
        )

    def evaluate(self, request):
        from typesafe_sdk import TypeSafeAPIError

        body = api_preview(request)
        if body["model"] != self.model:
            raise ValueError("Transport and request model pins differ")
        try:
            result = self.client.system_one(body["state"], body["questions"])
        except TypeSafeAPIError as exc:
            # Only a known rejected request is retried; unknown billing stays reserved.
            if exc.status == 429:
                retry_after = exc.headers.get("retry-after")
                try:
                    delay = float(retry_after) if retry_after else None
                    if delay is not None and (not math.isfinite(delay) or delay < 0):
                        delay = None
                except ValueError:
                    delay = None
                raise RejectedRetryable("Rate-limit rejection", retry_after_seconds=delay) from exc
            raise
        return result.raw_http_response.json()

    def close(self):
        self.client.close()


class DatasetExtractor:
    def __init__(
        self,
        cache,
        client,
        limits,
        *,
        gate=None,
        validate_context=None,
        sleep=time.sleep,
        phase="feature_creation",
        policy_revision_manifest=None,
    ):
        if cache.dataset_id is None:
            raise ValueError("Extraction requires a dataset-bound cache")
        if validate_context is None:
            raise ValueError("Supply a reviewed context validator; no silent truncation")
        self.cache, self.client, self.limits = cache, client, limits
        self.gate = gate or RateGate(limits)
        self.validate_context = validate_context
        self.sleep = sleep
        self.session_id = uuid.uuid4().hex
        self.log = paths.logs_dir(phase) / "extraction" / f"{cache.dataset_id}.jsonl"
        self.policy_sha256 = register_policy(
            cache, limits, revision_manifest=policy_revision_manifest, stage=phase
        )

    def _reserve(self, request):
        reservation = (
            self.limits.max_billed_input_tokens * self.limits.input_price_per_million / 1e6
        )
        with self.cache._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            active = db.execute("SELECT settings_json FROM extraction_policy WHERE id=1").fetchone()
            if not active or digest(json.loads(active[0])) != self.policy_sha256:
                raise ValueError("Spending policy changed; restart this extractor before dispatch")
            db.execute(
                "INSERT OR IGNORE INTO extraction_jobs(request_key,request_json,status) VALUES (?,?,'pending')",
                (request.cache_key, request.to_json()),
            )
            self.cache._trace(db, request, "verified_response")
            status, attempts, owner = db.execute(
                "SELECT status,attempts,owner_session FROM extraction_jobs WHERE request_key=?",
                (request.cache_key,),
            ).fetchone()
            if status == "sending":
                return "in_flight_reuse" if owner == self.session_id else "ambiguous"
            if status != "pending":
                return status
            if attempts >= self.limits.max_attempts:
                return "attempt_limit"
            total = db.execute(
                "SELECT COALESCE(SUM(reserved_usd+spent_usd),0) FROM extraction_jobs"
            ).fetchone()[0]
            if total + reservation > self.limits.dataset_budget_usd + 1e-12:
                return "budget_stop"
            db.execute(
                "UPDATE extraction_jobs SET status='sending',attempts=attempts+1,reserved_usd=?,owner_session=?,policy_sha256=? WHERE request_key=?",
                (reservation, self.session_id, self.policy_sha256, request.cache_key),
            )
        return "claimed"

    def _update(self, request, status, *, response=None, error=None, rejected=False):
        tokens = None
        if response is not None:
            tokens = response.get("usage", {}).get("input_tokens")
            if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
                tokens = None
                status = "ambiguous_usage"
        with self.cache._connect() as db:
            if tokens is not None:
                db.execute(
                    "UPDATE extraction_jobs SET reserved_usd=0,spent_usd=spent_usd+?,input_tokens=COALESCE(input_tokens,0)+? WHERE request_key=?",
                    (tokens * self.limits.input_price_per_million / 1e6, tokens, request.cache_key),
                )
            elif rejected:
                db.execute(
                    "UPDATE extraction_jobs SET reserved_usd=0 WHERE request_key=?",
                    (request.cache_key,),
                )
            db.execute(
                "UPDATE extraction_jobs SET status=?,raw_response_json=COALESCE(?,raw_response_json),error_type=? WHERE request_key=?",
                (
                    status,
                    canonical_json(response) if response is not None else None,
                    error,
                    request.cache_key,
                ),
            )
        append_event(
            self.log,
            status,
            request_key=request.cache_key,
            input_tokens=tokens,
            error_type=error,
            policy_sha256=self.policy_sha256,
        )
        return status

    def evaluate(self, request):
        cached = self.cache.get(request, kind="verified_response")
        if cached is not None:
            response_features(request, cached)
            append_event(self.log, "cache_hit", request_key=request.cache_key)
            return "cache_hit"
        with self.cache._connect() as db:
            saved = db.execute(
                "SELECT status,raw_response_json FROM extraction_jobs WHERE request_key=?",
                (request.cache_key,),
            ).fetchone()
        if saved and saved[0] in {"received", "completed"} and saved[1]:
            # A crash after recording usage/raw response must not require another paid call.
            response = json.loads(saved[1])
            response_features(request, response)
            self.cache.put(request, response, kind="verified_response")
            self._update(request, "completed")
            return "recovered_response"
        api_preview(request)
        self.validate_context(request)  # Fails BEFORE reserving money or calling a client.
        for attempt in range(self.limits.max_attempts):
            status = self._reserve(request)
            if status != "claimed":
                append_event(self.log, status, request_key=request.cache_key)
                return status
            try:
                self.gate.acquire()
                append_event(
                    self.log,
                    "request_started",
                    request_key=request.cache_key,
                    policy_sha256=self.policy_sha256,
                )
                response = self.client.evaluate(request)
            except RejectedRetryable as exc:
                self._update(request, "pending", rejected=True, error="RejectedRetryable")
                if attempt + 1 < self.limits.max_attempts:
                    delay = min(
                        self.limits.backoff_max_seconds,
                        self.limits.backoff_initial_seconds * 2**attempt,
                    )
                    if exc.retry_after_seconds is not None:
                        delay = max(delay, exc.retry_after_seconds)
                    self.sleep(delay)
                continue
            except Exception as exc:
                return self._update(request, "ambiguous", error=type(exc).__name__)
            status = self._update(request, "received", response=response)
            if status == "ambiguous_usage":
                return status
            if response["usage"]["input_tokens"] > self.limits.max_billed_input_tokens:
                self._update(request, "billing_bound_exceeded", error="BillingBoundViolation")
                raise RuntimeError("Verified billing-token bound was exceeded; stop extraction")
            try:
                response_features(request, response)
                self.cache.put(request, response, kind="verified_response")
            except Exception as exc:
                return self._update(request, "invalid_response", error=type(exc).__name__)
            return self._update(request, "completed")
        return "attempt_limit"

    def run(self, requests):
        """Bound queued futures by worker count; accept a streaming request iterator."""
        iterator = iter(requests)
        counts = {}
        with ThreadPoolExecutor(max_workers=self.limits.workers) as pool:
            pending = set()
            exhausted = False
            stop = False
            while pending or not exhausted:
                while not exhausted and not stop and len(pending) < self.limits.workers:
                    try:
                        pending.add(pool.submit(self.evaluate, next(iterator)))
                    except StopIteration:
                        exhausted = True
                if not pending:
                    break
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    status = future.result()
                    counts[status] = counts.get(status, 0) + 1
                    stop |= status in {
                        "budget_stop",
                        "ambiguous",
                        "ambiguous_usage",
                        "billing_bound_exceeded",
                        "invalid_response",
                        "sending",
                        "attempt_limit",
                    }
                if stop:
                    exhausted = True
        append_event(self.log, "dispatch_finished", counts=counts)
        return counts


if __name__ == "__main__":
    from src.jev.hosted import main

    raise SystemExit(main())
