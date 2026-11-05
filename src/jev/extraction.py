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
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from threading import Event, Lock

from src.jev.provider import api_preview, response_features
from src.jev.spending import register_policy
from src.utils import paths
from src.utils.events import append_event
from src.utils.serialization import canonical_json, digest
from src.utils.timing import TimingJournal


class RejectedRetryable(RuntimeError):
    """A verified, unbilled rejection (e.g. HTTP 429); safe to retry within the attempt budget."""

    def __init__(self, message="Rate-limit rejection", retry_after_seconds=None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class AccountCreditError(RuntimeError):
    """A payment/credit rejection; billing is unresolved until explicitly reviewed.

    TypeSafe does not document an exhausted-balance response or an unbilled guarantee.
    Recognize HTTP 402 and explicit structured credit codes conservatively. Never log
    the provider's free-form message, which could contain credentials or source text.
    """

    CODES = frozenset({
        "insufficient_credits", "insufficient_balance", "credit_balance_exhausted",
        "credits_exhausted", "out_of_credits", "payment_required",
    })

    def __init__(self, *, http_status=None, code=None):
        super().__init__("Account payment or credit limit requires review before resuming")
        self.http_status = http_status if type(http_status) is int else None
        self.code = code if code in self.CODES else None

    @classmethod
    def from_api_error(cls, error):
        body = error.body
        nested = body.get("error") if isinstance(body, dict) else None
        candidates = [body.get("code")] if isinstance(body, dict) else []
        if isinstance(nested, dict):
            candidates += [nested.get("code"), nested.get("type")]
        code = next((c.lower() for c in candidates
                     if isinstance(c, str) and c.lower() in cls.CODES), None)
        if error.status == 402 or code is not None:
            return cls(http_status=error.status, code=code)
        return None


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
    if review.get("owner_budget_eur") is not None:
        rate = review.get("usd_per_eur")
        euros = review["owner_budget_eur"]
        if (not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0
                or not isinstance(euros, (int, float)) or not math.isfinite(euros) or euros <= 0
                or not review.get("exchange_rate_evidence")):
            raise ValueError("Verify EUR-to-USD conversion before spending the owner's euro budget")
        if ceiling > euros * rate + 1e-12:
            raise ValueError("USD campaign limit exceeds the approved euro budget")
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
    """Non-bursting request gate; share ONE instance across concurrent datasets.

    Default pacing uses the full billing-token upper bound. An explicitly reviewed
    faster mode lets the server enforce TPS, with shared rejection backoff. This does
    not lower the monetary reservation or turn a character estimate into a token bound.
    """

    def __init__(self, limits, *, sleep=time.sleep, clock=time.monotonic,
                 provider_enforced_tokens=False):
        self.interval = 1 / limits.requests_per_second
        if not provider_enforced_tokens:
            self.interval = max(self.interval, limits.max_billed_input_tokens / limits.tokens_per_second)
        self.sleep, self.clock = sleep, clock
        self.lock, self.next_start = Lock(), 0.0

    def acquire(self):
        with self.lock:
            now = self.clock()
            delay = max(0.0, self.next_start - now)
            if delay:
                self.sleep(delay)
            self.next_start = self.clock() + self.interval

    def defer(self, seconds):
        """Share server backoff across workers instead of each continuing to dispatch."""
        with self.lock:
            self.next_start = max(self.next_start, self.clock() + seconds)


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

        from src.jev.auth import api_key

        self.model = cfg["jev"]["model"]
        self.client = TypeSafeClient(
            api_key=api_key(),
            base_url="https://api.typesafe.ai",
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
            credit_error = AccountCreditError.from_api_error(exc)
            if credit_error is not None:
                raise credit_error from exc
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
        timings=None,
    ):
        if cache.dataset_id is None:
            raise ValueError("Extraction requires a dataset-bound cache")
        if validate_context is None:
            raise ValueError("Supply a reviewed context validator; no silent truncation")
        self.cache, self.client, self.limits = cache, client, limits
        self.gate = gate or RateGate(limits)
        self.validate_context = validate_context
        self.sleep = sleep
        self.account_stopped = Event()
        self.stop_details = None
        self._stop_lock = Lock()
        self.session_id = uuid.uuid4().hex
        self.timings = timings or TimingJournal("jev", phase=phase, run_id=self.session_id)
        self.log = paths.logs_dir(phase) / "extraction" / f"{cache.dataset_id}.log"
        self.policy_sha256 = register_policy(
            cache, limits, revision_manifest=policy_revision_manifest, stage=phase
        )

    def _stop_account(self, request, error=None):
        self.account_stopped.set()
        with self._stop_lock:
            if self.stop_details is None:
                self.stop_details = {
                    "reason": "account_credit_or_payment_required",
                    "request_key": request.cache_key,
                    "http_status": error.http_status if error else None,
                    "provider_code": error.code if error else None,
                    "billing": "unresolved_reservation_preserved",
                    "next_action": "Check balance and billing, reconcile the blocked request, then resume",
                }
                append_event(self.log, "account_stopped", **self.stop_details)

    def _release_unsent(self, request):
        """A sibling hit the account stop while this worker waited; no call was made."""
        with self.cache._connect() as db:
            cursor = db.execute(
                "UPDATE extraction_jobs SET status='pending',reserved_usd=0,"
                "attempts=attempts-1,owner_session=NULL WHERE request_key=? "
                "AND status='sending' AND owner_session=? AND attempts>0",
                (request.cache_key, self.session_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Cannot release an unsent reservation no longer owned by this worker")
        append_event(self.log, "account_stopped_before_dispatch", request_key=request.cache_key)
        return "account_stopped"

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
            self.timings.record("cache_reuse", dataset_id=self.cache.dataset_id,
                                item_keys=[request.cache_key])
            return "cache_hit"
        with self.cache._connect() as db:
            saved = db.execute(
                "SELECT status,raw_response_json,error_type FROM extraction_jobs WHERE request_key=?",
                (request.cache_key,),
            ).fetchone()
        if saved and saved[0] == "ambiguous" and saved[2] == "AccountCreditError":
            self._stop_account(request)
            return "ambiguous"
        if saved and saved[0] in {"received", "completed", "invalid_response"} and saved[1]:
            # A crash after recording usage/raw response must not require another paid call.
            # Revalidate a paid response after a decoder fix; never dispatch it again.
            response = json.loads(saved[1])
            try:
                response_features(request, response)
                self.cache.put(request, response, kind="verified_response")
            except Exception as exc:
                return self._update(request, "invalid_response", error=type(exc).__name__)
            self._update(request, "completed")
            self.timings.record("recovered_response", dataset_id=self.cache.dataset_id,
                                item_keys=[request.cache_key])
            return "recovered_response"
        api_preview(request)
        self.validate_context(request)  # Fails BEFORE reserving money or calling a client.
        for attempt in range(self.limits.max_attempts):
            if self.account_stopped.is_set():
                return "account_stopped"
            status = self._reserve(request)
            if status != "claimed":
                append_event(self.log, status, request_key=request.cache_key)
                return status
            started = None
            try:
                with self.timings.measure("rate_limit_wait", dataset_id=self.cache.dataset_id):
                    self.gate.acquire()
                if self.account_stopped.is_set():
                    return self._release_unsent(request)
                append_event(
                    self.log,
                    "request_started",
                    request_key=request.cache_key,
                    policy_sha256=self.policy_sha256,
                )
                started = time.perf_counter()
                response = self.client.evaluate(request)
                seconds = time.perf_counter() - started
            except AccountCreditError as exc:
                self._stop_account(request, exc)
                # A top-up alone must never silently repay an uncertain attempt.
                status = self._update(request, "ambiguous", error="AccountCreditError")
                if started is not None:
                    self.timings.record(
                        "api_request", dataset_id=self.cache.dataset_id,
                        item_keys=[request.cache_key], seconds=time.perf_counter() - started,
                        error_type="AccountCreditError",
                    )
                return status
            except RejectedRetryable as exc:
                self._update(request, "pending", rejected=True, error="RejectedRetryable")
                self.timings.record(
                    "api_request", dataset_id=self.cache.dataset_id,
                    item_keys=[request.cache_key], seconds=time.perf_counter() - started,
                    error_type=type(exc).__name__,
                )
                if attempt + 1 < self.limits.max_attempts:
                    delay = min(
                        self.limits.backoff_max_seconds,
                        self.limits.backoff_initial_seconds * 2**attempt,
                    )
                    if exc.retry_after_seconds is not None:
                        delay = max(delay, exc.retry_after_seconds)
                    if callable(getattr(self.gate, "defer", None)):
                        self.gate.defer(delay)
                    else:
                        self.sleep(delay)
                continue
            except Exception as exc:
                status = self._update(request, "ambiguous", error=type(exc).__name__)
                http_status = getattr(exc, "status", None)
                if type(http_status) is int:
                    append_event(self.log, "api_error", request_key=request.cache_key,
                                 http_status=http_status, error_type=type(exc).__name__)
                if started is not None:
                    self.timings.record(
                        "api_request", dataset_id=self.cache.dataset_id,
                        item_keys=[request.cache_key], seconds=time.perf_counter() - started,
                        error_type=type(exc).__name__,
                    )
                return status
            status = self._update(request, "received", response=response)
            # Preserve the paid raw response/accounting BEFORE writing optional timing evidence.
            self.timings.record("api_request", dataset_id=self.cache.dataset_id,
                                item_keys=[request.cache_key], seconds=seconds)
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
                        "account_stopped",
                        "ambiguous",
                        "ambiguous_usage",
                        "billing_bound_exceeded",
                        "invalid_response",
                        "sending",
                        "attempt_limit",
                        "resolved_no_response",
                    }
                if stop:
                    exhausted = True
        append_event(self.log, "dispatch_finished", counts=counts,
                     timing=self.timings.write_summary())
        return counts


if __name__ == "__main__":
    from src.jev.hosted import main

    raise SystemExit(main())
