"""Documentation-verified TypeSafe wire preview and response decoding; NO HTTP client.

Reference: https://docs.typesafe.ai/api, checked 2026-09-29.
Live behaviour still requires a separately approved pilot. Stable option IDs preserve
the distinction between labels such as numeric 1 and string "1".
"""

from __future__ import annotations

import json
import math

from src.jev.requests import IntendedRequest, require_nonempty_text
from src.utils.serialization import canonical_json, digest

MAPPING_VERSION = "typesafe-systemone-v1"
QUESTION_ID = "semantic"
SCORE_VALIDATION_VERSION = "jev-two-decimal-envelope-v1"
PROBABILITY_VALIDATION_VERSION = "jev-two-decimal-simplex-v1"
# Strands' pinned infer._to_answer rounds probabilities AND weighted scores to
# four decimals. Jev's two-decimal precision is observed wire behaviour. Neither
# policy applies to another runtime, and neither reconstructs lost precision.
ROUNDING_DECIMALS = {"jev": 2, "strands": 4}


def _rounding_half_step(values, provider):
    digits = ROUNDING_DECIMALS.get(provider)
    if digits is None:
        return None
    scale = 10**digits
    if not all(math.isclose(v * scale, round(v * scale), rel_tol=0, abs_tol=1e-8) for v in values):
        return None
    return 0.5 / scale


def probability_distribution(values, provider):
    """Normalize only documented/observed provider rounding, retaining raw responses.

    Find the closest sum-one vector in squared distance, with every entry within
    half its provider's rounding step from its reported value. This is a bounded policy,
    not a provider precision guarantee or recovery of the unknown unrounded values.
    Jev uses two decimals; pinned Strands infer._to_answer uses four. Other
    open-weight outputs keep the original strict sum check.
    """
    values = [_number(v, 0, 1) for v in values]
    total = math.fsum(values)
    normalized = values
    adjusted = not math.isclose(total, 1, rel_tol=0, abs_tol=1e-5)
    if adjusted:
        half_step = _rounding_half_step(values, provider)
        if half_step is None or total <= 0:
            raise ValueError("Response probabilities do not sum to one.")
        lows = [max(0, v - half_step) for v in values]
        highs = [min(1, v + half_step) for v in values]
        if math.fsum(lows) > 1 + 1e-12 or math.fsum(highs) < 1 - 1e-12:
            raise ValueError("Rounded probabilities cannot represent a distribution.")
        # Projection onto the bounded probability simplex has form clip(p + shift).
        # Bisection is permutation-equivariant; class order cannot affect the repair.
        left, right = -half_step, half_step
        for _ in range(60):
            shift = (left + right) / 2
            normalized = [min(hi, max(lo, v + shift)) for v, lo, hi in zip(values, lows, highs)]
            if math.fsum(normalized) < 1:
                left = shift
            else:
                right = shift
        if not math.isclose(math.fsum(normalized), 1, abs_tol=1e-10, rel_tol=0):
            raise ValueError("Probability rounding reconstruction failed.")
    return normalized, {
        "policy": (
            "strands-four-decimal-simplex-v1" if provider == "strands"
            else PROBABILITY_VALIDATION_VERSION
        ),
        "reported_sum": total,
        "normalization_applied": adjusted,
        "maximum_absolute_adjustment": max(
            (abs(a - b) for a, b in zip(values, normalized)), default=0
        ),
    }


def score_consistency(score, values, provider):
    """Keep full-precision runtimes strict; allow bounded provider score rounding.

    Jev 1.13 returned score=7.99 beside p(8)=1.00 on 20 October. Its documentation
    defines a weighted mean but does not promise a decimal precision. We therefore
    accept a discrepancy only if a normalized latent distribution and score could
    both round to the observed two-decimal fields. Strands' pinned infer._to_answer
    explicitly rounds both fields to four decimals, so its envelope is 100x narrower.
    This is an explicit compatibility
    policy, not a claim that the provider guarantees this rounding.
    """
    expected = sum(i * p for i, p in enumerate(values))
    lower = upper = expected
    exact = math.isclose(score, expected, abs_tol=1e-4)
    half_step = _rounding_half_step([score, *values], provider)
    if not exact and half_step is not None:
        lows = [max(0, p - half_step) for p in values]
        highs = [min(1, p + half_step) for p in values]

        def extreme(order):
            candidate = lows.copy()
            remaining = 1 - sum(candidate)
            for i in order:
                addition = min(remaining, highs[i] - candidate[i])
                candidate[i] += addition
                remaining -= addition
            if abs(remaining) > 1e-8:
                raise ValueError("Rounded probabilities cannot represent a distribution.")
            return sum(i * p for i, p in enumerate(candidate))

        lower = extreme(range(len(values))) - half_step
        upper = extreme(reversed(range(len(values)))) + half_step
    if not exact and not lower - 1e-8 <= score <= upper + 1e-8:
        raise ValueError("Regression score is inconsistent with its probabilities.")
    return {
        "policy": (
            "strands-four-decimal-envelope-v1" if provider == "strands"
            else SCORE_VALIDATION_VERSION
        ),
        "reported_score": score,
        "probability_weighted_score": expected,
        "absolute_difference": abs(score - expected),
        "rounding_envelope_used": not exact,
    }


def api_preview(request: IntendedRequest) -> dict:
    """Build a JSON body without sending it or reading a credential."""
    payload = json.loads(request.payload_json)
    require_nonempty_text(payload)
    settings = json.loads(request.settings_json)
    if settings.get("provider") != "jev":
        raise ValueError("This response/request adapter supports Jev only")
    if not settings["version"] or settings["model"] != f"jev-{settings['version']}":
        raise ValueError("An explicit, matching model/version pin is required; no moving aliases.")
    if settings.get("api_mapping_version") != MAPPING_VERSION:
        raise ValueError("Unknown provider mapping version.")
    if settings["request_configuration"]:
        raise ValueError("Additional provider parameters have not been verified.")
    return {
        "model": settings["model"],
        "state": payload["state"],
        "questions": {QUESTION_ID: semantic_question(payload["question"])},
    }


def semantic_question(question: dict) -> dict:
    """The shared Noul/Choice/Score question semantics, independent of HTTP transport."""
    mapped = {"instructions": question["text"]}
    if question["kind"] == "binary_probability":
        mapped["type"] = "noul"
    elif question["kind"] == "choice":
        choices = question["allowed_answers"]
        if not 2 <= len(choices) <= 255:
            raise ValueError("The documented Choice limit is 2–255 options.")
        descriptions = question.get("answer_descriptions")
        if descriptions is not None and (
            len(descriptions) != len(choices)
            or any(not isinstance(d, str) or not d.strip() for d in descriptions)
        ):
            raise ValueError("Each raw class needs exactly one nonempty description")
        mapped.update(
            type="choice",
            criteria={
                f"c{i:03d}": canonical_json(c) + (" — " + descriptions[i] if descriptions else "")
                for i, c in enumerate(choices)
            },
        )
    elif question["kind"] == "ordered_choice":
        if [level["score"] for level in question["allowed_answers"]] != list(range(-4, 5)):
            raise ValueError("Expected the agreed nine-level regression scale.")
        mapped.update(
            type="score", criteria=[level["meaning"] for level in question["allowed_answers"]]
        )
    else:
        raise ValueError("Unsupported question kind.")
    return mapped


def _number(value, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Response values must be finite numbers.")
    if not low <= value <= high:
        raise ValueError("Response value outside the allowed range.")
    return float(value)


def response_features(request: IntendedRequest, response: dict) -> dict[str, float]:
    """Decode responses with only provider-specific bounded rounding correction."""
    settings = json.loads(request.settings_json)
    provider = settings.get("provider", "jev")
    if provider == "jev":
        wire = api_preview(request)
    else:
        from src.jev.catalog import SELF_HOSTED

        payload = json.loads(request.payload_json)
        require_nonempty_text(payload)
        if (
            provider not in SELF_HOSTED
            or settings.get("api_mapping_version") != f"{provider}-self-hosted-v1"
        ):
            raise ValueError("Provider response decoder is not implemented for this mapping")
        if (
            response.get("provider") != provider
            or response.get("version") != settings["version"]
            or response.get("settings_sha256") != digest(settings)
        ):
            raise ValueError("Self-hosted response differs from its pinned provider/runtime")
        wire = {
            "model": settings["model"],
            "questions": {QUESTION_ID: semantic_question(payload["question"])},
        }
    if response.get("model") != wire["model"]:
        raise ValueError("Response model differs from the pinned request model.")
    if set(response.get("answers", {})) != {QUESTION_ID}:
        raise ValueError("Response must contain exactly the requested semantic answer.")
    answer = response["answers"][QUESTION_ID]
    question = wire["questions"][QUESTION_ID]
    if answer.get("type") != question["type"]:
        raise ValueError("Response type does not match the request.")
    if answer["type"] == "noul":
        return {"p_positive": _number(answer.get("noul"), 0, 1)}
    keys = list(question["criteria"]) if answer["type"] == "choice" else [str(i) for i in range(9)]
    probabilities = answer.get("probabilities", {})
    if set(probabilities) != set(keys):
        raise ValueError("Response probability keys do not match the complete answer space.")
    raw_values = [_number(probabilities[k], 0, 1) for k in keys]
    values, _ = probability_distribution(raw_values, provider)
    features = {f"p_{key}": value for key, value in zip(keys, values)}
    features["confidence"] = _number(answer.get("confidence"), 0, 1)
    if answer["type"] == "choice":
        if answer.get("choice") not in keys:
            raise ValueError("Selected class is outside the allowed answer space.")
        if probabilities[answer["choice"]] < max(raw_values) - 1e-5:
            raise ValueError("Selected class is inconsistent with its probabilities.")
    else:
        if answer.get("legend") != dict(zip(keys, question["criteria"])):
            raise ValueError("Regression legend differs from the requested scale.")
        score = _number(answer.get("score"), 0, 8)
        score_consistency(score, raw_values, provider)
        expected = sum(i * p for i, p in enumerate(values))
        # Jev indexes levels 0..8; our agreed directional values are -4..+4.
        features["expected_direction"] = expected - 4
    return features
