"""Documentation-verified TypeSafe wire preview and response decoding; NO HTTP client.

Reference: https://docs.typesafe.ai/api, checked 2026-09-29.
Live behaviour still requires a separately approved pilot. Stable option IDs preserve
the distinction between labels such as numeric 1 and string "1".
"""

from __future__ import annotations

import json
import math

from src.jev.requests import IntendedRequest
from src.utils.serialization import canonical_json

MAPPING_VERSION = "typesafe-systemone-v1"
QUESTION_ID = "semantic"


def api_preview(request: IntendedRequest) -> dict:
    """Build a JSON body without sending it or reading a credential."""
    payload = json.loads(request.payload_json)
    settings = json.loads(request.settings_json)
    if not settings["version"] or settings["model"] != f"jev-{settings['version']}":
        raise ValueError("An explicit, matching model/version pin is required; no moving aliases.")
    if settings.get("api_mapping_version") != MAPPING_VERSION:
        raise ValueError("Unknown provider mapping version.")
    if settings["request_configuration"]:
        raise ValueError("Additional provider parameters have not been verified.")
    question = payload["question"]
    mapped = {"instructions": question["text"]}
    if question["kind"] == "binary_probability":
        mapped["type"] = "noul"
    elif question["kind"] == "choice":
        choices = question["allowed_answers"]
        if not 2 <= len(choices) <= 255:
            raise ValueError("The documented Choice limit is 2–255 options.")
        mapped.update(
            type="choice", criteria={f"c{i:03d}": canonical_json(c) for i, c in enumerate(choices)}
        )
    elif question["kind"] == "ordered_choice":
        if [level["score"] for level in question["allowed_answers"]] != list(range(-4, 5)):
            raise ValueError("Expected the agreed nine-level regression scale.")
        mapped.update(
            type="score", criteria=[level["meaning"] for level in question["allowed_answers"]]
        )
    else:
        raise ValueError("Unsupported question kind.")
    return {
        "model": settings["model"],
        "state": payload["state"],
        "questions": {QUESTION_ID: mapped},
    }


def _number(value, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Response values must be finite numbers.")
    if not low <= value <= high:
        raise ValueError("Response value outside the allowed range.")
    return float(value)


def response_features(request: IntendedRequest, response: dict) -> dict[str, float]:
    """Validate a supplied response and preserve all probabilities; never synthesize them."""
    wire = api_preview(request)
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
    values = [_number(probabilities[k], 0, 1) for k in keys]
    if not math.isclose(sum(values), 1, abs_tol=1e-5):
        raise ValueError("Response probabilities do not sum to one.")
    features = {f"p_{key}": value for key, value in zip(keys, values)}
    features["confidence"] = _number(answer.get("confidence"), 0, 1)
    if answer["type"] == "choice":
        if answer.get("choice") not in keys:
            raise ValueError("Selected class is outside the allowed answer space.")
        if probabilities[answer["choice"]] < max(values) - 1e-5:
            raise ValueError("Selected class is inconsistent with its probabilities.")
    else:
        if answer.get("legend") != dict(zip(keys, question["criteria"])):
            raise ValueError("Regression legend differs from the requested scale.")
        score = _number(answer.get("score"), 0, 8)
        expected = sum(i * p for i, p in enumerate(values))
        if not math.isclose(score, expected, abs_tol=1e-4):
            raise ValueError("Regression score is inconsistent with its probabilities.")
        # Jev indexes levels 0..8; our agreed directional values are -4..+4.
        features["expected_direction"] = expected - 4
    return features
