"""Pinned Strands infer._to_answer uses round(value, 4), not full precision.

Synthetic floats exercise that published serialization; no model or API call.
Source: strands-labs/strands-decider@b9fc83626bcca822d00f822d6c2e151b8d244ee9
src/strands_decider/infer.py::_to_answer.
"""

import copy
import math

import pytest

from src.jev.provider import probability_distribution, response_features, score_consistency
from src.jev.requests import IntendedRequest
from src.utils.serialization import canonical_json, digest


@pytest.mark.parametrize("latent", [[1 / 30] * 30, [0.00002] * 8 + [0.99984]])
def test_strands_publisher_rounding_has_bounded_order_independent_repair(latent):
    wire = [round(p, 4) for p in latent]
    before = wire.copy()
    assert not math.isclose(math.fsum(wire), 1, abs_tol=1e-5, rel_tol=0)
    values, diagnostics = probability_distribution(wire, "strands")
    reversed_values, _ = probability_distribution(wire[::-1], "strands")
    assert wire == before
    assert math.fsum(values) == pytest.approx(1, abs=1e-10)
    assert values == pytest.approx(reversed_values[::-1], abs=1e-12)
    assert all(0 <= p <= 1 and abs(p - q) <= 0.00005 + 1e-12 for p, q in zip(values, wire))
    assert diagnostics["policy"] == "strands-four-decimal-simplex-v1"
    assert diagnostics["normalization_applied"]


@pytest.mark.parametrize("values", [[0.4, 0.4], [0, 0, 0], [0.33331] * 3, [0.334] * 3])
def test_strands_rejects_impossible_rounding_and_unexpected_precision(values):
    with pytest.raises(ValueError):
        probability_distribution(values, "strands")


def test_strands_does_not_broaden_other_provider_precision():
    for provider in ("liquid", "perplexity", "cloudflare", "intern", "h2o", "kev"):
        with pytest.raises(ValueError, match="sum to one"):
            probability_distribution([0.0333] * 30, provider)


def test_strands_score_uses_its_four_decimal_envelope():
    latent = [0.00002] * 8 + [0.99984]
    wire = [round(p, 4) for p in latent]
    score = round(math.fsum(i * p for i, p in enumerate(latent)), 4)
    check = score_consistency(score, wire, "strands")
    assert check["rounding_envelope_used"]
    assert check["policy"] == "strands-four-decimal-envelope-v1"
    with pytest.raises(ValueError, match="inconsistent"):
        score_consistency(7.99, wire, "strands")  # Plausible at Jev precision, not Strands.
    with pytest.raises(ValueError, match="inconsistent"):
        score_consistency(score, wire, "intern")


@pytest.mark.parametrize("task", ["choice", "ordered_choice"])
def test_strands_decoder_retains_raw_probability_and_score_fields(task):
    if task == "choice":
        question = {"kind": task, "text": "Which class?", "allowed_answers": list(range(30))}
        keys = [f"c{i:03d}" for i in range(30)]
        answer = {
            "type": "choice", "choice": "c000", "confidence": 0.0333,
            "probabilities": dict.fromkeys(keys, 0.0333),
        }
    else:
        options = [{"score": i, "meaning": f"direction {i}"} for i in range(-4, 5)]
        question = {"kind": task, "text": "Lower or higher?", "allowed_answers": options}
        keys = list(map(str, range(9)))
        answer = {
            "type": "score", "score": 7.9993, "confidence": 0.9998,
            "probabilities": dict(zip(keys, [0.0] * 8 + [0.9998])),
            "legend": {k: o["meaning"] for k, o in zip(keys, options)},
        }
    settings = {
        "provider": "strands", "version": "pinned-test", "model": "test-model",
        "api_mapping_version": "strands-self-hosted-v1",
    }
    request = IntendedRequest(
        canonical_json({"state": {"text": {"column": "Example"}}, "question": question}),
        canonical_json(settings), "{}",
    )
    response = {
        "provider": "strands", "version": "pinned-test", "model": "test-model",
        "settings_sha256": digest(settings), "answers": {"semantic": answer},
    }
    original = copy.deepcopy(response)
    features = response_features(request, response)
    assert response == original
    assert sum(features[f"p_{k}"] for k in keys) == pytest.approx(1)
    if task == "ordered_choice":
        assert features["expected_direction"] == pytest.approx(
            sum((i - 4) * features[f"p_{i}"] for i in range(9))
        )
