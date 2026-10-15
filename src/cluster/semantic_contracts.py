"""Known-answer diagnostics, separate from deterministic decoder unit tests.

A failed live probe blocks acceptance but is not proof of a decoder bug: a model
can itself misunderstand the prompt. Keep its full response for that distinction.
"""

import json
from dataclasses import replace

from src.jev.provider import response_features
from src.jev.requests import IntendedRequest
from src.jev.self_hosted import cached_evaluate
from src.utils.serialization import canonical_json


def probes(settings):
    base = IntendedRequest(
        "{}", canonical_json(settings), canonical_json({"dataset_id": "EXP0_SEMANTICS"})
    )
    for answer in (True, False):
        yield (
            f"binary_{answer}",
            replace(
                base,
                payload_json=canonical_json(
                    {
                        "state": {"text": {"body": f"The light is {'on' if answer else 'off'}."}},
                        "question": {"kind": "binary_probability", "text": "Is the light on?"},
                    }
                ),
            ),
            float(answer),
        )
    for order in (("red", "green", "blue"), ("blue", "red", "green")):
        yield (
            "choice_" + "_".join(order),
            replace(
                base,
                payload_json=canonical_json(
                    {
                        "state": {"text": {"body": "The named colour is green."}},
                        "question": {
                            "kind": "choice",
                            "text": "Which colour is explicitly named in the text?",
                            "allowed_answers": list(order),
                        },
                    }
                ),
            ),
            order.index("green"),
        )
    for number in (-4, 0, 4):
        yield (
            f"score_{number}",
            replace(
                base,
                payload_json=canonical_json(
                    {
                        "state": {"text": {"body": f"The stated integer is {number}."}},
                        "question": {
                            "kind": "ordered_choice",
                            "text": "Which integer is explicitly stated in the text?",
                            "allowed_answers": [
                                {"score": i, "meaning": f"The stated integer is {i}."}
                                for i in range(-4, 5)
                            ],
                        },
                    }
                ),
            ),
            number,
        )


def exercise_semantics(client, cache):
    records = []
    for name, request, expected in probes(client.settings):
        cached_evaluate(cache, client, request)
        raw = cache.get(request, kind="verified_response")
        features = response_features(request, raw)
        kind = json.loads(request.payload_json)["question"]["kind"]
        if kind == "binary_probability":
            actual = features["p_positive"]
            passed = (actual > 0.5) if expected else (actual < 0.5)
        else:
            probabilities = {k: v for k, v in features.items() if k.startswith("p_")}
            # Inspect the unique maximum; ties do not establish correct label mapping.
            best = [k for k, v in probabilities.items() if v == max(probabilities.values())]
            key = f"p_c{expected:03d}" if kind == "choice" else f"p_{expected + 4}"
            passed = best == [key]
        records.append(
            {
                "probe": name,
                "expected": expected,
                "passed": passed,
                "features": features,
                "request_key": request.cache_key,
            }
        )
    return {
        "passed": all(r["passed"] for r in records),
        "probes": records,
        "interpretation": "Failure requires inspecting raw answers and label mapping; it is not automatically a decoder defect.",
    }
