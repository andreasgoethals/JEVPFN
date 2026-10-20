from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.jev.provider import QUESTION_ID, response_features
from src.jev.requests import IntendedRequest
from src.jev.self_hosted import SelfHostedClient
from src.utils.serialization import canonical_json


@pytest.mark.parametrize("provider", ["cloudflare", "perplexity"])
def test_native_batch_preserves_per_input_probabilities_and_counts(provider):
    cfg = {
        "provider": provider,
        "model": "fixed",
        "version": "fixed",
        "api_mapping_version": f"{provider}-self-hosted-v1",
        "request_configuration": {"batch_size": 2},
    }
    requests = [
        IntendedRequest(
            canonical_json(
                {
                    "state": {"text": {"t": text}},
                    "question": {"kind": "binary_probability", "text": "Supported?"},
                }
            ),
            canonical_json(cfg),
            canonical_json({"dataset_id": "fixture"}),
        )
        for text in ("first", "different second text")
    ]
    client = object.__new__(SelfHostedClient)
    client.settings, client.provider = cfg, provider
    client.counter = Mock(return_value=12)
    client.counter.limit = 20
    client.torch = SimpleNamespace(
        inference_mode=nullcontext, cuda=SimpleNamespace(synchronize=lambda: None)
    )
    client.model = Mock()
    client.model.predict_many.return_value = [
        {
            "answers": {QUESTION_ID: {"type": "noul", "noul": p}},
            "usage": {"input_tokens": 12, "output_tokens": 0},
        }
        for p in (0.2, 0.8)
    ]
    client.model.prepare.return_value = SimpleNamespace(input_tokens=24)
    client.model.predict.return_value = [[0.8, 0.2], [0.2, 0.8]]
    client.decode = lambda q, v: {"type": "noul", "noul": v[1]}
    results = client.evaluate_many(requests)
    assert [response_features(r, raw)["p_positive"] for r, raw in zip(requests, results)] == [
        0.2,
        0.8,
    ]
    assert [r["usage"]["input_tokens"] for r in results] == [12, 12]
    client.counter.return_value = 21
    client.model.reset_mock()
    with pytest.raises(ValueError, match="overflow"):
        client.evaluate_many(requests)
    client.model.predict.assert_not_called()
    client.model.predict_many.assert_not_called()
