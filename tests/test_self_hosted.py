"""Synthetic provider contracts. Never imports a model runtime or downloads weights."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from src.data.loaders import LoadedDataset
from src.data.metadata import extract_metadata
from src.jev.cache import ResponseCache
from src.jev.features import load_block, materialize_cached_block, save_block
from src.jev.provider import QUESTION_ID, response_features
from src.jev.requests import IntendedRequest, build_requests
from src.jev.self_hosted import SelfHostedClient, cached_evaluate, run
from src.utils.config import load_config
from src.utils.serialization import canonical_json, digest


def settings(provider):
    return {
        "provider": provider,
        "model": "pinned-model",
        "version": "pinned-revision",
        "api_mapping_version": f"{provider}-self-hosted-v1",
        "request_configuration": {},
        "template_version": "test",
    }


@pytest.mark.parametrize("provider", ["liquid", "strands", "perplexity"])
@pytest.mark.parametrize("kind", ["binary_probability", "choice", "ordered_choice"])
def test_all_three_provider_transports_preserve_task_vectors(provider, kind):
    question = {"kind": kind, "text": "Evaluate the supplied text."}
    if kind == "choice":
        question["allowed_answers"] = ["a", "b", "c"]
        answer = {
            "type": "choice",
            "choice": "c002",
            "probabilities": {"c000": 0.2, "c001": 0.3, "c002": 0.5},
            "confidence": 0.5,
        }
    elif kind == "ordered_choice":
        question["allowed_answers"] = [{"score": i, "meaning": str(i)} for i in range(-4, 5)]
        answer = {
            "type": "score",
            "score": 4,
            "confidence": 0,
            "probabilities": {str(i): 1 / 9 for i in range(9)},
            "legend": {str(i): str(i - 4) for i in range(9)},
        }
    else:
        answer = {"type": "noul", "noul": 0.7}
    request = IntendedRequest(
        canonical_json({"state": {"text": {"t": "body"}}, "question": question}),
        canonical_json(settings(provider)),
        canonical_json({"dataset_id": "fixture"}),
    )
    client = object.__new__(SelfHostedClient)
    client.settings, client.provider = settings(provider), provider
    client.counter = Mock(return_value=12)
    client.counter.limit = 20
    client.torch = SimpleNamespace(
        inference_mode=nullcontext, cuda=SimpleNamespace(synchronize=lambda: None)
    )
    raw = {"answers": {QUESTION_ID: answer}, "usage": {"input_tokens": 12, "output_tokens": 0}}
    client.model = Mock()
    client.model.system_one.return_value = raw
    client.model.evaluate.return_value = SimpleNamespace(model_dump=lambda **kwargs: raw)
    client.request_class = SimpleNamespace
    client.model.prepare.return_value = SimpleNamespace(input_tokens=12)
    client.model.predict.return_value = [[0.3, 0.7]]
    client.decode = lambda *args: answer
    response = client.evaluate(request)
    features = response_features(request, response)
    assert response["publisher_response"]["answers"][QUESTION_ID] == answer
    assert features.get("p_positive", 0.7) == 0.7
    assert features.get("expected_direction", 0) == pytest.approx(0)
    client.counter.return_value = 21
    with pytest.raises(ValueError, match="overflow"):
        client.evaluate(request)


def test_disabled_model_run_never_imports_or_loads_weights():
    with pytest.raises(RuntimeError, match="explicit execution"):
        SelfHostedClient("liquid")
    with pytest.raises(RuntimeError, match="disabled"):
        run("liquid")


@pytest.mark.parametrize("provider", ["liquid", "strands", "perplexity"])
def test_cache_resume_and_feature_materialization_for_every_provider(isolated_output, provider):
    frame = pd.DataFrame({"text": ["same", "same", None], "y": [0, 1, 0]})
    meta = extract_metadata(
        frame,
        {"task_type": "cls", "target": "y"},
        dataset_id="fixture",
        dataset_name="Fixture",
        text_columns=["text"],
    )
    dataset = LoadedDataset(frame, meta, {}, {"data_identity": "fixed"}, [])
    cfg = load_config("feature_creation")
    cfg["jev"]["representation"] = "per_column"
    cfg_settings = settings(provider)
    cache = ResponseCache.for_dataset("fixture")
    client = Mock()

    def evaluate(request):
        return {
            "provider": provider,
            "model": cfg_settings["model"],
            "version": cfg_settings["version"],
            "settings_sha256": digest(cfg_settings),
            "answers": {QUESTION_ID: {"type": "noul", "noul": 0.7}},
        }

    client.evaluate.side_effect = evaluate
    for row in range(3):
        for req in build_requests(dataset.features, meta, row, cfg):
            request = IntendedRequest(
                req.payload_json, canonical_json(cfg_settings), req.provenance_json
            )
            cached_evaluate(cache, client, request)
    assert client.evaluate.call_count == 1
    reopened = ResponseCache.for_dataset("fixture")
    block = materialize_cached_block(dataset, cfg, reopened, provider_settings=cfg_settings)
    file = save_block(block)
    loaded = load_block(file, block.manifest["artifact_id"])
    assert loaded.manifest["identity"]["provider"] == provider
    assert loaded.values.iloc[0, 1] == 0.7 and loaded.values.iloc[1, 1] == 0.7
    assert pd.isna(loaded.values.iloc[2, 1])
    assert loaded.requests.iloc[2].status == "missing_text"
    assert file.name == "fixture.sqlite3"
    bad = evaluate(request)
    bad["version"] = "wrong"
    with pytest.raises(ValueError, match="pinned"):
        response_features(request, bad)
