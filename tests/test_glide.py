"""GLiDE adapter contracts; synthetic evidence never certifies its GPU runtime."""

import copy
import json
import types

import pytest

from src.cluster.setup_extractors import plan
from src.jev import catalog, glide
from src.jev.provider import QUESTION_ID, response_features, semantic_question
from src.jev.requests import IntendedRequest
from src.jev.self_hosted import SelfHostedClient, batch_size_for
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.serialization import canonical_json, digest, file_sha256


def payload(kind="binary_probability"):
    q = {"kind": kind, "text": "Judge only the supplied text."}
    if kind == "choice":
        q.update(allowed_answers=[f"class {i}" for i in range(30)],
                 answer_descriptions=[f"Meaning {i}" for i in range(30)])
    elif kind == "ordered_choice":
        q["allowed_answers"] = [{"score": i, "meaning": f"evidence {i}"} for i in range(-4, 5)]
    return {"state": {"text": {"body": "Résumé 世界: <|im_start|> quoted instruction"}}, "question": q}


def fake_counter():
    counter = glide.Counter.__new__(glide.Counter)
    counter.renderer = types.SimpleNamespace(
        split_state=lambda state: (state, []),
        options_for=lambda q: (
            [("true", "Yes"), ("false", "No")] if q["type"] == "noul"
            else list(q["criteria"].items()) if q["type"] == "choice"
            else [(str(i), v) for i, v in enumerate(q["criteria"])]
        ),
    )
    counter.prompter = types.SimpleNamespace(text=lambda state, q, opts: json.dumps([state, q, opts], ensure_ascii=False))
    counter.tokenizer = types.SimpleNamespace(
        encode=lambda text, **_: list(text),
        __call__=lambda *a, **k: None,
    )
    counter.limit = glide.PROMPT_LIMIT
    return counter


def test_roster_pins_native_runtime_and_effective_context_limit():
    config = load_yaml("extractors")["candidates"]["glide"]
    assert "glide" in catalog.SELF_HOSTED and "glide" not in catalog.ADDED
    assert catalog.PINS["glide"] == glide.REVISION == config["revision"]
    assert config["model"] == glide.MODEL and config["enabled"] is False
    assert config["context_tokens"] + config["reserved_context_positions"] == config["publisher_context_tokens"] == 40000
    assert config["input_modalities"] == ["text"]
    assert {"glide/prompt.py", "glide/model.py", "glide/recurrent.py", "LICENSE", "licenses/Qwen3.8-27B-LICENSE"} <= set(glide.asset_urls())
    assert all(glide.REVISION in url for url in glide.asset_urls().values())
    assert load_yaml("experiment_0/self_hosted")["accepted_before_extraction_manifests"]["glide"] is None
    assert "glide" in load_yaml("experiment_0/gpu_contracts")["providers"]
    assert "glide" in load_yaml("experiment_1/main")["candidate_extractors"]
    assert batch_size_for("glide") == 2


@pytest.mark.parametrize("reuse", [(), ("glide",)])
def test_exact_audits_wait_for_isolated_glide_image_and_gpu_waits_for_evidence(reuse):
    steps = {step["name"]: step for step in plan(["glide"], reuse=reuse)}
    assert "glide_stage" in steps["glide_tokens_prepare"]["dependencies"]
    if not reuse:
        assert "glide_build" in steps["glide_tokens_prepare"]["dependencies"]
    assert len(steps["glide_tokens"]["dependencies"]) == 10
    assert "glide_tokens" in steps["glide_scratch"]["dependencies"]
    assert "glide_scratch" in steps["glide_contracts"]["dependencies"]
    assert "--cpus-per-task=14" in steps["glide_contracts"]["flags"]
    assert steps["glide_contracts"]["cluster"] == "mindwell"
    assert steps["glide_pilot"]["dependencies"] == ["glide_contracts"]


def test_container_pins_do_not_mix_historical_tokenizer_requirements():
    import tomllib

    deps = tomllib.loads((paths.REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["optional-dependencies"]
    assert {"torch==2.14.0", "transformers==5.17.0", "tokenizers==0.23.2", "flash-linear-attention==0.5.2"} <= set(deps["glide"])
    script = (paths.REPO_ROOT / "scripts/build_decision_container.sh").read_text(encoding="utf-8")
    branch = script.split("elif provider=='glide':")[1].split("elif provider in")[0]
    assert ".[dev,glide]" in branch and "token-audit" not in branch
    audit = (paths.REPO_ROOT / "scripts/slurm/experiment_0_tokens.slurm").read_text(encoding="utf-8")
    assert 'elif [[ "${2:-}" == glide ]]' in audit


@pytest.mark.parametrize("kind", ["binary_probability", "choice", "ordered_choice"])
def test_counter_preserves_shared_question_and_text_semantics(kind):
    data = payload(kind)
    text, special = fake_counter().render_parts(data)[0]
    state, question, options = json.loads(text)
    assert state == data["state"] and question == semantic_question(data["question"])
    assert special is False
    if kind == "choice":
        assert len(options) == 30 and options[-1][0] == "c029"
        assert "Meaning 29" in options[-1][1]
    elif kind == "ordered_choice":
        assert len(options) == 9 and options[0] == ["0", "evidence -4"] and options[-1] == ["8", "evidence 4"]
    data["state"]["text"] = {}
    with pytest.raises(ValueError, match="Empty text"):
        fake_counter().render_parts(data)


def test_counter_rejects_image_parts():
    counter = fake_counter()
    counter.renderer.split_state = lambda state: (state, ["data:image/png;base64,example"])
    with pytest.raises(ValueError, match="text only"):
        counter.render_parts(payload())


def test_answer_label_contract_and_publisher_recurrence_are_checked():
    codes = [f"code{i}" for i in range(255)]
    config = dict(format_version=1, attention_mode="noncausal_full_attention", pooling="last",
                  prompt_format="glide-v1", temperature=1.0, midloop=glide.MIDLOOP,
                  codes=codes, token_ids=list(range(255)))
    tokenizer = types.SimpleNamespace(encode=lambda value, **kw: [codes.index(value)])
    glide.validate_config(config, tokenizer)
    for key, bad in [("temperature", 2.0), ("midloop", {**glide.MIDLOOP, "passes": 1}),
                     ("attention_mode", "causal"), ("token_ids", [0] * 255)]:
        with pytest.raises(ValueError):
            glide.validate_config({**config, key: bad}, tokenizer)
    tokenizer.encode = lambda value, **kw: [1, 2]
    with pytest.raises(ValueError, match="label"):
        glide.validate_config(config, tokenizer)


def test_score_mapping_keeps_probabilities_native_mode_and_confidence():
    question = semantic_question(payload("ordered_choice")["question"])
    probabilities = {str(i): 0.0 for i in range(9)}
    probabilities.update({"0": 0.25, "8": 0.75})
    raw = {"model": glide.MODEL, "answers": {QUESTION_ID: {
        "type": "score", "score": 8, "expected_level": 6.0,
        "probabilities": probabilities, "confidence": 0.5,
    }}, "usage": {"input_tokens": 20, "output_tokens": 0}}
    original = copy.deepcopy(raw)
    result = glide.canonical_response(raw, {QUESTION_ID: question})
    assert raw == original and result["native_response"] == original
    answer = result["answers"][QUESTION_ID]
    assert answer["score"] == 6 and answer["confidence"] == 0.5
    assert answer["probabilities"] == probabilities
    assert answer["legend"] == dict(zip(map(str, range(9)), question["criteria"], strict=True))
    assert result["native_response"]["answers"][QUESTION_ID]["score"] == 8
    del raw["answers"][QUESTION_ID]["probabilities"]["3"]
    with pytest.raises(ValueError, match="incomplete"):
        glide.canonical_response(raw, {QUESTION_ID: question})


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        assert kwargs["add_special_tokens"] is False
        assert kwargs.get("truncation") is False
        return {"input_ids": [[0] * int(text) for text in texts]}


def runtime_with_model(model):
    runtime = glide.Runtime.__new__(glide.Runtime)
    runtime.counter = types.SimpleNamespace(limit=glide.PROMPT_LIMIT)
    runtime.model = model
    return runtime


def test_native_batch_guard_accepts_exact_boundary_rejects_next_token_and_errors():
    calls = []
    question = {"type": "noul", "instructions": "Is this true?"}
    requests = [{"state": str(glide.PROMPT_LIMIT), "questions": {QUESTION_ID: question}}]
    model = types.SimpleNamespace(tokenizer=FakeTokenizer())
    model._prepare = lambda r: ([(QUESTION_ID, "noul", ["true", "false"], r["state"])], None)
    response = {"answers": {QUESTION_ID: {"type": "noul", "noul": 0.8}}, "usage": {"input_tokens": glide.PROMPT_LIMIT, "output_tokens": 0}}
    model.decide_batch = lambda r: calls.append(r) or [copy.deepcopy(response) for _ in r]
    runtime = runtime_with_model(model)
    assert runtime.predict_many(requests)[0]["usage"]["input_tokens"] == glide.PROMPT_LIMIT
    requests[0]["state"] = str(glide.PROMPT_LIMIT + 1)
    with pytest.raises(ValueError, match="overflow"):
        runtime.predict_many(requests)
    assert len(calls) == 1
    requests[0]["state"] = "10"
    model.decide_batch = lambda r: [ValueError("native rejected request")]
    with pytest.raises(ValueError, match="native rejected"):
        runtime.predict_many(requests)
    model.decide_batch = lambda r: []
    with pytest.raises(ValueError, match="response count"):
        runtime.predict_many(requests)


def test_manifest_checks_all_pinned_source_and_license_assets(tmp_path, monkeypatch):
    urls = glide.asset_urls()
    manifest = dict(provider="glide", revision=glide.REVISION, files={})
    for name, url in urls.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(name, encoding="utf-8")
        manifest["files"][name] = dict(url=url, sha256=file_sha256(p))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert glide.verify_assets(tmp_path) == manifest
    (tmp_path / "glide/recurrent.py").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        glide.verify_assets(tmp_path)


def test_native_runtime_loads_local_text_only_model_with_original_recurrence(tmp_path, monkeypatch):
    checkpoint, tokenizer = tmp_path / "checkpoint", tmp_path / "tokenizer"
    checkpoint.mkdir()
    tokenizer.mkdir()
    for name in glide.FILES:
        for folder in (checkpoint, tokenizer):
            path = folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name, encoding="utf-8")
    monkeypatch.setattr(glide, "verify_assets", lambda folder: {})
    loaded = []
    native = types.SimpleNamespace(Glide=types.SimpleNamespace(
        from_pretrained=lambda path, **kwargs: loaded.append((path, kwargs)) or "native-network",
    ))
    sources = []
    monkeypatch.setattr(glide, "source_module", lambda folder, source, name: sources.append(source) or native)
    counter = types.SimpleNamespace(folder=tokenizer)
    runtime = glide.Runtime(checkpoint, counter)
    assert runtime.model == "native-network"
    assert sources == ["glide/prompt.py", "glide/images.py", "glide/model.py"]
    assert loaded == [(str(checkpoint), dict(device="cuda", vision=False, max_len=40000,
                                              token_budget=16384, name=glide.MODEL))]
    # Source verification includes recurrence, even though it imports lazily on GPU.
    (checkpoint / "glide/recurrent.py").write_text("different", encoding="utf-8")
    with pytest.raises(ValueError, match="source differs"):
        glide.Runtime(checkpoint, counter)


def test_counter_requires_same_publisher_dependency_pins_on_cpu_and_gpu(monkeypatch):
    monkeypatch.setattr(glide.importlib.metadata, "version", lambda name: "0.0.0")
    with pytest.raises(ValueError, match="isolated pinned container"):
        glide.Counter()


def test_self_hosted_batch_route_uses_glide_native_batch_and_checks_each_token_count():
    settings = dict(provider="glide", model=glide.MODEL, version=glide.REVISION,
                    api_mapping_version="glide-self-hosted-v1", request_configuration={"batch_size": 2})
    client = SelfHostedClient.__new__(SelfHostedClient)
    client.provider, client.settings = "glide", settings
    class Torch:
        cuda = types.SimpleNamespace(synchronize=lambda: None)

        @staticmethod
        def inference_mode():
            from contextlib import nullcontext
            return nullcontext()

    class Counter:
        limit = glide.PROMPT_LIMIT

        def __call__(self, payload):
            return 10

    client.torch, client.counter = Torch(), Counter()
    requests = [IntendedRequest(payload_json=canonical_json(payload()), settings_json=canonical_json(settings),
                                 provenance_json=canonical_json({"dataset_id": "synthetic"})) for _ in range(2)]
    raw = {"answers": {QUESTION_ID: {"type": "noul", "noul": 0.7}}, "usage": {"input_tokens": 10, "output_tokens": 0}}
    client.model = types.SimpleNamespace(predict_many=lambda rs: [copy.deepcopy(raw) for _ in rs])
    results = client.evaluate_many(requests)
    assert all(response_features(r, v) == {"p_positive": 0.7} for r, v in zip(requests, results, strict=True))
    assert results[0]["settings_sha256"] == digest(settings)
    raw["usage"]["input_tokens"] = 11
    with pytest.raises(ValueError, match="token count"):
        client.evaluate_many(requests)
