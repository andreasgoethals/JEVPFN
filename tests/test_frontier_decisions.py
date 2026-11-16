"""Native adapter contracts. These offline tests do not replace actual B200 evidence."""

import json
import types

import pytest

from src.cluster.setup_extractors import plan
from src.jev import catalog
from src.jev import frontier_decisions as native
from src.jev.provider import probability_distribution, semantic_question
from src.jev.registry import source_module
from src.utils import paths
from src.utils.config import load_yaml
from tests.test_glide import payload


@pytest.mark.parametrize("provider", native.MODELS)
def test_all_entry_points_require_separate_native_acceptance(provider):
    cfg = load_yaml("extractors")["candidates"][provider]
    assert (cfg["model"], cfg["revision"]) == native.MODELS[provider]
    assert catalog.PINS[provider] == cfg["revision"]
    assert provider in catalog.SELF_HOSTED and not cfg["enabled"]
    assert cfg["context_tokens"] == native.LIMIT
    assert load_yaml("experiment_0/self_hosted")["accepted_before_extraction_manifests"][provider] is None
    steps = {s["name"]: s for s in plan([provider])}
    assert provider + "_build" in steps[provider + "_tokens_prepare"]["dependencies"]
    assert provider + "_tokens" in steps[provider + "_scratch"]["dependencies"]
    assert provider + "_contracts" in steps[provider + "_pilot"]["dependencies"]
    assert steps[provider + "_contracts"]["cluster"] == "mindwell"
    assert "--cpus-per-task=14" in steps[provider + "_contracts"]["flags"]


def mammoth_counter():
    root, _ = native.vendor_files()
    c = native.Counter.__new__(native.Counter)
    c.provider = "mammoth"
    native.package("test_mammoth", root / "qev")
    c.schema = source_module(root, "qev/schema.py", "test_mammoth.schema")
    c.renderer = source_module(root, "qev/prompt.py", "test_mammoth.prompt")
    c.tokenizer = types.SimpleNamespace(apply_chat_template=lambda messages, **kwargs: json.dumps(messages))
    return c


@pytest.mark.parametrize("kind", ["binary_probability", "choice", "ordered_choice"])
def test_native_mammoth_renderer_preserves_all_fields_and_options(kind):
    c = mammoth_counter()
    p = payload(kind)
    text, specials = c.render_parts(p)[0]
    assert specials is False
    messages = json.loads(text)
    question = semantic_question(p["question"])
    assert "Résumé 世界" in messages[1]["content"]
    assert question["instructions"] in messages[1]["content"]
    if kind == "choice":
        for meaning in question["criteria"].values():
            assert meaning in messages[1]["content"]
    if kind == "ordered_choice":
        assert messages[1]["content"].index("evidence -4") < messages[1]["content"].index("evidence 4")


def test_native_mammoth_rejects_empty_text_and_keeps_yes_first():
    c = mammoth_counter()
    p = payload()
    q = c.schema.SystemOneRequest(state=p["state"], questions={"q": semantic_question(p["question"])})
    assert c.renderer.render_question("q", q.questions["q"]).labels == ["yes", "no"]
    p["state"]["text"]["body"] = ""
    with pytest.raises(ValueError):
        c.render_parts(p)


def test_mammoth_rounding_is_provider_bound_not_a_general_normalization():
    values = [0.033333] * 30
    corrected, evidence = probability_distribution(values, "mammoth")
    assert sum(corrected) == pytest.approx(1, abs=1e-5)
    assert evidence["policy"] == "mammoth-six-decimal-simplex-v1"
    with pytest.raises(ValueError):
        probability_distribution([0.01] * 30, "mammoth")


def test_publisher_patch_and_source_are_hash_verified(tmp_path, monkeypatch):
    root, manifest = native.vendor_files()
    assert {"qev/engine.py", "qev/readout.py", "qev/prompt.py", "serve.patch"} <= manifest["files"].keys()
    assert "block_bidirectional" in (root / "qev/readout.py").read_text()
    assert "source/qev/readout.py" in native.asset_urls("mammoth")
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError):
        native.vendor_files()


@pytest.mark.parametrize("provider", native.MODELS)
def test_runtime_rejects_multiple_questions_and_images_before_model_call(provider):
    runtime = native.Runtime.__new__(native.Runtime)
    runtime.provider = provider
    for request in ({"questions": {"a": {}, "b": {}}}, {"questions": {"a": {}}, "images": ["x"]}):
        with pytest.raises(ValueError):
            runtime.predict(request)


def test_drex_native_preparation_is_rechecked_and_never_truncated():
    r = native.Runtime.__new__(native.Runtime)
    r.provider = "drex_max"
    calls = []
    r.model = types.SimpleNamespace(
        prepare=lambda rows, **kwargs: calls.append((rows, kwargs)) or types.SimpleNamespace(
            inputs={"attention_mask": types.SimpleNamespace(sum=lambda: types.SimpleNamespace(item=lambda: 123))}),
        predict=lambda rows, **kwargs: calls.append((rows, kwargs)) or [[0.2, 0.8]],
    )
    r.decode = lambda q, p: {"type": "noul", "noul": p[1]}
    raw = r.predict({"state": {"body": "abc"}, "questions": {"semantic": {"type": "noul"}}})
    assert raw["answers"]["semantic"]["noul"] == 0.8
    assert raw["usage"]["input_tokens"] == 123
    assert all(k["max_length"] == 16384 for _, k in calls)
    assert calls[1][1]["batch_size"] == 1
