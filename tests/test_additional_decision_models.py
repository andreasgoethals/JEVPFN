"""Publisher tokenization parity and fail-closed contexts, no inference/network."""

import sys

import pytest

from src.jev.decision_tokenization import SELF_HOSTED, ExactCounter
from src.jev.open_runtimes import ClefRuntime, probability_answer
from src.jev.provider import QUESTION_ID, semantic_question
from src.utils import paths
from src.utils.config import load_yaml


def test_owner_selected_roster_and_delivery():
    candidates = load_yaml("extractors")["candidates"]
    assert set(candidates) == {
        "jev",
        "liquid",
        "strands",
        "perplexity",
        "cloudflare",
        "intern",
    }
    assert set(SELF_HOSTED) == {k for k, v in candidates.items() if v["delivery"] == "self_hosted"}
    assert all(not v["enabled"] for v in candidates.values())


@pytest.mark.parametrize("provider", ["cloudflare", "intern"])
@pytest.mark.parametrize("kind", ["binary_probability", "choice", "ordered_choice"])
def test_actual_publisher_renderer_parity_when_staged(provider, kind):
    pytest.importorskip("transformers")
    if not (paths.tokenizer_dir(provider) / "manifest.json").exists():
        pytest.skip("Stage pinned tokenizer/source files for the full Exp0 parity test")
    question = {"kind": kind, "text": "What does this text indicate?"}
    if kind == "choice":
        question["allowed_answers"] = ["first", "second", "third"]
    if kind == "ordered_choice":
        question["allowed_answers"] = [{"score": i, "meaning": str(i)} for i in range(-4, 5)]
    counter = ExactCounter(provider)
    payload = {
        "state": {"text": {"a": "Résumé, 世界!", "b": "Another field"}},
        "question": question,
    }
    row = {"state": payload["state"], "questions": {QUESTION_ID: semantic_question(question)}}
    if provider == "cloudflare":
        expected = len(
            counter.renderer["encode_record"](
                counter.tokenizer, row, max_length=sys.maxsize
            ).input_ids
        )
    else:
        compiled = counter.renderer["compile_row"](row)
        text = counter.tokenizer.apply_chat_template(
            compiled.messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
            add_vision_id=True,
        )
        expected = len(counter.tokenizer(text, add_special_tokens=False).input_ids)
    assert counter.count_many([payload, payload]) == [expected, expected]
    assert counter.limit == (16384 if provider == "cloudflare" else 8192)
    with pytest.raises(ValueError, match="Empty text"):
        counter({**payload, "state": {"text": {"a": " "}}})


def test_clef_keeps_full_precision():
    p = {f"c{i:03d}": 1 / 30 for i in range(30)}
    answer = probability_answer({"type": "choice"}, p)
    assert sum(answer["probabilities"].values()) == pytest.approx(1)
    assert answer["probabilities"]["c000"] == 1 / 30
    score = probability_answer(
        {"type": "score", "criteria": list(map(str, range(9)))}, {str(i): 1 / 9 for i in range(9)}
    )
    assert score["score"] - 4 == pytest.approx(0)


def test_clef_rejects_before_forward():
    from types import SimpleNamespace
    from unittest.mock import Mock

    runtime = object.__new__(ClefRuntime)
    runtime.limit = 10
    runtime.processor = SimpleNamespace(tokenizer=None)
    runtime.publisher = SimpleNamespace(
        encode_record=Mock(return_value=SimpleNamespace(input_ids=range(11)))
    )
    runtime.model = Mock()
    with pytest.raises(ValueError, match="overflow"):
        runtime.predict({})
    runtime.model.assert_not_called()


@pytest.mark.parametrize("publisher_temperature", [1.99241824, 1.0])
def test_intern_preserves_published_calibration(monkeypatch, tmp_path, publisher_temperature):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from src.jev import open_runtimes

    engine = Mock()
    publisher = SimpleNamespace(DEFAULT_TEMPERATURE=publisher_temperature, DecisionEngine=engine)
    monkeypatch.setattr(open_runtimes, "_module", lambda *args: publisher)
    if publisher_temperature != 1.99241824:
        with pytest.raises(ValueError, match="calibration"):
            open_runtimes.load_runtime("intern", tmp_path, 8192)
        engine.assert_not_called()
    else:
        open_runtimes.load_runtime("intern", tmp_path, 8192)
        engine.assert_called_once_with(
            checkpoint=str(tmp_path),
            device="cuda",
            max_length=8192,
            temperature=publisher_temperature,
            backend="hf",
        )
