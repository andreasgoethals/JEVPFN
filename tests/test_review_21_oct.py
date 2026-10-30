"""Integration regressions from the 21 October independent review."""

import copy

import pytest

from src.evaluation.models import predictor_settings_identity


def test_predictor_identity_excludes_only_output_routing():
    settings = {"seed": 7, "checkpoint": "pinned", "parameters": {"epochs": 2}}
    assert predictor_settings_identity({**settings, "output_phase": "experiment_3"}) == (
        predictor_settings_identity({**settings, "output_phase": "experiment_2"})
    )
    assert predictor_settings_identity(settings) != predictor_settings_identity(
        {**settings, "seed": 8}
    )


def test_tabstar_campaign_pairs_across_output_phases(isolated_output, monkeypatch):
    from src.evaluation import campaign
    from tests.test_frozen_text import fixture_data

    data = fixture_data()
    base = dict(
        dataset_id="fixture",
        model="tabstar",
        provider=None,
        phase="experiment_3",
        feature_set="original_features",
    )
    original_config = campaign.load_yaml
    selector = dict(
        provider="jev",
        model="jev-1.14.0",
        version="1.14.0",
        question_bank_id="bank",
        question_variant="accepted",
    )

    def config(name):
        cfg = copy.deepcopy(original_config(name))
        cfg["enabled"] = True
        cfg["full_fold_profile_enabled"] = True
        if name == "experiment_1/models":
            cfg["models"]["tabstar"]["runtime"]["device"] = "cpu"
        if "feature_selectors" in cfg:
            cfg["feature_selectors"]["jev"] = selector
        return cfg

    monkeypatch.setattr(campaign, "load_yaml", config)
    monkeypatch.setattr(campaign, "reviewed_acceptance", lambda *a, **kw: None)
    monkeypatch.setattr(campaign, "benchmark_plan", lambda **kw: [base])
    monkeypatch.setattr(campaign, "load_catalog", lambda: {"datasets": [{"id": "fixture"}]})
    monkeypatch.setattr(campaign, "load_dataset", lambda *a, **kw: data)
    monkeypatch.setattr(campaign, "shared_splits", lambda *a, **kw: {})
    monkeypatch.setattr(campaign, "load_feature_selection", lambda *a, **kw: [])
    monkeypatch.setattr(campaign, "estimator_factory", lambda *a, **kw: lambda: None)
    matrix = campaign.original_features(data).drop(columns=["a", "b"])
    matrix["p_positive"] = 0.5
    matrix.attrs["feature_artifact_ids"] = []
    monkeypatch.setattr(campaign, "assemble_feature_set", lambda *a, **kw: matrix)
    captured = []

    def evaluate(*a, **kw):
        captured.append(kw)
        return []

    monkeypatch.setattr(campaign, "evaluate_cv", evaluate)
    campaign.run_arm(base, allow_model_runs=True)
    campaign.run_arm(base, allow_model_runs=True, profile=True)
    assert [c["estimator_settings"]["output_phase"] for c in captured] == [
        "experiment_3",
        "experiment_0",
    ]
    assert captured[0]["checkpoint_id"] == captured[1]["checkpoint_id"]


@pytest.mark.parametrize("processor", ["bert", "eos"])
def test_chunk_processor_keeps_special_tokens_and_every_content_id(processor):
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    from src.data.text_embeddings import model_input_chunks, tokenizer_contract

    vocab = {
        word: i
        for i, word in enumerate(
            [
                "[UNK]",
                "[PAD]",
                "[CLS]",
                "[SEP]",
                "<|endoftext|>",
                "query:",
                "x",
                "one",
                "two",
                "three",
                "four",
                "five",
            ]
        )
    }
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]" if processor == "bert" else "$A <|endoftext|>",
        special_tokens=[(key, vocab[key]) for key in ("[CLS]", "[SEP]", "<|endoftext|>")],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        model_input_names=["input_ids", "attention_mask", "token_type_ids"],
    )
    prefix = "query: " if processor == "bert" else ""
    assert tokenizer_contract(tokenizer, prefix=prefix, max_tokens=64)["publisher_input_parity"]
    text = "one two three four five one two three four five"
    chunks = list(model_input_chunks(tokenizer, text, prefix=prefix, max_tokens=6))
    prefix_length = len(tokenizer.encode(prefix, add_special_tokens=False))
    start = 1 if processor == "bert" else 0
    content = []
    for inputs, count in chunks:
        assert len(inputs["input_ids"]) <= 6
        assert inputs["input_ids"][-1] == vocab["[SEP]" if processor == "bert" else "<|endoftext|>"]
        values = inputs["input_ids"][start + prefix_length : -1]
        assert count == len(values)
        content.extend(values)
    assert content == tokenizer.encode(text, add_special_tokens=False)
