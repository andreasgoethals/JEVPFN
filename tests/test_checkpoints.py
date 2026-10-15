"""Staging/identity contracts using tiny fake weights, without network or GPU imports."""

import os
import sys
from types import SimpleNamespace

import pytest

from src.jev import checkpoints
from src.jev.tokenization import PINS
from src.utils import paths


def test_staging_redirects_download_caches_and_detects_changed_weights(
    isolated_output, monkeypatch
):
    for name in ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE"):
        monkeypatch.setenv(name, "must-be-replaced")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assets = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    tokenizer = paths.tokenizer_dir("liquid")
    tokenizer.mkdir(parents=True)
    for name in assets:
        (tokenizer / name).write_text("tiny fixture", encoding="utf-8")
    monkeypatch.setattr(checkpoints, "stage_tokenizers", lambda providers: None)
    calls = []

    def download(repo, *, revision, local_dir, cache_dir):
        calls.append(repo)
        assert revision == PINS["liquid"]
        assert cache_dir == paths.checkpoints_dir("huggingface", "hub")
        assert os.environ["HF_HOME"] == str(paths.checkpoints_dir("huggingface"))
        assert os.environ["HF_HUB_CACHE"] == str(cache_dir)
        assert os.environ["HF_XET_CACHE"] == str(paths.checkpoints_dir("huggingface", "xet"))
        local_dir.mkdir(parents=True)
        for name in assets:
            (local_dir / name).write_bytes((tokenizer / name).read_bytes())
        (local_dir / "model.safetensors").write_bytes(b"synthetic weights")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    checkpoints.stage("liquid")
    folder, identity = checkpoints.verify_checkpoint("liquid")
    assert len(calls) == 1 and len(identity) == 64
    (folder / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="immutable identity"):
        checkpoints.verify_checkpoint("liquid")


def test_publisher_import_does_not_mutate_pinned_checkpoint(isolated_output, monkeypatch):
    from src.jev.open_runtimes import _module
    from src.jev.self_hosted import inference_environment

    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    folder = paths.checkpoints_dir("decision_models", "intern")
    tokenizer = paths.tokenizer_dir("intern")
    for directory in (folder, tokenizer):
        directory.mkdir(parents=True)
        (directory / "inference.py").write_text("ANSWER = 42\n", encoding="utf-8")
    before = checkpoints.checkpoint_files(folder)
    inference_environment()
    module = _module("intern", folder, "inference.py")
    assert module.ANSWER == 42
    assert not (folder / "__pycache__").exists()
    assert checkpoints.checkpoint_files(folder) == before
