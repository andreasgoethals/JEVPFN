"""CPU staging is a valid Slurm job; GPUs/frozen campaigns must never download."""

import sys
from types import SimpleNamespace

import pytest

from src.cluster import freeze
from src.cluster.scheduling import require_cpu_staging
from src.data import text_embeddings as embeddings


@pytest.fixture
def cpu_job(monkeypatch):
    for key in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_GPUS", "SLURM_GPUS_ON_NODE",
                "SLURM_GPUS_PER_NODE", "SLURM_GPUS_PER_TASK"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "62423298")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "batch")
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "0")
    monkeypatch.setattr(freeze, "check_if_present", lambda: {"frozen": False})


def test_cpu_staging_downloads_once_then_verifies_existing_weights(isolated_output, cpu_job, monkeypatch):
    calls = []

    def download(model, **kwargs):
        calls.append((model, kwargs["revision"]))
        folder = kwargs["local_dir"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "model.safetensors").write_bytes(b"synthetic weights")
        (folder / "config.json").write_text("{}")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    cfg = embeddings.encoder_config()
    first = embeddings.stage(cfg)
    assert calls == [(cfg["model"], cfg["revision"])]
    assert embeddings.stage(cfg) == first
    assert len(calls) == 1
    (embeddings.checkpoint_dir(cfg) / "model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="differs"):
        embeddings.stage(cfg)
    assert len(calls) == 1  # Do not replace a corrupt asset behind its immutable receipt.


@pytest.mark.parametrize("key,value", [
    ("SLURM_JOB_GPUS", "0"), ("SLURM_STEP_GPUS", "0"),
    ("SLURM_JOB_PARTITION", "gpu_b200"), ("SLURM_JOB_PARTITION", "gpu_h100"),
    ("SLURM_GPUS_ON_NODE", "1"), ("SLURM_GPUS", "1"),
    ("SLURM_GPUS_PER_NODE", "b200:1"), ("SLURM_GPUS_PER_TASK", "1"),
])
def test_gpu_staging_rejected_before_download(cpu_job, monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(RuntimeError, match="CPU setup"):
        embeddings.stage(embeddings.encoder_config())


def test_cpu_and_login_allowed_but_frozen_campaign_refuses_staging(cpu_job, monkeypatch):
    require_cpu_staging()
    monkeypatch.delenv("SLURM_JOB_ID")
    require_cpu_staging()
    monkeypatch.setattr(freeze, "check_if_present", lambda: {"frozen": True})
    with pytest.raises(RuntimeError, match="freezing"):
        embeddings.stage(embeddings.encoder_config())
