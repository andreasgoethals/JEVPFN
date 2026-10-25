"""SLURM global GPU IDs may be remapped inside the allocated device cgroup."""

import subprocess
from types import SimpleNamespace

import pytest

from src.cluster import monitor


def setup_query(monkeypatch, inventory, *, allocated="3", visible="0", returncode=6):
    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)
    monkeypatch.setenv("SLURM_JOB_GPUS", allocated)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    commands = []

    def query(command, **kwargs):
        commands.append(command)
        if len(commands) == 1:
            raise subprocess.CalledProcessError(
                returncode, command, output="No devices were found\n"
            )
        if len(commands) == 2:
            assert command[1] == "--query-gpu=index,uuid"
            return SimpleNamespace(stdout=inventory)
        assert command[1:3] == ["-i", "GPU-allocation"]
        return SimpleNamespace(stdout="GPU-allocation, H100, 75, 25, 1000, 80000, 200\n")

    monkeypatch.setattr(monitor.subprocess, "run", query)
    return commands


def test_single_allocation_remap_uses_visible_uuid_not_other_devices(monkeypatch):
    commands = setup_query(monkeypatch, "0, GPU-allocation\n")
    assert monitor.gpu_sample()[0][0] == "GPU-allocation"
    assert len(commands) == 3


@pytest.mark.parametrize("inventory", ["", "0, GPU-a\n1, GPU-b\n", "1, GPU-a\n", "0, bad\n"])
def test_remap_refuses_ambiguous_or_unallocated_inventory(monkeypatch, inventory):
    commands = setup_query(monkeypatch, inventory)
    with pytest.raises(subprocess.CalledProcessError):
        monitor.gpu_sample()
    assert len(commands) == 2


@pytest.mark.parametrize(
    "allocated,visible,code", [("2,3", "0,1", 6), ("3", "3", 6), ("3", "0", 1)]
)
def test_remap_does_not_mask_unrelated_errors_or_guess_multi_gpu_mapping(
    monkeypatch, allocated, visible, code
):
    commands = setup_query(
        monkeypatch, "0, GPU-allocation\n", allocated=allocated, visible=visible, returncode=code
    )
    with pytest.raises(subprocess.CalledProcessError):
        monitor.gpu_sample()
    assert len(commands) == 1
