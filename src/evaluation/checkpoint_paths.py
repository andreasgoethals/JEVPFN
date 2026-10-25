"""Resolve reviewed predictor assets on the filesystem used by the current job.

Configurations may name durable PROJECT paths. Mindwell executes the CPU-staged
copy under native GPFS instead; a configuration path never authorizes bulk reads
from a different filesystem or an unreviewed external asset tree.
"""

import os
from pathlib import Path

from src.utils import paths


def checkpoint_path(value):
    """Resolve relative pins, rebasing only approved durable pins on Mindwell."""
    source = Path(value)
    active = paths.checkpoints_dir()
    if os.environ.get("SLURM_CLUSTER_NAME") != "mindwell":
        return source if source.is_absolute() else active / source

    active = active.resolve()
    durable = (
        Path(os.environ.get("JEVPFN_DURABLE_ROOT", paths.STAGING_FALLBACK))
        / "JEVPFN"
        / "checkpoints"
    ).resolve()
    if active.is_relative_to(durable):
        raise ValueError("Mindwell predictor assets need their native staged checkpoint tree")
    if source.is_absolute():
        source = source.resolve()
        if source.is_relative_to(active):
            result = source
        elif source.is_relative_to(durable):
            result = active / source.relative_to(durable)
        else:
            raise ValueError("Predictor asset is outside the approved native/durable checkpoints")
    else:
        result = active / source
    result = result.resolve()
    if not result.is_relative_to(active):
        raise ValueError("Predictor asset escapes the native staged checkpoint tree")
    return result


def model_checkpoint_argument(value):
    """Keep existing local publisher arguments, but require a local asset on Mindwell."""
    if os.environ.get("SLURM_CLUSTER_NAME") != "mindwell":
        return value
    result = checkpoint_path(value)
    if not result.exists():
        raise ValueError("Native predictor checkpoint is missing; stage it on CPU first")
    return str(result)
