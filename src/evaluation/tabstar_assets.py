"""Offline TabSTAR v1.1 assets: separate task checkpoints and their E5 dependency.

The publisher resolves repository names through the Hugging Face cache, including
a warm-cache check in fit(). Preserve that interface; stage pinned snapshots and
checked main aliases on CPU rather than passing a local directory as a repo ID.
"""

import json
import os
from pathlib import Path

from src.evaluation.checkpoint_paths import checkpoint_path
from src.utils import paths
from src.utils.serialization import digest, file_sha256

MARKER = "_JEVPFN_PREDICTOR.json"


def manifest_path():
    return paths.checkpoints_dir("predictors", "tabstar", MARKER)


def asset_manifest(settings, *, root=None):
    root = Path(root) if root is not None else paths.checkpoints_dir()
    body = json.loads((root / "predictors" / "tabstar" / MARKER).read_text(encoding="utf-8"))
    identity = {k: v for k, v in body.items() if k not in {"manifest_id", "created_utc"}}
    if (body.get("schema") != "jevpfn.tabstar-assets.v1"
            or body.get("manifest_id") != digest(identity)
            or body.get("repositories") != settings["repositories"]
            or not body.get("assets")):
        raise ValueError("TabSTAR assets differ from the configured v1.1 repositories")
    for pin in body["assets"]:
        relative = Path(pin["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("TabSTAR asset escapes the checkpoint tree")
    return body


def verified_settings(settings, task):
    """Resolve all assets locally before importing TabSTAR; never fall back online."""
    body = asset_manifest(settings)
    for pin in body["assets"]:
        asset = checkpoint_path(pin["path"])
        if not asset.is_file() or file_sha256(asset) != pin["sha256"]:
            raise ValueError("Staged TabSTAR asset is missing or its checksum differs")
    expected = paths.checkpoints_dir("huggingface", "hub").resolve()
    if os.environ.get("HF_HUB_CACHE") and Path(os.environ["HF_HUB_CACHE"]).resolve() != expected:
        raise ValueError("TabSTAR needs the active native checkpoint cache, not an external HF cache")
    os.environ.update(HF_HOME=str(expected.parent), HF_HUB_CACHE=str(expected),
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    kind = "regression" if task == "regression" else "classification"
    return body["repositories"][kind]["model"], body["assets"]
