"""Explicit connected-host staging of pinned self-hosted weights onto PROJECT.

No download on import. Run only when ready for the selected GPU pilot:
python -m src.jev.checkpoints --provider liquid
"""

import argparse
import json
import os

from src.jev.registry import ADDED, PINS, SELF_HOSTED, asset_urls, stage_tokenizers
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.serialization import digest, file_sha256, write_json


def checkpoint_path(provider):
    return paths.checkpoints_dir("decision_models", paths.output_component(provider))


def checkpoint_files(folder):
    from src.jev.checkpoint_hashes import hashes

    return hashes(folder)


def verify_checkpoint(provider):
    folder = checkpoint_path(provider)
    manifest = json.loads((folder / "_JEVPFN_CHECKPOINT.json").read_text(encoding="utf-8"))
    if (
        manifest["provider"] != provider
        or manifest["revision"] != PINS[provider]
        or checkpoint_files(folder) != manifest["files"]
    ):
        raise ValueError("Checkpoint files differ from their staged immutable identity")
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if file_sha256(folder / name) != file_sha256(paths.tokenizer_dir(provider) / name):
            raise ValueError("Inference checkpoint and audited tokenizer differ")
    if provider in {"strands", "nimble"}:
        candidate = load_yaml("extractors")["candidates"][provider]
        base = paths.checkpoints_dir("decision_models", provider + "_base")
        expected = manifest.get("base")
        if (
            not expected
            or expected["revision"] != candidate["base_model_revision"]
            or checkpoint_files(base) != expected["files"]
        ):
            raise ValueError("Separate base checkpoint differs from the staged identity")
    if provider in ADDED:
        for name in asset_urls(provider):
            if file_sha256(folder / name) != file_sha256(paths.tokenizer_dir(provider) / name):
                raise ValueError("Runtime source differs from the audited provider assets")
    return folder, digest(manifest)


def stage(provider):
    from src.cluster.preflight import storage_policy_failures
    from src.cluster.scheduling import require_cpu_staging

    failures = storage_policy_failures(os.environ.get("SLURM_CLUSTER_NAME"))
    if failures:
        raise ValueError("; ".join(failures))
    require_cpu_staging()
    # HF constants are fixed at first import. Explicit cache_dir alone does not redirect
    # the potentially large Xet download cache away from HOME/personal storage.
    os.environ["HF_HOME"] = str(paths.checkpoints_dir("huggingface"))
    os.environ["HF_HUB_CACHE"] = str(paths.checkpoints_dir("huggingface", "hub"))
    os.environ["HF_XET_CACHE"] = str(paths.checkpoints_dir("huggingface", "xet"))
    from huggingface_hub import snapshot_download

    stage_tokenizers([provider])
    candidate = load_yaml("extractors")["candidates"][provider]
    folder = paths.require_project_output(checkpoint_path(provider))
    cache = paths.checkpoints_dir("huggingface", "hub")
    snapshot_download(
        candidate["model"], revision=PINS[provider], local_dir=folder, cache_dir=cache
    )
    if provider in ADDED:
        import shutil
        for name in asset_urls(provider):
            target = folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            source = paths.tokenizer_dir(provider) / name
            if target.exists() and file_sha256(target) != file_sha256(source):
                raise ValueError("Runtime source/checkpoint conflict; no files were overwritten")
            if not target.exists():
                shutil.copyfile(source, target)
    manifest = {
        "provider": provider,
        "model": candidate["model"],
        "revision": PINS[provider],
        "files": checkpoint_files(folder),
    }
    if provider in {"strands", "nimble"}:
        # Populate the HF cache used by the publisher's base-model ID lookup, and retain
        # a portable directory for verification. Symlinks avoid a second physical copy.
        source = snapshot_download(
            candidate["base_model"], revision=candidate["base_model_revision"], cache_dir=cache
        )
        from pathlib import Path

        base = paths.checkpoints_dir("decision_models", provider + "_base")
        base.mkdir(parents=True, exist_ok=True)
        for file in Path(source).rglob("*"):
            if file.is_file():
                link = base / file.relative_to(source)
                link.parent.mkdir(parents=True, exist_ok=True)
                if not link.exists():
                    link.symlink_to(file.resolve())
        manifest["base"] = {
            "revision": candidate["base_model_revision"],
            "files": checkpoint_files(base),
        }
    destination = folder / "_JEVPFN_CHECKPOINT.json"
    if destination.exists() and json.loads(destination.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Checkpoint manifest changed; preserve the old checkpoint separately")
    write_json(destination, manifest)
    print(f"Staged and fingerprinted {provider}: {folder}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True, choices=SELF_HOSTED)
    args = parser.parse_args()
    stage(args.provider)


if __name__ == "__main__":
    main()
