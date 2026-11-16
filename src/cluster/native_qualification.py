"""Prepare isolated, non-production model qualification during a frozen campaign.

Run in a CPU allocation with an explicit separate PROJECT root. Immutable downloaded
weights are hardlinked only after checksum verification. No canonical checkpoint or
container markers are changed, and no production extraction is enabled.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

from src.jev.frontier_decisions import MODELS, asset_urls, stage_tokenizer
from src.utils import paths
from src.utils.locking import file_lock
from src.utils.serialization import file_sha256, write_json


def verified_links(source, destination, records):
    """Validate the complete input inventory before adding any immutable hardlinks."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or destination.is_relative_to(source):
        raise ValueError("Qualification destination must be separate from the source")
    for name, record in records.items():
        file = source / name
        if (not name or Path(name).is_absolute() or ".." in Path(name).parts
                or file.is_symlink() or not file.resolve().is_relative_to(source)
                or file_sha256(file) != record["sha256"]
                or file.stat().st_size != record["bytes"]):
            raise ValueError("Downloaded asset differs from its receipt")
        target = destination / name
        if target.exists() and (target.is_symlink() or file_sha256(target) != record["sha256"]):
            raise ValueError("Qualification destination conflicts; nothing overwritten")
    for name in records:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            os.link(source / name, target)  # Same allocated filesystem; no fallback full copy.


def prepare(provider, canonical):
    if provider not in MODELS or not paths.on_vsc() or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Native qualification requires a selected provider and CPU allocation")
    from src.cluster.preflight import storage_policy_failures
    from src.cluster.scheduling import require_cpu_staging
    from src.jev.checkpoints import checkpoint_files

    require_cpu_staging()
    if failures := storage_policy_failures(os.environ.get("SLURM_CLUSTER_NAME")):
        raise ValueError("; ".join(failures))
    canonical = Path(canonical).resolve(strict=True)
    root = paths.checkpoints_dir().parent.resolve()
    if root == canonical or not root.is_relative_to(canonical / "qualification"):
        raise ValueError("Qualification must use an isolated PROJECT qualification subtree")
    source = canonical / "checkpoints/decision_models" / provider
    receipt = json.loads((source / "_JEVPFN_DOWNLOAD.json").read_text())
    if (receipt.get("provider") != provider
            or (receipt.get("model"), receipt.get("revision")) != MODELS[provider]
            or receipt.get("status") != "downloaded_and_hashed_requires_native_0A"):
        raise ValueError("Download receipt is not the reviewed model release")
    target = root / "checkpoints/decision_models" / provider
    with file_lock(root / f".{provider}.qualification.lock", timeout=1):
        verified_links(source, target, receipt["files"])
        stage_tokenizer(provider)
        for name in asset_urls(provider):
            file = paths.tokenizer_dir(provider) / name
            dest = target / name
            if dest.exists() and file_sha256(dest) != file_sha256(file):
                raise ValueError("Audited source conflicts with downloaded runtime")
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(file, dest)
        marker = dict(provider=provider, model=MODELS[provider][0],
                      revision=MODELS[provider][1], files=checkpoint_files(target))
        marker_path = target / "_JEVPFN_CHECKPOINT.json"
        if marker_path.exists() and json.loads(marker_path.read_text()) != marker:
            raise ValueError("Qualification checkpoint identity changed")
        write_json(marker_path, marker)
    return dict(provider=provider, qualification_root=str(root),
                canonical_root_unchanged=True, full_extraction_enabled=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=MODELS, required=True)
    parser.add_argument("--canonical-project", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.provider, args.canonical_project), indent=2))


if __name__ == "__main__":
    main()
