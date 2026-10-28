"""Explicit CPU staging of the reviewed TabSTAR v1.1 snapshots; no model execution."""

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from src.cluster.freeze import check_if_present
from src.evaluation.tabstar_assets import asset_manifest, manifest_path
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.files import atomic_text
from src.utils.locking import file_lock
from src.utils.serialization import digest, file_sha256, write_json


def _download_snapshot(download, *, repo_id, revision, cache_dir):
    """Retry interrupted public downloads using the same pinned cache, at most 3 times."""
    for attempt in range(1, 4):
        try:
            return download(repo_id=repo_id, revision=revision, cache_dir=cache_dir)
        except httpx.TransportError as exc:
            # Do not retry HTTP status, configuration or integrity errors. Reusing
            # the Hub cache resumes valid partial downloads without force_download.
            if attempt == 3:
                raise
            delay = 5 * attempt
            print(json.dumps({"event": "predictor_download_retry", "repository": repo_id,
                              "attempt": attempt, "error_type": type(exc).__name__,
                              "delay_seconds": delay}), flush=True)
            time.sleep(delay)


def stage_tabstar(*, execute=False, download=None):
    if not execute:
        raise RuntimeError("Pass --execute explicitly to stage the configured weights")
    if check_if_present().get("frozen"):
        raise ValueError("Stage predictor assets before freezing the tested stage")
    if os.environ.get("SLURM_JOB_GPUS") or os.environ.get("CUDA_VISIBLE_DEVICES") not in {None, "", "-1"}:
        raise ValueError("Download and hash predictor assets on CPU, before GPU allocation")
    settings = load_yaml("experiment_1/models")["models"]["tabstar"]["runtime"]
    root = paths.checkpoints_dir()
    marker = manifest_path()
    marker.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(marker.with_suffix(".lock"), timeout=1):
        if marker.exists():
            body = asset_manifest(settings)
            for pin in body["assets"]:
                if file_sha256(root / pin["path"]) != pin["sha256"]:
                    raise ValueError("Existing predictor assets changed; inspect before restaging")
            return body
        if download is None:
            from huggingface_hub import snapshot_download
            download = snapshot_download
        cache = root / "huggingface" / "hub"
        assets = {}
        for repo in settings["repositories"].values():
            folder = Path(_download_snapshot(
                download, repo_id=repo["model"], revision=repo["revision"], cache_dir=cache))
            expected = cache / ("models--" + repo["model"].replace("/", "--"))
            if folder != expected / "snapshots" / repo["revision"]:
                raise ValueError("Snapshot downloader returned an unexpected cache path")
            for source in sorted(folder.rglob("*")):
                if source.is_file():
                    if not source.resolve().is_relative_to(root.resolve()):
                        raise ValueError("Snapshot link escapes the checkpoint tree")
                    assets[source.relative_to(root).as_posix()] = file_sha256(source)
            reference = expected / "refs" / "main"
            if reference.exists() and reference.read_text(encoding="utf-8").strip() != repo["revision"]:
                raise ValueError("A different main alias already exists; review before changing it")
            reference.parent.mkdir(parents=True, exist_ok=True)
            atomic_text(reference, repo["revision"])
            assets[reference.relative_to(root).as_posix()] = file_sha256(reference)
        body = {"schema": "jevpfn.tabstar-assets.v1", "runtime": "tabstar",
                "repositories": settings["repositories"],
                "assets": [{"path": p, "sha256": sha} for p, sha in sorted(assets.items())]}
        body["manifest_id"] = digest(body)
        body["created_utc"] = datetime.now(UTC).isoformat()
        write_json(marker, body)
        return body


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = stage_tabstar(execute=args.execute)
    print(json.dumps({"manifest": str(manifest_path()), "files": len(result["assets"]),
                      "manifest_id": result["manifest_id"], "inference_executed": False}))
