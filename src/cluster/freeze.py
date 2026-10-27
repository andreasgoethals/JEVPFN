"""Freeze a validation campaign's source/base config; notebook output is not source.

The receipt lives in output and is created AFTER the deliberate final cleanup.
Private *.local.yaml files have separate per-run config identities and are excluded.
No Git operation modifies the checkout. A mismatch stops jobs, never edits files.
"""

import argparse
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from src.utils import paths
from src.utils.locking import file_lock
from src.utils.serialization import digest, file_sha256, write_json


def receipt_path():
    return paths.manifests_dir("experiment_0") / "frozen_revision.json"


def runtime_pins():
    # The frozen campaign pins durable assets, independent of each staged subset.
    root = (
        Path(os.environ["JEVPFN_DURABLE_ROOT"]) / "JEVPFN"
        if os.environ.get("JEVPFN_STORAGE_MODE") == "scratch_mirror"
        else paths.checkpoints_dir().parent
    )
    patterns = (
        "containers/*/_JEVPFN_CONTAINER.json",
        "containers/*/*/_JEVPFN_CONTAINER.json",
        "environments/*/JEVPFN_RUNTIME.json",
        "checkpoints/decision_models/*/_JEVPFN_CHECKPOINT.json",
        "checkpoints/text_embeddings/*/_JEVPFN_EMBEDDING.json",
        "checkpoints/predictors/*/_JEVPFN_PREDICTOR.json",
    )
    return {
        p.relative_to(root).as_posix(): file_sha256(p)
        for pattern in patterns
        for p in sorted(root.glob(pattern))
    }


def source_hash(path):
    """Treat CRLF/LF as equal in text source; shell scripts must actually use LF."""
    content = path.read_bytes()
    if path.suffix in {".sh", ".slurm"} and b"\r" in content:
        raise ValueError(f"Shell source must use LF line endings: {path.name}")
    return hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()


def source_snapshot():
    root = paths.REPO_ROOT
    files = {}
    for directory, suffixes in (
        ("src", {".py"}),
        ("tests", {".py"}),
        ("scripts", {".sh", ".slurm"}),
        ("config", {".yaml", ".toml"}),
    ):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and path.suffix in suffixes and not path.name.endswith(".local.yaml"):
                files[path.relative_to(root).as_posix()] = source_hash(path)
    for name in ("pyproject.toml", "AGENTS.md", ".gitignore", ".gitattributes"):
        if (root / name).is_file():
            files[name] = source_hash(root / name)
    for path in sorted((root / "notebooks").rglob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        files[path.relative_to(root).as_posix()] = digest(
            [
                {"type": cell["cell_type"], "source": "".join(cell.get("source", []))}
                for cell in notebook["cells"]
            ]
        )
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    return {
        "schema": "jevpfn.frozen-source.v2",
        "git_head": git.stdout.strip() or None,
        "source_sha256": digest(files),
        "files": files,
    }


def check_if_present():
    path = receipt_path()
    if not path.exists():
        return {"frozen": False, "stage": "setup_and_debugging"}
    stored = json.loads(path.read_text(encoding="utf-8"))
    current = source_snapshot()
    if any(stored.get(k) != current[k] for k in current):
        changed = sorted(
            k
            for k in stored.get("files", {}).keys() | current["files"].keys()
            if stored.get("files", {}).get(k) != current["files"].get(k)
        )
        raise ValueError(
            "Frozen revision changed; stop the campaign and review: " + ", ".join(changed)
        )
    if stored.get("runtime_pins", {}) != runtime_pins():
        raise ValueError(
            "Frozen checkpoint/container/runtime receipts changed; review before continuing"
        )
    return {
        "frozen": True,
        "source_sha256": stored["source_sha256"],
        "git_head": stored["git_head"],
    }


def record(*, expected_file=None):
    current = source_snapshot()
    if expected_file is not None:
        expected = json.loads(expected_file.read_text(encoding="utf-8"))
        if (
            expected.get("source_sha256") != current["source_sha256"]
            or expected.get("git_head") != current["git_head"]
            or expected.get("schema") != current["schema"]
        ):
            differences = sorted(
                k
                for k in expected.get("files", {}).keys() | current["files"].keys()
                if expected.get("files", {}).get(k) != current["files"].get(k)
            )
            raise ValueError(
                "Local and VSC source revisions differ; do not start the clean run. "
                f"HEAD equal: {expected.get('git_head') == current['git_head']}; "
                f"schema equal: {expected.get('schema') == current['schema']}; "
                "different files: " + ", ".join(differences)
            )
    path = receipt_path()
    if path.exists():
        return check_if_present()
    current["created_utc"] = datetime.now(UTC).isoformat()
    current["runtime_pins"] = runtime_pins()
    write_json(path, current)
    return {"receipt": str(path), **check_if_present()}


def finish_stage(*, reviewed_by, writers_stopped=False):
    """Keep the tested receipt before explicitly opening the next stage's setup window."""
    if not reviewed_by or not reviewed_by.strip() or not writers_stopped:
        raise ValueError("A named reviewer and stopped writers are required to finish a stage")
    if paths.on_vsc():
        from src.cluster.maintenance import require_idle_vsc

        require_idle_vsc()
    path = receipt_path()
    with file_lock(path.with_suffix(".lock"), timeout=1):
        if not path.exists():
            raise ValueError("No active frozen stage to finish")
        check_if_present()  # A changed revision cannot be passed off as the tested one.
        stored = json.loads(path.read_text(encoding="utf-8"))
        completed = path.parent / "completed_revisions" / f"{digest(stored)}.json"
        if not completed.exists():
            write_json(
                completed,
                {
                    "frozen_receipt": stored,
                    "reviewed_by": reviewed_by.strip(),
                    "completed_utc": datetime.now(UTC).isoformat(),
                    "meaning": "Owner ended the fixed revision stage; not automatic test acceptance",
                },
            )
        elif json.loads(completed.read_text(encoding="utf-8")).get("frozen_receipt") != stored:
            raise ValueError("Completed revision receipt conflicts; active freeze preserved")
        path.unlink()
    return {"frozen": False, "completed_receipt": str(completed), "stage": "next_stage_setup"}


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--record", action="store_true")
    mode.add_argument("--finish-stage", action="store_true")
    parser.add_argument("--expected-file", type=Path)
    parser.add_argument("--reviewed-by")
    parser.add_argument("--writers-stopped", action="store_true")
    args = parser.parse_args()
    if args.expected_file and not args.record:
        parser.error("--expected-file requires --record")
    result = (
        finish_stage(reviewed_by=args.reviewed_by, writers_stopped=args.writers_stopped)
        if args.finish_stage
        else record(expected_file=args.expected_file)
        if args.record
        else check_if_present()
    )
    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
