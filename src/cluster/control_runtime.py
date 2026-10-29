"""Seal and verify the CPU control image; bootstrap verification uses only stdlib.

Run this file directly with the host Python when activating the image. Imports of
project dependencies are deliberately confined to build/acceptance operations.
"""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

MARKER = "_JEVPFN_CONTAINER.json"
LEAF = "control-v1"


def child_environment():
    """Do not leak a parent image's Conda stack or automatic binds into a child image."""
    result = dict(os.environ)
    for key in list(result):
        if key.startswith(("CONDA_", "_CONDA_")) or key in {
            "_CE_M",
            "_CE_CONDA",
            "APPTAINER_BIND",
            "SINGULARITY_BIND",
        }:
            result.pop(key)
    return result


def integration_test():
    from src.utils import paths
    from src.utils.serialization import write_json

    image = Path(os.environ["JEVPFN_CONTROL_IMAGE"])
    manifest = verify(image)
    root = Path(os.environ["JEVPFN_STAGING_ROOT"]) / "JEVPFN"
    commands = [
        [
            "sacct",
            "--clusters=wice",
            "-j",
            os.environ["SLURM_JOB_ID"],
            "--noheader",
            "--format=JobID,State",
        ],
        [
            str(root / "containers/blackwell-cu130/cloudflare/bin/python"),
            "-I",
            "-c",
            "import sys; assert sys.version_info[:2]==(3,12); print('NESTED_IMAGE_PASSED')",
        ],
    ]
    results = []
    for command in commands:
        run = subprocess.run(
            command,
            env=child_environment(),
            text=True,
            capture_output=True,
            timeout=120,
            check=True,
        )
        results.append(dict(command=command, stdout=run.stdout[-4000:], stderr=run.stderr[-2000:]))
    result = dict(
        status="control_integration_passed",
        container_id=manifest["container_id"],
        job_id=os.environ["SLURM_JOB_ID"],
        checks=results,
    )
    write_json(
        paths.manifests_dir("experiment_0")
        / "before_extraction"
        / ("control_" + os.environ["SLURM_JOB_ID"] + ".json"),
        result,
    )
    return result


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_hash(path):
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def image_files(folder):
    result = {}
    for path in sorted(folder.rglob("*")):
        if path.name in {MARKER, "retired_source.json"}:
            continue
        if path.is_symlink():
            result[path.relative_to(folder).as_posix()] = digest(
                {"symlink": str(Path(os.readlink(path)))}
            )
        elif path.is_file():
            result[path.relative_to(folder).as_posix()] = file_hash(path)
    return result


def verify(folder):
    folder = Path(folder)
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError("CPU image must be a real container directory")
    manifest = json.loads((folder / MARKER).read_text(encoding="utf-8"))
    if (
        manifest.get("provider") != "control"
        or manifest.get("runtime_profile") != "cpu-control-v1"
        or manifest.get("container_id")
        != digest({k: v for k, v in manifest.items() if k != "container_id"})
        or manifest.get("files") != image_files(folder)
    ):
        raise ValueError("CPU control image differs from its sealed identity")
    if not any(name.endswith(".sqfs") for name in manifest["files"]):
        raise ValueError("CPU dependency SquashFS is missing")
    return manifest


def seal(folder):
    from datetime import UTC, datetime

    from src.cluster.containers import files
    from src.utils.config import load_yaml
    from src.utils.serialization import write_json

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Build/seal the CPU image in a Slurm allocation")
    folder = Path(folder)
    if (folder / MARKER).exists():
        raise ValueError("A sealed CPU image cannot be overwritten")
    cfg = load_yaml("experiment_0/control_runtime")
    code = (
        "import json,sys,importlib.metadata as m; import src; "
        "print('CONTROL_RUNTIME='+json.dumps({'python':sys.version.split()[0],"
        "'source':src.__file__,'packages':{d.metadata['Name'].lower().replace('_','-'):d.version "
        "for d in m.distributions() if d.metadata['Name']}}))"
    )
    output = subprocess.check_output(
        [str(folder / "bin/python"), "-I", "-c", code], text=True, timeout=180
    )
    records = [
        json.loads(s.split("=", 1)[1])
        for s in output.splitlines()
        if s.startswith("CONTROL_RUNTIME=")
    ]
    if len(records) != 1:
        raise ValueError("CPU image returned no unique runtime identity")
    runtime = records[0]
    expected = {k.lower().replace("_", "-"): v for k, v in cfg["packages"].items()}
    if runtime["python"] != cfg["python"] or any(
        runtime["packages"].get(k) != v for k, v in expected.items()
    ):
        raise ValueError("CPU image does not reproduce the recorded dependency versions")
    from src.utils.paths import REPO_ROOT

    if Path(runtime["source"]).resolve() != (REPO_ROOT / "src/__init__.py").resolve():
        raise ValueError("CPU image imports another checkout")
    subprocess.run([str(folder / "bin/python"), "-m", "pip", "check"], check=True, timeout=120)
    result = dict(
        schema="jevpfn.container.v1",
        provider="control",
        runtime_profile="cpu-control-v1",
        created_utc=datetime.now(UTC).isoformat(),
        runtime=runtime,
        files=files(folder),
        wrapper_module="hpc-container-wrapper/0.4.2",
    )
    result["container_id"] = digest(result)
    write_json(folder / MARKER, result)
    return verify(folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "seal", "test"))
    parser.add_argument("folder", type=Path, nargs="?")
    args = parser.parse_args()
    result = (
        integration_test()
        if args.action == "test"
        else (verify if args.action == "verify" else seal)(args.folder)
    )
    print(result["container_id"])


if __name__ == "__main__":
    main()
