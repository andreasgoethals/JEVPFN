"""Seal and verify the CPU control image; bootstrap verification uses only stdlib.

Run this file directly with the host Python when activating the image. Imports of
project dependencies are deliberately confined to build/acceptance operations.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

MARKER = "_JEVPFN_CONTAINER.json"
LEAF = "control-v1"
HOST_VIEW = Path("/JEVPFN_CONTROL_HOST")


def host_view(folder):
    """Use the read-only unmasked bind while running inside this image."""
    image = os.environ.get("JEVPFN_CONTROL_IMAGE")
    if image and Path(folder).resolve() == Path(image).resolve() and HOST_VIEW.is_dir():
        return HOST_VIEW
    return Path(folder)


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


def child_command(command):
    """Reuse the current CPU interpreter rather than nesting the same CPU image again."""
    result = list(command)
    image = os.environ.get("JEVPFN_CONTROL_IMAGE")
    if image and result and Path(result[0]) == Path(image) / "bin/python":
        result[0] = sys.executable
    return result


def integration_test():
    from src.utils import paths
    from src.utils.serialization import write_json

    image = Path(os.environ["JEVPFN_CONTROL_IMAGE"])
    manifest = verify(image)
    root = Path(os.environ["JEVPFN_STAGING_ROOT"]) / "JEVPFN"
    commands = [
        [
            "sbatch",
            "--test-only",
            "--clusters=wice",
            "--account=lp_verbekelab",
            "--partition=batch",
            "--time=00:01:00",
            "--cpus-per-task=1",
            "--mem=1000M",
            "--wrap=true",
        ],
        [
            "sacct",
            "--clusters=wice",
            "-j",
            os.environ["SLURM_JOB_ID"],
            "--noheader",
            "--format=JobID,State",
        ],
        [
            sys.executable,
            "-m",
            "src.cluster.monitor",
            "--",
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
            check=False,
        )
        results.append(
            dict(
                command=command,
                returncode=run.returncode,
                stdout=run.stdout[-4000:],
                stderr=run.stderr[-2000:],
            )
        )
    result = dict(
        status="control_integration_passed"
        if all(r["returncode"] == 0 for r in results)
        else "control_integration_failed",
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
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "control_integration_passed":
        raise RuntimeError("Control integration failed; complete command diagnostics were saved")
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


def verify(folder, *, active=None):
    folder = Path(folder)
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError("CPU image must be a real container directory")
    manifest = json.loads((folder / MARKER).read_text(encoding="utf-8"))
    if (
        manifest.get("provider") != "control"
        or manifest.get("runtime_profile") != "cpu-control-v1"
        or manifest.get("container_id")
        != digest({k: v for k, v in manifest.items() if k != "container_id"})
        or manifest.get("files") != image_files(host_view(folder))
    ):
        raise ValueError("CPU control image differs from its sealed identity")
    if not any(name.endswith(".sqfs") for name in manifest["files"]):
        raise ValueError("CPU dependency SquashFS is missing")
    if active:
        selected = json.loads(Path(active).read_text(encoding="utf-8"))
        accepted = selected.get("container_id")
        if accepted not in {
            manifest["container_id"],
            manifest.get("relocation", {}).get("source_container_id"),
        }:
            raise ValueError("Active CPU image is different from the reviewed cutover")
    return manifest


def activate(job_id):
    from src.cluster.maintenance import require_idle_vsc
    from src.utils import paths
    from src.utils.serialization import write_json

    require_idle_vsc()
    root = paths.manifests_dir("experiment_0")
    before = root / "before_extraction"
    manifest = verify(paths.checkpoints_dir().parent / "containers" / LEAF)
    reports = []
    for files, status in (
        (before.glob("control_*.json"), "control_integration_passed"),
        (root.glob("preflight_*.json"), "preflight_passed"),
        (
            (before / "storage_contracts").glob("*.json"),
            "storage_contracts_passed_requires_quota_and_backup_review",
        ),
    ):
        matches = [
            p
            for p in files
            if (d := json.loads(p.read_text(encoding="utf-8"))).get("status") == status
            and str(d.get("job_id")) == job_id
        ]
        if len(matches) != 1:
            raise ValueError(
                "Cutover requires the successful integration, CPU and storage reports from one job"
            )
        reports.append(
            dict(
                path=str(matches[0]),
                sha256=file_hash(matches[0]),
                report=json.loads(matches[0].read_text(encoding="utf-8")),
            )
        )
    integration = json.loads(Path(reports[0]["path"]).read_text())
    if integration["container_id"] != manifest["container_id"]:
        raise ValueError("Control test belongs to another image")
    target = paths.checkpoints_dir().parent / "containers/control.active.json"
    if target.exists():
        raise ValueError("A control cutover is already recorded")
    write_json(
        target,
        dict(
            container_id=manifest["container_id"],
            validation_job_id=job_id,
            evidence=reports,
            legacy_environment_preserved_at_cutover=True,
        ),
    )
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
    parser.add_argument("action", choices=("verify", "seal", "test", "activate"))
    parser.add_argument("folder", type=Path, nargs="?")
    parser.add_argument("--active", type=Path)
    parser.add_argument("--test-job-id")
    args = parser.parse_args()
    result = (
        activate(args.test_job_id)
        if args.action == "activate"
        else (
            integration_test()
            if args.action == "test"
            else verify(args.folder, active=args.active)
            if args.action == "verify"
            else seal(args.folder)
        )
    )
    print(result["container_id"])


if __name__ == "__main__":
    main()
