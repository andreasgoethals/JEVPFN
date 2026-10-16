"""Seal/verify compact decision runtimes and retire only a verified redundant environment.

Images/wrappers remain on PROJECT. Building uses the site's hpc-container-wrapper
0.4.2 in scratch. This module never downloads weights, installs packages or submits jobs.
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import yaml

from src.cluster.maintenance import reject_links, require_idle_vsc
from src.jev.decision_tokenization import SELF_HOSTED
from src.utils import paths
from src.utils.serialization import digest, file_sha256, write_json

MARKER = "_JEVPFN_CONTAINER.json"


def container_dir(provider):
    if provider not in SELF_HOSTED:
        raise ValueError("Unknown decision provider")
    return paths.checkpoints_dir().parent / "containers" / provider


def native_dir(provider):
    container_dir(provider)  # Validate before constructing the maintenance target.
    return paths.checkpoints_dir().parent / "environments" / provider


def files(folder):
    """Hash image bytes and wrapper links without following image-only executable paths."""
    result = {}
    root = folder.resolve()
    config_file = folder / "share/conf.yaml"
    image_root = (
        (yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}).get("installation_path")
        if config_file.exists()
        else None
    )
    for path in sorted(folder.rglob("*")):
        if path.name in {MARKER, "retired_source.json"}:
            continue
        if path.is_symlink():
            link_text = os.readlink(path)
            if os.name == "nt" and link_text.startswith("\\\\?\\"):
                link_text = link_text[4:]
            target = Path(link_text)
            internal = path.resolve().is_relative_to(root)
            # hpc-container-wrapper deliberately points _bin/* at paths mounted
            # INSIDE its image; after retirement they need not exist on the host.
            image_link = (
                path.relative_to(folder).parts[0] == "_bin"
                and image_root
                and target.is_absolute()
                and ".." not in target.parts
                and target.is_relative_to(Path(image_root))
            )
            if not internal and not image_link:
                raise ValueError("Container wrapper has an external symlink")
            result[path.relative_to(folder).as_posix()] = digest({"symlink": str(target)})
            continue
        if path.is_file():
            result[path.relative_to(folder).as_posix()] = file_sha256(path)
    if not any(p.endswith(".sif") for p in result) or not any(p.endswith(".sqfs") for p in result):
        raise ValueError("Expected a complete container plus SquashFS installation image")
    if "bin/python" not in result:
        raise ValueError("Container Python wrapper is absent")
    return result


def probe(python, sentinel=None):
    code = (
        "import json,sys,os,platform,importlib.metadata as m; from pathlib import Path; "
        "import src; "
        "print('JEVPFN_RUNTIME='+json.dumps(dict(python=platform.python_version(),"
        "prefix=sys.prefix, source=str(Path(src.__file__).resolve()), "
        "project_visible=(Path(os.environ['JEVPFN_STAGING_ROOT'])/'JEVPFN/checkpoints').is_dir(), "
        "sentinel_visible=Path(sys.argv[1]).exists() if sys.argv[1] else False, "
        "packages={d.metadata['Name']:d.version for d in m.distributions() if d.metadata['Name']})))"
    )
    output = subprocess.check_output(
        [str(python), "-I", "-c", code, str(sentinel or "")], text=True, timeout=180
    )
    records = [
        line.partition("=")[2] for line in output.splitlines() if line.startswith("JEVPFN_RUNTIME=")
    ]
    if len(records) != 1:
        raise ValueError("Container runtime did not return one unambiguous receipt")
    runtime = json.loads(records[0])
    if not runtime["python"].startswith("3.12."):
        raise ValueError("Container must use Python 3.12")
    if not runtime["project_visible"]:
        raise ValueError("Container cannot access PROJECT checkpoints")
    if Path(runtime["source"]).parent != (paths.REPO_ROOT / "src").resolve():
        raise ValueError("Container imports a different project checkout")
    return runtime


def seal(provider, *, wrapped_existing=False):
    if not paths.on_vsc() or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Seal a container in its VSC build allocation")
    folder = reject_links(container_dir(provider))
    paths.require_project_output(folder)
    if (folder / MARKER).exists():
        raise ValueError("A sealed container already exists; do not replace it")
    sentinel = None
    baseline = None
    try:
        if wrapped_existing:
            original = reject_links(native_dir(provider))
            baseline = probe(original / "bin/python")
            sentinel = original / f".jevpfn_source_only_{uuid4().hex}"
            sentinel.write_text(
                "Must not be visible inside the already-built image", encoding="utf-8"
            )
        runtime = probe(folder / "bin/python", sentinel)
        if runtime["sentinel_visible"]:
            raise ValueError(
                "Wrapper reads the unpacked source instead of hiding it with the image"
            )
        if baseline and runtime["packages"] != baseline["packages"]:
            raise ValueError("Packed runtime package versions differ from the original")
        manifest = {
            "schema": "jevpfn.container.v1",
            "provider": provider,
            "created_utc": datetime.now(UTC).isoformat(),
            "wrapper_module": "hpc-container-wrapper/0.4.2",
            "wrapped_existing": wrapped_existing,
            "runtime": runtime,
            "files": files(folder),
        }
        manifest["container_id"] = digest(manifest)
        write_json(folder / MARKER, manifest)
        return manifest
    finally:
        if sentinel is not None:
            sentinel.unlink(missing_ok=True)


def verify(provider):
    folder = reject_links(container_dir(provider))
    manifest = json.loads((folder / MARKER).read_text(encoding="utf-8"))
    if (
        manifest.get("provider") != provider
        or manifest.get("schema") != "jevpfn.container.v1"
        or manifest.get("container_id")
        != digest({k: v for k, v in manifest.items() if k != "container_id"})
        or manifest.get("files") != files(folder)
    ):
        raise ValueError("Container fingerprint differs from its sealed receipt")
    return manifest


def active_identity():
    """The job launcher verifies image bytes before exposing this immutable identity."""
    provider = os.environ.get("JEVPFN_CONTAINER_PROVIDER")
    if not provider:
        return None
    manifest = json.loads((container_dir(provider) / MARKER).read_text(encoding="utf-8"))
    if manifest.get("container_id") != os.environ.get("JEVPFN_CONTAINER_ID") or manifest[
        "container_id"
    ] != digest({k: v for k, v in manifest.items() if k != "container_id"}):
        raise ValueError("Active container does not match its verified launch identity")
    return {
        "provider": provider,
        "container_id": manifest["container_id"],
        "files": manifest["files"],
        "wrapper_module": manifest["wrapper_module"],
    }


def retire(provider, *, apply=False, writers_stopped=False):
    """Retire only a wrapped source whose same image passed real GPU contracts."""
    manifest = verify(provider)
    target = reject_links(native_dir(provider))
    if not manifest["wrapped_existing"]:
        raise ValueError("This container has no redundant source environment to retire")
    evidence = []
    root = paths.manifests_dir("experiment_0") / "before_extraction" / "gpu_contracts"
    for p in root.glob("*.json"):
        report = json.loads(p.read_text(encoding="utf-8"))
        image = (
            report.get("settings", {}).get("request_configuration", {}).get("execution_container")
            or {}
        )
        if (
            report.get("provider") == provider
            and report.get("status") == "gpu_contracts_passed_requires_review"
            and report.get("runtime_token_count_parity") is True
            and image.get("container_id") == manifest["container_id"]
            and report.get("settings_sha256") == digest(report.get("settings", {}))
            and report.get("semantic_diagnostics", {}).get("passed") is True
        ):
            evidence.append(str(p))
    result = {
        "action": "retire_redundant_unpacked_environment",
        "path": str(target),
        "container": str(container_dir(provider)),
        "gpu_evidence": evidence,
        "applied": False,
        "preserved": ["weights", "data", "outputs", "other_environments", "container"],
    }
    if not apply:
        return result
    if not writers_stopped or not evidence:
        raise ValueError(
            "Stop writers and first obtain passing GPU evidence for this exact container"
        )
    require_idle_vsc()
    expected = container_dir(provider).parent.parent / "environments" / provider
    if target.resolve() != expected.resolve() or not target.resolve().is_relative_to(
        expected.parent.resolve()
    ):
        raise ValueError("Environment target is outside the selected PROJECT environment")
    if target.is_dir():
        if target.is_mount() or any(p.is_mount() for p in target.rglob("*") if not p.is_symlink()):
            raise ValueError("Environment contains a mount; retirement refused")
        shutil.rmtree(target)
    target.mkdir(exist_ok=True)  # Keep the empty image mount point for --mask wrappers.
    result["applied"] = True
    write_json(container_dir(provider) / "retired_source.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("seal", "verify", "retire"))
    parser.add_argument("provider", choices=SELF_HOSTED)
    parser.add_argument("--wrapped-existing", action="store_true")
    parser.add_argument("--shell", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--writers-stopped", action="store_true")
    args = parser.parse_args()
    if args.shell and args.action != "verify":
        parser.error("--shell is for verified runtime selection only")
    if args.action == "seal":
        result = seal(args.provider, wrapped_existing=args.wrapped_existing)
    elif args.action == "verify":
        result = verify(args.provider)
    else:
        result = retire(args.provider, apply=args.apply, writers_stopped=args.writers_stopped)
    if args.shell:
        print(f"export JEVPFN_CONTAINER_PROVIDER={shlex.quote(args.provider)}")
        print(f"export JEVPFN_CONTAINER_ID={shlex.quote(result['container_id'])}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
