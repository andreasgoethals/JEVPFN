"""Native B200 working sets with durable, conflict-checked publication to Lustre.

CPU staging copies only a selected runtime and its required inputs. GPFS scratch is
never called a backup. Publication uses SQLite snapshots and existing cache unions;
interrupted jobs leave their working set intact for explicit recovery.
"""

import argparse
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from src.cluster.maintenance import reject_links
from src.jev.merge import merge_cache, snapshot
from src.utils import paths
from src.utils.locking import file_lock
from src.utils.serialization import digest, file_sha256, write_json

MARKER = "_JEVPFN_WORKSPACE.json"


def roots(runtime):
    from src.cluster.containers import RUNTIMES

    if runtime not in RUNTIMES:
        raise ValueError("Unknown runtime for B200 working set")
    durable = Path(os.environ.get("JEVPFN_DURABLE_ROOT", paths.STAGING_FALLBACK))
    scratch = os.environ.get("VSC_SCRATCH_GPFS1")
    if not scratch or not Path(scratch).is_absolute():
        raise ValueError("VSC_SCRATCH_GPFS1 must identify your existing GPFS scratch allocation")
    scratch = reject_links(Path(scratch))
    durable = reject_links(durable)
    if not scratch.is_dir() or not durable.is_dir():
        raise ValueError("Both native scratch and durable project allocations must exist")
    if scratch == durable or scratch.is_relative_to(durable) or durable.is_relative_to(scratch):
        raise ValueError("Scratch and durable roots must be separate allocations")
    if paths.on_vsc() and ("gpfs" not in str(scratch) or "lustre" not in str(durable)):
        raise ValueError("B200 uses GPFS scratch and durable Lustre, not personal DATA")
    return (
        reject_links(durable / "JEVPFN"),
        reject_links(scratch / "jevpfn-work" / runtime / "JEVPFN"),
    )


@contextmanager
def staging_environment(root):
    previous = os.environ.get("JEVPFN_STAGING_ROOT")
    os.environ["JEVPFN_STAGING_ROOT"] = str(root.parent)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("JEVPFN_STAGING_ROOT", None)
        else:
            os.environ["JEVPFN_STAGING_ROOT"] = previous


@contextmanager
def workspace_lease(root, *, shared=False):
    """GPU jobs share a lease; restaging requires exclusivity. Kernel releases on crash."""
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".workspace.lock"
    if os.name == "nt":
        with file_lock(lock, timeout=1):
            yield
        return
    import fcntl

    with lock.open("a+b") as handle:
        try:
            fcntl.flock(handle, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "B200 working set is in use; do not restage an active runtime"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _regular_files(folder, *, link_boundary=None):
    reject_links(folder)
    if not folder.exists():
        return
    for source in sorted(folder.rglob("*")):
        # HF download metadata and transient writers are not model assets.
        if any(part == ".cache" for part in source.relative_to(folder).parts):
            continue
        if (source.is_symlink() or getattr(source, "is_junction", lambda: False)()) and (
            link_boundary is None or not source.resolve().is_relative_to(link_boundary.resolve())
        ):
            raise ValueError(f"Linked input escapes the approved checkpoint tree: {source}")
        if source.is_file() and not source.name.endswith(
            (".lock", "-wal", "-shm", "-journal", ".partial")
        ):
            yield source


def _dataset_stores(folder):
    """Only active dataset DBs: recovery backups and control files are not inputs."""
    reject_links(folder)
    for source in sorted(folder.glob("*.sqlite3")):
        reject_links(source)
        if source.is_file():
            yield source


def _copy(source, destination, *, snapshot_sqlite=True):
    reject_links(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".stage-", dir=destination.parent) as temporary:
        copied = Path(temporary) / source.name
        if source.suffix == ".sqlite3" and snapshot_sqlite:
            snapshot(source, copied)
        else:
            if source.suffix == ".sqlite3" and any(
                Path(str(source) + suffix).exists() and Path(str(source) + suffix).stat().st_size
                for suffix in ("-wal", "-journal")
            ):
                raise ValueError("Prepared inputs must be closed/checkpointed before transfer")
            before = file_sha256(source)
            shutil.copyfile(source, copied)
            if before != file_sha256(source) or before != file_sha256(copied):
                raise ValueError("Input changed during staging")
        sha = file_sha256(copied)
        if destination.is_file() and file_sha256(destination) == sha:
            return sha
        os.replace(copied, destination)
        if file_sha256(destination) != sha:
            raise ValueError("Staging checksum mismatch")
        return sha


def _publish(durable, work, *, periodic=False):
    """Never replace a durable response DB with a partial scratch copy."""
    from src.cluster.transfer import _publish_file
    from src.data.text_embeddings import merge as merge_embeddings
    from src.evaluation.store import merge as merge_evaluations

    counts = {"decision_stores": 0, "embedding_stores": 0, "outputs": 0}
    staged = (
        json.loads((work / MARKER).read_text()).get("files", {}) if (work / MARKER).exists() else {}
    )
    durable.mkdir(parents=True, exist_ok=True)
    with (
        staging_environment(durable),
        file_lock(durable / ".scratch-publication.lock", timeout=600),
        tempfile.TemporaryDirectory(prefix=".publish-", dir=durable) as temporary,
    ):
        for kind, folder, merger in (
            ("decision_stores", "data/jev_cache", merge_cache),
            ("embedding_stores", "data/text_embeddings", merge_embeddings),
            ("outputs", "output_JEVPFN", None),
        ):
            sources = (
                _dataset_stores(work / folder)
                if kind == "decision_stores"
                else _regular_files(work / folder)
            )
            for source in sources:
                relative = source.relative_to(work)
                if kind != "outputs" and source.suffix != ".sqlite3":
                    continue
                if kind == "outputs" and "results" not in relative.parts:
                    continue
                if (
                    periodic
                    and kind == "outputs"
                    and not (source.suffix == ".sqlite3" and "evaluation" in relative.parts)
                ):
                    continue
                destination = durable / relative
                reject_links(destination)
                # Immutable CPU inputs already have an authoritative durable copy.
                if (
                    kind == "outputs"
                    and relative.as_posix() in staged
                    and file_sha256(source) == staged[relative.as_posix()]
                ):
                    continue
                copied = Path(temporary) / relative
                copied.parent.mkdir(parents=True, exist_ok=True)
                if source.suffix == ".sqlite3":
                    snapshot(source, copied)
                else:
                    shutil.copyfile(source, copied)
                if kind == "decision_stores":
                    # Both working and durable copies survive; the atomic union
                    # does not need a growing full-store backup every ten minutes.
                    merger(copied, destination, backup_destination=False, deduplicate_ledger=True)
                elif merger is not None:
                    merger(copied, destination)
                elif source.suffix == ".sqlite3" and "evaluation" in relative.parts:
                    merge_evaluations(copied, destination)
                else:
                    _publish_file(copied, destination)
                counts[kind] += 1
    return counts


def publish(runtime):
    durable, work = roots(runtime)
    if not (work / MARKER).is_file():
        raise ValueError("No staged B200 workspace; nothing can be published")
    with workspace_lease(work, shared=True):
        marker = json.loads((work / MARKER).read_text())
        if marker["durable_root"] != str(durable) or marker["runtime"] != runtime:
            raise ValueError("Workspace publication destination differs from its stage receipt")
        counts = _publish(durable, work)
    report = {"runtime": runtime, "status": "published", "durable_root": str(durable), **counts}
    write_json(
        paths.manifests_dir("experiment_0") / "scratch" / f"{runtime}_publication.json", report
    )
    return report


def stage(runtime):
    from src.cluster import containers
    from src.cluster.runtime_profiles import BLACKWELL, selected
    from src.cluster.scheduling import require_cpu_staging

    if selected() != BLACKWELL:
        raise ValueError("B200 staging requires JEVPFN_RUNTIME_PROFILE=blackwell-cu130")

    durable, work = roots(runtime)
    require_cpu_staging()
    with workspace_lease(work):
        if (work / MARKER).exists():
            previous = json.loads((work / MARKER).read_text())
            if previous.get("runtime") != runtime or previous.get("durable_root") != str(durable):
                raise ValueError("Workspace publication destination differs from its stage receipt")
            _publish(durable, work)  # Preserve committed results before updating any inputs.
        write_json(
            work / MARKER,
            {
                "runtime": runtime,
                "runtime_profile": selected(),
                "durable_root": str(durable),
                "status": "staging",
                "files": {},
            },
        )
        inputs = ["data/raw", "data/decision_inputs", "data/jev_cache"]
        if runtime == "cloudflare" or runtime not in containers.SELF_HOSTED:
            inputs += ["data/text_embeddings"]
        if runtime in containers.SELF_HOSTED:
            inputs += [
                f"checkpoints/decision_models/{runtime}",
                f"checkpoints/tokenizers/{runtime}",
            ]
            if runtime in {"nimble", "strands"}:
                inputs += [f"checkpoints/decision_models/{runtime}_base"]
            if runtime == "strands":
                # The publisher loads its separate base by HF ID, even offline.
                from src.utils.config import load_yaml

                base = load_yaml("extractors")["candidates"][runtime]["base_model"]
                inputs += ["checkpoints/huggingface/hub/models--" + base.replace("/", "--")]
        else:
            # Predictor pins explicitly declare assets; never duplicate the ten
            # unrelated decision checkpoints in every predictor working set.
            from src.utils.config import load_yaml

            runtime_cfg = load_yaml("experiment_1/models")["models"][runtime]["runtime"]

            def pinned_paths(value):
                if isinstance(value, dict):
                    for key, item in value.items():
                        if key == "path" and item:
                            asset = Path(item)
                            if not asset.is_absolute():
                                asset = durable / "checkpoints" / asset
                            if not asset.resolve().is_relative_to(
                                (durable / "checkpoints").resolve()
                            ):
                                raise ValueError("Predictor pin is outside durable checkpoints")
                            yield asset.relative_to(durable)
                        else:
                            yield from pinned_paths(item)
                elif isinstance(value, list):
                    for item in value:
                        yield from pinned_paths(item)

            inputs += [str(p) for p in pinned_paths(runtime_cfg)]
        if runtime == "cloudflare":
            inputs += ["checkpoints/text_embeddings"]
        # Include prepared CPU artifacts (tokens and fold-specific PCA), never arbitrary outputs.
        inputs += [
            f"output_JEVPFN/{paths.phase_folder(p)}/results/{name}"
            for p, name in (
                ("experiment_0", "token_audit"),
                ("experiment_2", "pca"),
                ("experiment_1", "pca"),
            )
        ]
        candidates = {}
        for folder in inputs:
            location = durable / folder
            sources = (
                _dataset_stores(location)
                if folder == "data/jev_cache"
                else (
                    [location]
                    if location.is_file()
                    else _regular_files(location, link_boundary=durable / "checkpoints")
                )
            )
            for source in sources:
                relative = source.relative_to(durable)
                if (
                    "token_audit" in relative.parts
                    and runtime in containers.SELF_HOSTED
                    and not source.name.startswith(runtime + "_")
                ):
                    continue
                candidates[relative] = source
        if not any(p.parts[:2] == ("data", "decision_inputs") for p in candidates):
            raise ValueError("Prepared questions are absent from durable storage")
        with staging_environment(durable):
            source_container = containers.container_dir(runtime)
        needed = sum(p.stat().st_size for p in candidates.values())
        needed += sum(p.stat().st_size for p in source_container.rglob("*") if p.is_file())
        if shutil.disk_usage(work).free < needed + 2 * 1024**3:
            raise OSError(
                "Insufficient filesystem capacity for this working set; also check myquota"
            )
        files = {}
        for relative, source in candidates.items():
            destination = work / relative
            reject_links(destination)
            # Preserve operational queues in scratch; never replace them with the
            # durable union, whose imported queues intentionally remain archival.
            if relative.parts[:2] == ("data", "jev_cache") and destination.is_file():
                with (
                    staging_environment(work),
                    tempfile.TemporaryDirectory(prefix=".merge-", dir=destination.parent) as temp,
                ):
                    incoming = snapshot(source, Path(temp) / source.name)
                    merge_cache(
                        incoming, destination, backup_destination=False, deduplicate_ledger=True
                    )
                files[relative.as_posix()] = file_sha256(destination)
            elif relative.parts[:2] == ("data", "text_embeddings") and destination.is_file():
                from src.data.text_embeddings import merge

                with tempfile.TemporaryDirectory(prefix=".merge-", dir=destination.parent) as temp:
                    merge(snapshot(source, Path(temp) / source.name), destination)
                files[relative.as_posix()] = file_sha256(destination)
            else:
                # Token receipts bind the exact prepared-input file bytes. SQLite
                # backup changes its header, so immutable inputs must copy verbatim.
                files[relative.as_posix()] = _copy(
                    source,
                    destination,
                    snapshot_sqlite=relative.parts[:2] != ("data", "decision_inputs"),
                )
        with staging_environment(work):
            target_container = containers.container_dir(runtime)
        if not target_container.exists():
            shutil.copytree(source_container, target_container, symlinks=True)
        containers.relocate(source_container, target_container)
        if runtime in containers.SELF_HOSTED:
            from src.jev.checkpoints import verify_checkpoint

            # Prime file identities on the native filesystem during the CPU transfer.
            with staging_environment(work):
                verify_checkpoint(runtime)
        report = {
            "schema": "jevpfn.scratch.v1",
            "runtime": runtime,
            "runtime_profile": selected(),
            "durable_root": str(durable),
            "working_root": str(work),
            "files": files,
            "input_identity": digest(files),
            "status": "staged_requires_b200_contracts",
            "quota_verified": False,
            "scratch_is_backup": False,
        }
        write_json(work / MARKER, report)
        write_json(
            paths.manifests_dir("experiment_0") / "scratch" / f"{runtime}_stage.json", report
        )
        return {k: v for k, v in report.items() if k != "files"}


def run(runtime, command):
    """Hold the working-set lease through the child, and publish even after failure."""
    import signal
    import subprocess
    import threading

    from src.cluster.runtime_profiles import selected

    durable, work = roots(runtime)
    if Path(os.environ["JEVPFN_STAGING_ROOT"]).resolve() != work.parent.resolve():
        raise ValueError("GPU job did not select its native working set")
    with workspace_lease(work, shared=True):
        # Read after acquiring the lease: a CPU restage can otherwise replace the
        # marker between validation and admission of this inference process.
        marker = json.loads((work / MARKER).read_text())
        if (
            marker["runtime"] != runtime
            or marker["durable_root"] != str(durable)
            or marker["runtime_profile"] != selected()
            or marker.get("status") != "staged_requires_b200_contracts"
        ):
            raise ValueError("Wrong scratch workspace identity")
        process = subprocess.Popen(command)
        stopped = threading.Event()
        publication_errors = []

        def checkpoint():
            # Only committed SQLite pages are copied. This also protects long
            # extraction jobs that never get a normal exit after node failure.
            while not stopped.wait(600):
                try:
                    _publish(durable, work, periodic=True)
                except Exception as exc:
                    publication_errors.append(exc)
                    process.terminate()
                    return

        thread = threading.Thread(target=checkpoint, daemon=True)
        thread.start()
        previous = signal.signal(signal.SIGTERM, lambda *_: process.terminate())
        try:
            code = process.wait()
        finally:
            stopped.set()
            thread.join()
            signal.signal(signal.SIGTERM, previous)
            _publish(durable, work)
        if publication_errors:
            raise RuntimeError(
                "Periodic durable publication failed; scratch is retained"
            ) from publication_errors[0]
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stage", "publish", "run", "path"))
    parser.add_argument("--runtime", required=True)
    args, command = parser.parse_known_args()
    if args.action == "path":
        print(roots(args.runtime)[1].parent)
    elif args.action == "run":
        command = command[1:] if command[:1] == ["--"] else command
        if not command:
            parser.error("run needs -- COMMAND ...")
        raise SystemExit(run(args.runtime, command))
    else:
        if command:
            parser.error("Unknown arguments: " + " ".join(command))
        print(json.dumps((stage if args.action == "stage" else publish)(args.runtime), indent=2))


if __name__ == "__main__":
    main()
