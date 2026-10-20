"""Export PROJECT/DATA results together, then import locally without overwriting caches.

No SSH, deletion, credentials, weights, environments or raw datasets are involved.
Use scp to download the resulting ZIP. An interrupted import can be repeated safely.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from uuid import uuid4

from src.jev.merge import merge_cache, snapshot, validate_cache
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.locking import file_lock
from src.utils.serialization import file_sha256, write_json


def transfer_dir():
    return paths.require_project_output(paths.large_outputs_dir() / "transfers")


def _safe_relative(value):
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or any(p in {".", ".."} or ":" in p or "\\" in p for p in path.parts)
    ):
        raise ValueError("Unsafe transfer path")
    if path.as_posix() != value:
        raise ValueError("Noncanonical transfer path")
    return path


def export_bundle() -> Path:
    """Run after writers finish. Bundle is large, so it always lives on PROJECT on VSC."""
    if paths.on_vsc():
        from src.cluster.preflight import storage_policy_failures

        failures = storage_policy_failures()
        if failures:
            raise ValueError("; ".join(failures))
    root = transfer_dir()
    root.mkdir(parents=True, exist_ok=True)
    bundle_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    destination = root / f"JEVPFN_{bundle_id}.zip"
    partial = destination.with_suffix(".partial")
    records, seen = [], {}
    with tempfile.TemporaryDirectory(dir=root, prefix="snapshot_") as scratch:
        with zipfile.ZipFile(
            partial, "w", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as archive:

            def add(source, logical_path, kind):
                _safe_relative(logical_path)
                name = f"files/{len(records):06d}/{source.name}"
                sha = file_sha256(source)
                if logical_path in seen:
                    if seen[logical_path] == sha:
                        return
                    # Same relative path on the two tiers must not erase either copy.
                    p = PurePosixPath(logical_path)
                    logical_path = str(p.with_name(f"{p.stem}__{sha[:12]}{p.suffix}"))
                seen[logical_path] = sha
                archive.write(source, name)
                records.append(
                    {
                        "member": name,
                        "path": logical_path,
                        "kind": kind,
                        "sha256": sha,
                        "bytes": source.stat().st_size,
                    }
                )

            for source in sorted(paths.jev_cache_path().glob("*.sqlite3")):
                with file_lock(source.with_suffix(".extraction.lock"), timeout=1):
                    copied = snapshot(source, Path(scratch) / source.name)
                dataset_id = validate_cache(copied)
                if source.stem != dataset_id:
                    raise ValueError("Dataset cache filename and identity differ")
                add(copied, f"data/jev_cache/{source.name}", "cache")
            from src.data.text_embeddings import validate as validate_embeddings

            for source in sorted(paths.jev_cache_path("text_embeddings").glob("*.sqlite3")):
                with file_lock(source.with_suffix(".lock"), timeout=1):
                    copied = snapshot(source, Path(scratch) / "embeddings" / source.name)
                if validate_embeddings(copied)["dataset_id"] != source.stem:
                    raise ValueError("Embedding cache filename and identity differ")
                add(copied, f"data/text_embeddings/{source.name}", "embedding_cache")
            for tier, output_root in enumerate(
                dict.fromkeys([paths.large_outputs_dir(), paths.outputs_dir()])
            ):
                for source in sorted(output_root.rglob("*")):
                    relative = source.relative_to(output_root)
                    if not source.is_file() or source.is_symlink():
                        continue
                    # Only numbered research phases; generated local notebook reports stay local.
                    if not relative.parts or relative.parts[0] not in {
                        paths.phase_folder(p)
                        for p in (
                            "question_preparation",
                            "feature_creation",
                            "experiment_0",
                            "experiment_1",
                            "experiment_2",
                            "experiment_3",
                            "experiment_4",
                        )
                    }:
                        continue
                    if (
                        source.name.startswith(".")
                        or source.suffix in {".lock", ".partial"}
                        or source.name.endswith(("-wal", "-shm", "-journal"))
                    ):
                        continue
                    copied = Path(scratch) / str(tier) / relative
                    copied.parent.mkdir(parents=True, exist_ok=True)
                    if source.suffix == ".sqlite3":
                        snapshot(source, copied)
                    else:
                        shutil.copyfile(source, copied)
                    add(copied, f"output_JEVPFN/{relative.as_posix()}", "output")
            manifest = {
                "schema": "jevpfn.transfer.v1",
                "bundle_id": bundle_id,
                "created_utc": datetime.now(UTC).isoformat(),
                "files": records,
                "omitted": [
                    "weights",
                    "environments",
                    "raw datasets",
                    "prepared inputs",
                    "credentials",
                ],
            }
            archive.writestr("manifest.json", json.dumps(manifest, indent=2))
        partial.rename(destination)
    write_json(
        paths.manifests_dir("experiment_0") / "transfers" / f"export_{bundle_id}.json",
        {
            "bundle_id": bundle_id,
            "bundle": str(destination),
            "sha256": file_sha256(destination),
            "files": len(records),
            "bytes": destination.stat().st_size,
        },
    )
    return destination


def _publish_file(source, destination):
    """Atomic copy; preserve a local collision as a separate content-addressed version."""
    sha = file_sha256(source)
    if destination.exists():
        if file_sha256(destination) == sha:
            return "reused", destination
        destination = destination.with_name(
            f"{destination.stem}__vsc_{sha[:12]}{destination.suffix}"
        )
        if destination.exists():
            if file_sha256(destination) != sha:
                raise ValueError("Conflicting versioned output file")
            return "reused", destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)
    return "added", destination


def _combine_token_reports(source, destination):
    """Extend the notebook's summary without losing provider audits already held locally."""
    incoming = json.loads(source.read_text(encoding="utf-8"))
    existing = (
        json.loads(destination.read_text(encoding="utf-8")) if destination.exists() else incoming
    )
    records = {r["provider"]: r for r in existing["self_hosted"]}
    for report in incoming["self_hosted"]:
        prior = records.get(report["provider"])
        if prior is None or prior["audit_id"] == report["audit_id"]:
            records[report["provider"]] = report
        # Different audit identities are retained in the versioned incoming report;
        # they need review/recount, never automatic replacement of a local audit.
    for report in records.values():
        report["counts_database"] = str(
            paths.token_audit_dir() / f"{report['provider']}_{report['audit_id']}.sqlite3"
        )
    from src.jev.decision_tokenization import HOSTED_UNAVAILABLE

    combined = {
        **existing,
        "self_hosted": list(records.values()),
        "hosted_unverified": {
            k: v
            for k, v in HOSTED_UNAVAILABLE.items()
            if k in load_yaml("extractors")["candidates"]
        },
        "imported_utc": datetime.now(UTC).isoformat(),
    }
    temporary = destination.with_name(destination.name + ".partial")
    write_json(temporary, combined)
    temporary.replace(destination)


def import_bundle(bundle: Path) -> dict:
    """Validate every archived byte/path first. Each cache merge is one atomic transaction.

    A failure after earlier datasets committed is safe to resume: repeat this command.
    The original bundle and per-dataset pre-merge backups remain available.
    """
    if paths.on_vsc():
        raise RuntimeError("Bundle import is for the local repository, not VSC")
    root = transfer_dir()
    root.mkdir(parents=True, exist_ok=True)
    bundle = Path(bundle).resolve()
    with (
        file_lock(root / ".import.lock", timeout=1),
        tempfile.TemporaryDirectory(dir=root, prefix="import_") as scratch,
    ):
        with zipfile.ZipFile(bundle) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("Duplicate ZIP members")
            manifest = json.loads(archive.read("manifest.json"))
            if manifest.get("schema") != "jevpfn.transfer.v1":
                raise ValueError("Unknown transfer schema")
            bundle_id = paths.output_component(manifest["bundle_id"])
            expected = {"manifest.json"}
            logical_paths = set()
            prepared = []
            for record in manifest["files"]:
                member, relative = _safe_relative(record["member"]), _safe_relative(record["path"])
                if str(member) in expected or str(relative) in logical_paths:
                    raise ValueError("Duplicate manifest entry")
                expected.add(str(member))
                logical_paths.add(str(relative))
                if record["kind"] in {"cache", "embedding_cache"}:
                    folder = "jev_cache" if record["kind"] == "cache" else "text_embeddings"
                    if (
                        relative.parts[:2] != ("data", folder)
                        or len(relative.parts) != 3
                        or relative.suffix != ".sqlite3"
                    ):
                        raise ValueError("Invalid cache destination")
                elif record["kind"] == "output":
                    if (
                        relative.parts[0] != "output_JEVPFN"
                        or len(relative.parts) < 3
                        or relative.parts[1] in {"transfers", "figures"}
                    ):
                        raise ValueError("Invalid output destination")
                else:
                    raise ValueError("Unknown transfer file kind")
                info = archive.getinfo(str(member))
                if (
                    info.file_size != record["bytes"]
                    or (info.external_attr >> 16) & 0o170000 == 0o120000
                ):
                    raise ValueError("Invalid archive size or symlink")
                target = Path(scratch) / str(len(prepared))
                with archive.open(info) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                if file_sha256(target) != record["sha256"]:
                    raise ValueError("Transfer checksum mismatch")
                if record["kind"] == "cache" and validate_cache(target) != relative.stem:
                    raise ValueError("Cache dataset and filename differ")
                if record["kind"] == "embedding_cache":
                    from src.data.text_embeddings import validate as validate_embeddings

                    if validate_embeddings(target)["dataset_id"] != relative.stem:
                        raise ValueError("Embedding dataset and filename differ")
                prepared.append((record, target, relative))
            if set(names) != expected:
                raise ValueError("Unexpected archive members")
        report = {
            "bundle_id": bundle_id,
            "bundle_sha256": file_sha256(bundle),
            "caches": [],
            "embedding_caches": [],
            "outputs": [],
        }
        token_report = None
        for record, source, relative in prepared:
            if record["kind"] == "cache":
                report["caches"].append(
                    merge_cache(
                        source,
                        paths.dataset_cache_path(relative.stem),
                        backup_dir=root / "cache_backups",
                    )
                )
            elif record["kind"] == "embedding_cache":
                from src.data.text_embeddings import cache_path, merge

                report["embedding_caches"].append(merge(source, cache_path(relative.stem)))
            else:
                destination = paths.outputs_dir().joinpath(*relative.parts[1:])
                # Refuse pre-existing symlink parents pointing outside the local output root.
                if not destination.resolve().is_relative_to(paths.outputs_dir().resolve()):
                    raise ValueError("Output destination escapes the local output folder")
                if relative.suffix == ".sqlite3" and relative.parts[-3:-1] == (
                    "results",
                    "evaluation",
                ):
                    from src.evaluation.store import merge as merge_evaluations

                    status = merge_evaluations(source, destination)
                    published = destination
                else:
                    status, published = _publish_file(source, destination)
                report["outputs"].append({"status": status, "path": str(published)})
                if (
                    relative.name == "token_audit.json"
                    and relative.parts[-2] == "before_extraction"
                ):
                    token_report = source, destination
        if token_report:
            _combine_token_reports(*token_report)
        write_json(
            paths.manifests_dir("experiment_0") / "transfers" / f"import_{bundle_id}.json", report
        )
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("export")
    importer = sub.add_parser("import")
    importer.add_argument("bundle", type=Path)
    args = parser.parse_args()
    if args.command == "export":
        print(export_bundle(), flush=True)
    else:
        print(json.dumps(import_bundle(args.bundle), indent=2), flush=True)


if __name__ == "__main__":
    main()
