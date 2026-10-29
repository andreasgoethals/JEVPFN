"""Package the 18 frozen question stores locally; validate/install that bundle on VSC.

Raw datasets, responses, weights, credentials and code are never included. Existing
remote input files are backed up before replacement. Run with all writers stopped.
"""

import argparse
import json
import os
import shutil
import sqlite3
import tempfile
import zipfile
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from src.cluster.maintenance import reject_links, require_idle_vsc
from src.data.loaders import load_catalog
from src.data.selection import selected_entries
from src.jev.inputs import _read_set
from src.utils import paths
from src.utils.config import load_yaml
from src.utils.locking import file_lock
from src.utils.serialization import file_sha256, write_json

SCHEMA = "jevpfn.input_transfer.v1"


def selected_ids():
    return sorted(e["id"] for e in selected_entries(load_catalog(), load_yaml("feature_creation")))


def describe_input(path):
    path = Path(path).resolve(strict=True)
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("PRAGMA trusted_schema=OFF")
        if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise ValueError("Corrupt prepared-input database")
        identity, recipe, summary, entry = _read_set(db, None)
        if path.stem != recipe["dataset_id"]:
            raise ValueError("Input filename and dataset identity differ")
        if db.execute("PRAGMA foreign_key_check").fetchone():
            raise ValueError("Broken prepared row/input links")
        observed = [
            db.execute(sql, (identity,)).fetchone()[0]
            for sql in (
                "SELECT COUNT(*) FROM unique_inputs WHERE input_set_id=?",
                "SELECT COUNT(*) FROM row_map WHERE input_set_id=?",
            )
        ]
        if observed != [summary["unique_inputs"], recipe["row_count"] * len(recipe["slots"])]:
            raise ValueError("Incomplete prepared-input database")
    return {
        "dataset_id": recipe["dataset_id"],
        "filename": path.name,
        "input_set_id": identity,
        "bank_id": recipe["bank_id"],
        "question_version": entry["variants"][recipe["question_variant"]]["version"],
        "unique_inputs": summary["unique_inputs"],
        "rows": recipe["row_count"],
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def copy_closed_input(source, destination):
    """Byte-exact copy under a SQLite read transaction, preserving audit file hashes.

    Rollback-journal readers block commits. WAL databases/active journal files are
    refused rather than copying a main file without its committed sidecar pages.
    """
    source = Path(source).resolve(strict=True)
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True, timeout=1)) as db:
        if db.execute("PRAGMA journal_mode").fetchone()[0] == "wal":
            raise ValueError("Close/checkpoint the input database and leave WAL mode before upload")
        db.execute("BEGIN")
        db.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
        if any(Path(str(source) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")):
            raise ValueError("Input database has active/recovery sidecars; stop its writers first")
        shutil.copyfile(source, destination)
        if file_sha256(source) != file_sha256(destination):
            raise ValueError("Prepared-input copy differs from the source")


def pack(destination):
    if paths.on_vsc():
        raise RuntimeError("Build the input upload bundle on the local computer")
    destination = Path(destination).resolve()
    if destination.suffix != ".zip":
        raise ValueError("Choose a .zip output filename")
    ids = selected_ids()
    source_root = paths.decision_inputs_dir()
    if {p.stem for p in source_root.glob("*.sqlite3")} != set(ids):
        raise ValueError("Expected exactly the selected dataset question files")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        with zipfile.ZipFile(destination) as archive:
            prior = json.loads(archive.read("manifest.json"))
        if prior.get("schema") != SCHEMA or {r["dataset_id"] for r in prior["files"]} != set(ids):
            raise ValueError("Existing bundle belongs to another selection; choose a new name")
        if any(file_sha256(source_root / r["filename"]) != r["sha256"] for r in prior["files"]):
            raise FileExistsError("Inputs changed; choose a new bundle filename")
        expected = destination.with_suffix(".zip.sha256").read_text().split()[0]
        if file_sha256(destination) != expected:
            raise ValueError("Existing bundle checksum differs; preserve it and use a new name")
        return destination
    partial = destination.with_suffix(".zip.partial")
    if partial.exists():
        raise FileExistsError("Interrupted bundle exists; choose a new output filename")
    with tempfile.TemporaryDirectory(prefix="input_pack_", dir=destination.parent) as temporary:
        with zipfile.ZipFile(
            partial, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True
        ) as archive:
            records = []
            for dataset_id in ids:
                filename = f"{dataset_id}.sqlite3"
                copied = Path(temporary) / filename
                copy_closed_input(source_root / filename, copied)
                record = describe_input(copied)
                records.append(record)
                archive.write(copied, filename)
                copied.unlink()  # Only our temporary copy, never a source input.
                print(f"Packed {dataset_id}: {record['unique_inputs']:,} unique inputs", flush=True)
            if len({r["bank_id"] for r in records}) != 1:
                raise ValueError("All selected datasets must use the same active question bank")
            archive.writestr(
                "manifest.json", json.dumps({"schema": SCHEMA, "files": records}, indent=2)
            )
        partial.rename(destination)
    destination.with_suffix(".zip.sha256").write_text(
        file_sha256(destination) + "  " + destination.name + "\n", encoding="utf-8"
    )
    return destination


def install(bundle, *, writers_stopped=False):
    if not writers_stopped:
        raise ValueError("Stop manual writers and jobs, then pass --writers-stopped")
    if os.environ.get("SLURM_JOB_ID"):
        require_idle_vsc(allow_current=True)
    else:
        require_idle_vsc()
    bundle = paths.require_project_output(Path(bundle))
    expected = bundle.with_suffix(".zip.sha256").read_text().split()[0]
    if file_sha256(bundle) != expected:
        raise ValueError("Upload checksum mismatch; no inputs were replaced")
    root = paths.require_project_output(paths.decision_inputs_dir())
    reject_links(root)
    root.mkdir(parents=True, exist_ok=True)
    transfer_root = paths.require_project_output(paths.large_outputs_dir() / "transfers")
    transfer_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    report = {"status": "validating", "bundle_sha256": expected, "datasets": []}
    receipt = (
        paths.manifests_dir("experiment_0") / "before_extraction" / f"input_upload_{run_id}.json"
    )
    with (
        file_lock(paths.outputs_dir().parent / ".maintenance.lock", timeout=1),
        tempfile.TemporaryDirectory(prefix="input_install_", dir=transfer_root) as temporary,
    ):
        with zipfile.ZipFile(bundle) as archive:
            if archive.getinfo("manifest.json").file_size > 256_000:
                raise ValueError("Oversized upload manifest")
            manifest = json.loads(archive.read("manifest.json"))
            records = manifest.get("files", [])
            ids = selected_ids()
            if manifest.get("schema") != SCHEMA or sorted(r["dataset_id"] for r in records) != ids:
                raise ValueError("Bundle must contain exactly the selected dataset collection")
            names = {f"{dataset_id}.sqlite3" for dataset_id in ids} | {"manifest.json"}
            if set(archive.namelist()) != names or len(archive.namelist()) != len(names):
                raise ValueError("Unexpected/duplicate/path-traversal bundle members")
            # Validate every member before replacing any file; never use extractall.
            for record in records:
                name = f"{record['dataset_id']}.sqlite3"
                if record["filename"] != name or archive.getinfo(name).file_size != record["bytes"]:
                    raise ValueError("Bundle filename/size differs from manifest")
                copied = Path(temporary) / name
                with archive.open(name) as source, copied.open("xb") as target:
                    shutil.copyfileobj(source, target)
                if describe_input(copied) != record:
                    raise ValueError("Prepared input checksum/metadata mismatch")
                reject_links(root / name)
                if any(
                    Path(str(root / name) + suffix).exists()
                    for suffix in ("-wal", "-shm", "-journal")
                ):
                    raise ValueError(
                        "Remote input has active/recovery sidecars; stop its writers first"
                    )
            if len({r["bank_id"] for r in records}) != 1:
                raise ValueError("Mixed question-bank versions")
        write_json(receipt, report)
        try:
            for record in records:
                target = root / record["filename"]
                copied = Path(temporary) / target.name
                item = {"dataset_id": record["dataset_id"], "input_set_id": record["input_set_id"]}
                if target.exists() and file_sha256(target) == record["sha256"]:
                    item["status"] = "already_current"
                else:
                    if target.exists():
                        backup = transfer_root / "input_backups" / run_id / target.name
                        backup.parent.mkdir(parents=True, exist_ok=True)
                        copy_closed_input(target, backup)
                        item["backup"] = str(backup)
                    copied.replace(target)  # Atomic replacement, same PROJECT filesystem.
                    item["status"] = "installed"
                report["datasets"].append(item)
                write_json(receipt, report)
            report["status"] = "inputs_installed"
        except BaseException as exc:
            report.update(status="interrupted_retry_safe", error_type=type(exc).__name__)
            raise
        finally:
            write_json(receipt, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    packing = sub.add_parser("pack")
    packing.add_argument(
        "--output", type=Path, default=Path("output_JEVPFN/transfers/decision_inputs_v3.zip")
    )
    installing = sub.add_parser("install")
    installing.add_argument("bundle", type=Path)
    installing.add_argument("--writers-stopped", action="store_true")
    args = parser.parse_args()
    if args.command == "pack":
        print(pack(args.output))
    else:
        print(json.dumps(install(args.bundle, writers_stopped=args.writers_stopped), indent=2))


if __name__ == "__main__":
    main()
