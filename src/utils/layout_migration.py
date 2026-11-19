"""Move an existing output tree into the current layout (config/structure.yaml).

The 18-11-2026 layout was revised on 19-11-2026: one flat notebook per folder (records in
`figures/<folder>/`, `{logs,manifests,results}/notebooks/<folder>/`), `exp09_adaptation`
renamed `exp10_adaptation`, and records of the four merged adaptation notebooks archived
under `archives/layout_20261119/`. The same command migrates either previous layout.

Preview by default; nothing changes without --apply --writers-stopped:

    python -m src.utils.layout_migration --tier local
    python -m src.utils.layout_migration --tier local --apply --writers-stopped
    python -m src.utils.layout_migration --tier data      # on VSC: $VSC_DATA/JEVPFN/output_JEVPFN
    python -m src.utils.layout_migration --tier project   # on VSC: <durable>/JEVPFN/output_JEVPFN
    python -m src.utils.layout_migration --root <.../output_JEVPFN>

The plan uses the same translation as `paths`, so code and data agree afterwards. Directories
are renamed within the one output root (never copied); nothing is deleted except source
directories that the moves left empty. Before moving, every file is inventoried (count, size,
SHA-256); after moving, the same files are verified at their destinations. A destination that
already exists, a link, a live SQLite sidecar or a locked database refuses the whole plan.
Paths outside the layout (transfers, workspaces, archives, ...) stay where they are. The
receipt `<root>/manifests/layout_migration_<UTC>.json` maps every moved file old -> new.
Internal phase IDs, recipes, receipts and hashes are untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import stat
import subprocess
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from src.utils import paths, structure
from src.utils.serialization import file_sha256, write_json

SCHEMA = "jevpfn.layout-migration.v1"
SIDECARS = ("-wal", "-shm", "-journal")


def _linked(path: Path) -> bool:
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


def tier_root(tier: str) -> Path:
    """The output root of one tier; VSC tiers only resolve in a VSC shell."""
    if tier == "local":
        if paths.on_vsc():
            raise RuntimeError("--tier local is for the local computer")
        return paths.notebook_outputs_dir()
    if not paths.on_vsc():
        raise RuntimeError("DATA/PROJECT tiers resolve only in a VSC login shell; use --root to preview a copy")
    if tier == "data":
        return paths.outputs_dir()
    if tier == "project":
        durable = paths.durable_project_override()
        if durable is None:
            raise ValueError("Select the durable PROJECT root explicitly")
        return paths._under(durable, paths.OUTPUT_DIR_NAME)
    raise ValueError("Tier must be local, data or project")


def _files_under(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file() or _linked(p))


def _uniform(root: Path, relative: PurePosixPath, target: PurePosixPath) -> bool:
    """Whether every file below `relative` moves by the same prefix substitution."""
    for file in _files_under(root / relative):
        sub = PurePosixPath(file.relative_to(root / relative).as_posix())
        if structure.translate_output(relative / sub) != target / sub:
            return False
    return True


def _occupied(root: Path, target: PurePosixPath, planned: list[PurePosixPath]) -> bool:
    if (root / target).exists() or _linked(root / target):
        return True
    return any(target == p or target in p.parents or p in target.parents for p in planned)


def plan(root: Path) -> dict:
    """Disjoint renames (source dir/file -> destination) plus files left unmapped."""
    root = Path(root)
    if root.name != paths.OUTPUT_DIR_NAME:
        raise ValueError("Select an exact output_JEVPFN folder")
    if _linked(root) or not root.is_dir():
        raise ValueError("Output root must be a real directory")
    moves: list[tuple[PurePosixPath, PurePosixPath]] = []
    empty: list[PurePosixPath] = []
    unmapped: list[str] = []
    conflicts: list[str] = []

    def visit(relative: PurePosixPath) -> None:
        path = root / relative
        if _linked(path):
            conflicts.append(f"link: {relative}")
            return
        target = structure.translate_output(relative)
        if path.is_file():
            if target is None:
                unmapped.append(relative.as_posix())
            elif target != relative:
                if _occupied(root, target, [m[1] for m in moves]):
                    conflicts.append(f"destination exists: {relative} -> {target}")
                else:
                    moves.append((relative, target))
            return
        logical = structure.logical_output(relative.parts)
        # A whole phase tree is never one unit: two previous trees (e.g. `4. experiment_1`
        # and `rq2/4. experiment_1`) merge into one folder, child by child.
        whole_tree = logical is not None and not logical[1]
        if (target is not None and target != relative and not whole_tree
                and _uniform(root, relative, target)):
            if _occupied(root, target, [m[1] for m in moves]):
                if not any(path.iterdir()):
                    empty.append(relative)
                    return  # An empty legacy directory; removed after the moves.
                # A partly existing destination: move the children individually.
            else:
                moves.append((relative, target))
                return
        for child in sorted(path.iterdir(), key=lambda p: p.name):
            visit(relative / child.name)

    for child in sorted(root.iterdir(), key=lambda p: p.name):
        visit(PurePosixPath(child.name))
    sources = {s for s, _ in moves}
    return {"root": str(root), "moves": moves, "unmapped": unmapped, "conflicts": conflicts,
            "sources": sources, "empty": empty}


def _inventory(root: Path, moves) -> dict[str, dict]:
    inventory = {}
    for source, target in moves:
        for file in _files_under(root / source):
            if _linked(file):
                raise ValueError(f"Linked file requires separate review: {file}")
            if file.name.endswith(SIDECARS):
                raise ValueError(f"Database sidecar present; stop writers first: {file}")
            if file.suffix == ".sqlite3":
                with closing(sqlite3.connect(file.resolve().as_uri() + "?mode=rw", uri=True, timeout=1)) as db:
                    db.execute("BEGIN EXCLUSIVE")  # Refuse while any writer holds the database.
                    db.rollback()
            relative = PurePosixPath(file.relative_to(root).as_posix())
            sub = relative.relative_to(source) if relative != source else PurePosixPath()
            destination = target / sub if sub.parts else target
            inventory[relative.as_posix()] = {
                "destination": destination.as_posix(),
                "bytes": file.stat().st_size,
                "sha256": file_sha256(file),
            }
    return inventory


def _git_head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.REPO_ROOT, capture_output=True,
                              text=True, timeout=30, check=False).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _summary(result: dict) -> dict:
    return {
        "root": result["root"],
        "moves": [{"source": s.as_posix(), "destination": d.as_posix()} for s, d in result["moves"]],
        "conflicts": result["conflicts"],
        "empty_legacy_directories": [e.as_posix() for e in result["empty"]],
        "unmapped_files": len(result["unmapped"]),
        "unmapped_top_level": sorted({PurePosixPath(u).parts[0] for u in result["unmapped"]}),
    }


def migrate(root: Path, *, writers_stopped: bool = False) -> dict:
    """Apply the plan; verify count, sizes and SHA-256 at every destination; write a receipt."""
    if not writers_stopped:
        raise ValueError("Confirm every writer/consumer of this output tree has stopped")
    if paths.on_vsc():
        from src.cluster.maintenance import require_idle_vsc

        require_idle_vsc()
    root = Path(root)
    result = plan(root)
    if result["conflicts"]:
        raise ValueError("Layout migration refused: " + "; ".join(result["conflicts"]))
    moves = result["moves"]
    inventory = _inventory(root, moves)
    started = datetime.now(UTC)
    receipt_path = root / "manifests" / f"layout_migration_{started.strftime('%Y%m%dT%H%M%SZ')}.json"
    receipt = {
        "schema": SCHEMA,
        "status": "moving",
        "created_utc": started.isoformat(),
        "root": str(root),
        "registry": str(structure.REGISTRY),
        "registry_sha256": file_sha256(structure.REGISTRY),
        "git_head": _git_head(),
        "planned_moves": [{"source": s.as_posix(), "destination": d.as_posix()} for s, d in moves],
        "files": inventory,
        "file_count": len(inventory),
        "bytes": sum(item["bytes"] for item in inventory.values()),
        "unmapped_files_left_in_place": result["unmapped"],
        "deleted_files": 0,
        "scientific_identities_changed": False,
    }
    # Written before anything moves, so an interruption still leaves the old -> new map.
    write_json(receipt_path, receipt)
    moved = []
    for source, target in moves:
        destination = root / target
        if destination.exists() or _linked(destination):
            raise ValueError(f"Destination appeared during migration: {target}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.rename(root / source, destination)  # Same output root: an atomic rename, no copy.
        moved.append({"source": source.as_posix(), "destination": target.as_posix()})
    receipt.update(moves=moved, status="moved_verifying")
    write_json(receipt_path, receipt)
    failures = []
    for old, item in inventory.items():
        file = root / item["destination"]
        if (not file.is_file() or file.stat().st_size != item["bytes"]
                or file_sha256(file) != item["sha256"]):
            failures.append(old)
    receipt.update(verified=not failures, verification_failures=failures,
                   status="verified" if not failures else "verification_failed")
    write_json(receipt_path, receipt)
    if failures:
        raise RuntimeError("Moved files failed verification; both layouts preserved for review: "
                           + ", ".join(failures[:10]))
    removed, kept = remove_empty_legacy_directories(root, [s for s, _ in moves], result["empty"])
    receipt.update(removed_empty_directories=removed, empty_directories_left=kept,
                   completed_utc=datetime.now(UTC).isoformat(), status="complete")
    write_json(receipt_path, receipt)
    return receipt


def _remove_empty(directory: Path) -> bool:
    """rmdir only; a read-only attribute (Windows/Drive restore) is cleared on the EMPTY dir."""
    try:
        directory.rmdir()
        return True
    except PermissionError:
        if os.name != "nt" or any(directory.iterdir()):
            return False
        try:
            os.chmod(directory, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
            directory.rmdir()
            return True
        except OSError:
            return False


def remove_empty_legacy_directories(root: Path, sources=(), empty=()) -> tuple[list[str], list[str]]:
    """Remove previous-layout directories that the moves left empty. Never removes a file.

    `empty` lists previous-layout directories the plan found already empty (e.g. a former
    `figures/<folder>/<notebook>/` that never held a figure).
    """
    root = Path(root)
    candidates = {root / e for e in empty}
    for source in sources:
        for parent in (root / source).parents:
            if parent == root or not parent.is_relative_to(root):
                break
            candidates.add(parent)
    for directory in root.rglob("*"):
        if directory.is_dir() and not _linked(directory):
            relative = PurePosixPath(directory.relative_to(root).as_posix())
            legacy = (relative.parts[0] in {"rq1", "rq2", "rq3", "regression_readout"}
                      or structure.legacy_phase(relative.parts[0]) is not None
                      or (len(relative.parts) >= 2 and (relative.parts[1] in structure.LEGACY_GROUPS
                                                         or structure.legacy_phase(relative.parts[1])))
                      or (len(relative.parts) >= 3 and relative.parts[1] == "notebooks"
                          and (relative.parts[2] in structure.LEGACY_GROUPS
                               or structure.legacy_phase(relative.parts[2]))))
            if legacy and relative.parts[0] not in {"archives", "transfers", "workspaces"}:
                candidates.add(directory)
    removed, kept = [], []
    for directory in sorted(candidates, key=lambda p: -len(p.parts)):
        if not directory.is_dir() or _linked(directory) or any(directory.iterdir()):
            continue
        relative = PurePosixPath(directory.relative_to(root).as_posix()).as_posix()
        (removed if _remove_empty(directory) else kept).append(relative)
    return sorted(removed), sorted(kept)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--tier", choices=("local", "data", "project"))
    where.add_argument("--root", type=Path)
    parser.add_argument("--apply", action="store_true", help="move; default is a preview")
    parser.add_argument("--writers-stopped", action="store_true")
    args = parser.parse_args(argv)
    root = args.root or tier_root(args.tier)
    if args.apply:
        receipt = migrate(root, writers_stopped=args.writers_stopped)
        report = {k: receipt[k] for k in ("root", "file_count", "bytes", "verified", "moves",
                                           "removed_empty_directories")}
    else:
        report = _summary(plan(root))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
