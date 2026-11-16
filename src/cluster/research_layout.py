"""Move idle scientific result trees into research-question folders, without copying databases.

Operational extraction/token/0A records retain their recorded paths. Run only
after consumer reconciliation; the default prints a plan. Nothing is deleted.
"""

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from src.utils import paths
from src.utils.serialization import file_sha256, write_json

PHASES = ("experiment_1", "experiment_2", "experiment_3", "experiment_4", "synthesis")


def plan(root):
    root = Path(root).resolve()
    items = []
    for phase in PHASES:
        old = root / paths.phase_folder(phase) / "results"
        new = root / paths.result_relative(phase)
        if old.exists():
            if new.exists():
                raise ValueError("Both layouts contain results; review rather than overwrite or merge")
            items.append((old, new))
    return items


def migrate(root, *, writers_stopped=False):
    if not writers_stopped:
        raise ValueError("Confirm all result consumers stopped before changing paths")
    if paths.on_vsc():
        from src.cluster.maintenance import require_idle_vsc

        require_idle_vsc()
    root = Path(root).resolve()
    items = plan(root)
    inventory = {}
    for old, _new in items:
        for p in [old, *old.rglob("*")]:
            if p.is_symlink() or getattr(p, "is_junction", lambda: False)():
                raise ValueError("Linked result tree requires separate review")
            if not p.resolve().is_relative_to(root):
                raise ValueError("Result path escaped its owned output root")
            if p.is_file():
                if p.name.endswith(("-wal", "-journal", "-shm")):
                    raise ValueError("Database sidecar present; reconcile writers first")
                if p.suffix == ".sqlite3":
                    with closing(sqlite3.connect(p, timeout=1)) as db:
                        db.execute("BEGIN EXCLUSIVE")
                        if db.execute("PRAGMA quick_check").fetchone() != ("ok",):
                            raise ValueError("Result database integrity check failed")
                        db.rollback()
                inventory[str(p.relative_to(old)) + "|" + str(old)] = file_sha256(p)
    moved = []
    for old, new in items:
        new.parent.mkdir(parents=True, exist_ok=True)
        old.rename(new)
        for p in new.rglob("*"):
            if p.is_file() and file_sha256(p) != inventory[str(p.relative_to(new)) + "|" + str(old)]:
                raise RuntimeError("Moved result checksum mismatch; preserve both paths for review")
        moved.append(dict(source=str(old), destination=str(new), verified=True))
    report = dict(moved=moved, deleted_files=0, scientific_identities_changed=False)
    write_json(root / "manifests/research_layout.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=paths.large_outputs_dir())
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--writers-stopped", action="store_true")
    args = parser.parse_args()
    report = migrate(args.root, writers_stopped=args.writers_stopped) if args.apply else {
        "moves": [dict(source=str(a), destination=str(b)) for a, b in plan(args.root)]}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
