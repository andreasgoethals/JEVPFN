"""List or remove only reproducible notebook reports and legacy question exports.

    python -m src.utils.clean_run          # preview, no deletion
    python -m src.utils.clean_run --clean  # remove listed rebuildable files

Logs, manifests, predictions, token audits, transfer bundles/backups, data and weights
are always preserved. Stop notebook writers before cleaning; rerun them to rebuild.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.utils.paths import large_outputs_dir, notebook_outputs_dir, outputs_dir

KEEP = frozenset({".gitkeep", ".gitignore"})


def _linked(path):
    return path.is_symlink() or getattr(path, "is_junction", lambda: False)()


def roots() -> list[Path]:
    found = []
    for root in (outputs_dir(), large_outputs_dir(), notebook_outputs_dir()):
        if _linked(root):
            raise ValueError("Cleanup refuses a linked output root")
        root = root.resolve()
        if not any(root.is_relative_to(previous) for previous in found):
            found.append(root)
    return found


def candidates(root: Path):
    """Allowlisted files under verified output roots; never follow junctions/symlinks."""
    root = Path(root)
    if _linked(root) or root.resolve() not in roots():
        raise ValueError("Cleanup is restricted to configured output directories")
    root = root.resolve()
    for directory, folders, files in os.walk(root, followlinks=False):
        folders[:] = [name for name in folders if not _linked(Path(directory) / name)]
        for name in files:
            path = Path(directory) / name
            if name in KEEP or path.is_symlink() or not path.resolve().is_relative_to(root):
                continue
            relative = path.relative_to(root)
            if (
                (
                    relative.parts[0] == "figures"
                    and (
                        path.suffix.lower() in {".pdf", ".png", ".svg", ".json", ".npz", ".csv"}
                        or name in {"Captions.md", "_report.txt", "_stdout.txt"}
                    )
                )
                or relative.as_posix() in {"All Results.md", "all_results.md", "Captions.md"}
                or (
                    relative.parts[:2] == ("1. question_preparation", "questions")
                    and path.suffix == ".json"
                )
            ):
                yield path


def measure(root: Path) -> tuple[int, int]:
    files = list(candidates(root))
    return len(files), sum(path.stat().st_size for path in files)


def wipe(root: Path) -> int:
    """Delete only the listed rebuildable files; never recursively remove a tree."""
    files = list(candidates(root))
    for path in files:
        path.unlink()
    return len(files)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true", help="remove listed rebuildable files only")
    args = parser.parse_args(argv)
    targets = roots()
    for root in targets:
        count, size = measure(root)
        print(f"{count:,} rebuildable files, {size / 1e6:.1f} MB: {root}")
    print("Preserved: logs, manifests, predictions, audits, transfers/backups, data and weights.")
    if not args.clean:
        print("Nothing was deleted. Use --clean to remove only these rebuildable files.")
        return 0
    for root in targets:
        print(f"Removed {wipe(root):,} rebuildable files from {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
