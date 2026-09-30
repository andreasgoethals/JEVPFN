"""Wipe what the previous run produced, so the next one starts clean.

    python -m src.utils.clean_run                        list what is there, delete nothing
    python -m src.utils.clean_run --clean                 delete it
    python -m src.utils.clean_run --clean --processed      ...and the data/processed cache too

Clears the **whole `output_JEVPFN/` tree on both storage tiers** — `$VSC_DATA` and project storage — so
one invocation is enough whether you are on a laptop or on the cluster. Off-cluster both tiers
collapse into the repository and it is simply `output_JEVPFN/`.

`--processed` additionally clears `data/processed/`, the preprocessing cache. It is separate
because rebuilding that cache can cost far more than re-running the notebooks, so "clean the last
run" should not silently throw it away.

LISTS BY DEFAULT. The two mistakes are not symmetric: a listing you meant as a deletion costs one
more command, and a deletion you meant as a listing costs the run.

NEVER TOUCHES `data/raw/` or `checkpoints/` or `tfm-library/` — the inputs are irreplaceable and
the weights are either downloaded or a training run to reproduce.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.paths import large_outputs_dir, outputs_dir, processed_dir

#: Tracked so an empty directory survives a clone. Not run output, so never counted or deleted —
#: removing them would leave a fresh clone with nowhere to write.
KEEP = frozenset({".gitkeep", ".gitignore"})


def roots(*, processed: bool = False) -> list[Path]:
    """Every tree to clear. Two `output_JEVPFN/` roots on the cluster, one locally, plus the cache.

    `results_dir()` is listed separately because on the cluster it is the one part of `output_JEVPFN/`
    on project storage — clearing only `outputs_dir()` there would leave the largest files behind.
    """
    found = [outputs_dir()]
    results = large_outputs_dir()
    if not results.is_relative_to(found[0]):
        found.append(results)
    if processed:
        found.append(processed_dir())
    return found


def measure(root: Path) -> tuple[int, int]:
    """(files, bytes) under a root, ignoring the structure markers."""
    if not root.is_dir():
        return 0, 0
    files = [p for p in root.rglob("*") if p.is_file() and p.name not in KEEP]
    return len(files), sum(p.stat().st_size for p in files)


def wipe(root: Path) -> int:
    """Delete everything under a root except the structure markers. Returns files removed.

    Two passes, and the order matters: files first, then empty directories bottom-up. That leaves
    exactly the directories holding a tracked `.gitkeep` and removes the per-run ones
    (`figures/<notebook>/`) that do not. An `rmtree` of the subtree would take
    `output_JEVPFN/<phase>/figures/.gitkeep` with it, and the next clone would have nowhere to write.
    """
    if not root.is_dir():
        return 0
    removed = 0
    for path in root.rglob("*"):
        if path.is_file() and path.name not in KEEP:
            path.unlink()
            removed += 1
    for path in sorted((p for p in root.rglob("*") if p.is_dir()),
                       key=lambda p: len(p.parts), reverse=True):
        if not any(path.iterdir()):
            path.rmdir()
    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--clean", action="store_true", help="actually delete; default lists only")
    parser.add_argument("--processed", action="store_true",
                        help="also clear data/processed/, the preprocessing cache")
    args = parser.parse_args(argv)

    targets = roots(processed=args.processed)
    total_files = total_bytes = 0
    print("Output from the previous run:\n")
    for root in targets:
        files, size = measure(root)
        total_files += files
        total_bytes += size
        state = f"{files:>6} files  {size / 1e6:>9.1f} MB" if files else "         empty"
        print(f"  {state}  {root}")

    print(f"\nTOTAL: {total_files} files, {total_bytes / 1e9:.2f} GB")
    print("Never touched: data/raw/, checkpoints/, tfm-library/.")

    if not args.clean:
        if total_files:
            print("\nNothing was deleted. Re-run with --clean to delete.")
        return 0

    print("\nDeleting:")
    for root in targets:
        print(f"  removed {wipe(root):>6} files from {root}")
    print("\nClean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
