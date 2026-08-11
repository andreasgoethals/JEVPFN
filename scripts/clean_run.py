"""Delete what a previous run left behind. LISTS BY DEFAULT — deletes only with --clean.

    python scripts/clean_run.py                          list everything, delete nothing
    python scripts/clean_run.py --clean                   delete the cheap categories
    python scripts/clean_run.py --clean --all             cheap + results + processed
    python scripts/clean_run.py --clean --only logs runs  exactly these

Runs identically on a laptop and on the cluster: `src/utils/run_artifacts` resolves both
storage tiers, so one invocation covers `$VSC_DATA` and project storage.

WHY LISTING IS THE DEFAULT and deleting is the flag: the two mistakes are not equally
expensive. A listing you meant as a deletion costs one more command; a deletion you meant
as a listing costs the run.

NEVER REMOVED, whatever you pass: `data/raw/`, `checkpoints/`, `tfm-library/`, and the
repository's own directories. That is enforced in `run_artifacts.protected_paths()`, not
here, so no combination of arguments can reach them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Template idiom: a script is invoked as a file, not imported, so the repository root has
# to be on sys.path before `from src...` can resolve. This is why ruff's E402 is disabled
# for scripts/ in pyproject.toml.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.run_artifacts import (  # noqa: E402
    CATEGORIES,
    CHEAP,
    EXPENSIVE,
    clean,
    find_artifacts,
    summarise,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="actually delete. Without this, nothing is removed.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help=f"include the expensive categories {EXPENSIVE} as well as {CHEAP}",
    )
    parser.add_argument(
        "--only", nargs="+", metavar="CATEGORY", choices=CATEGORIES,
        help=f"clean exactly these: {' '.join(CATEGORIES)}",
    )
    args = parser.parse_args(argv)

    artifacts = find_artifacts()
    print(summarise(artifacts))

    if not args.clean:
        if artifacts:
            print("\nNothing was deleted. Re-run with --clean to delete.")
        return 0

    if args.only:
        categories = tuple(args.only)
    elif args.all:
        categories = tuple(CHEAP) + tuple(EXPENSIVE)
    else:
        categories = tuple(CHEAP)

    print(f"\nDeleting: {', '.join(categories)}")
    report = clean(categories, dry_run=False)
    for path in report["removed"]:
        print(f"  removed  {path}")
    for failure in report["failed"]:
        print(f"  FAILED   {failure}")
    print(f"\nFreed {report['freed_gb']:.2f} GB from {len(report['removed'])} locations.")
    if report["failed"]:
        print(f"{len(report['failed'])} locations could not be removed — see above.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
