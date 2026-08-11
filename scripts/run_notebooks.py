"""Re-run every notebook in parallel and rebuild CAPTIONS.md and All_Results.md.

    python scripts/run_notebooks.py                       every notebook in notebooks/
    python scripts/run_notebooks.py --only exploration    just these, by stem
    python scripts/run_notebooks.py --workers 2           fewer processes
    python scripts/run_notebooks.py --summaries-only      rebuild the two .md files only

`--summaries-only` exists because both documents are built from what the notebooks left on
disk (`_figures.json`), so after an interactive Jupyter session they can be regenerated
without executing anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.run_notebooks import (  # noqa: E402
    discover,
    run_all,
    summarise,
    write_all_results,
    write_captions,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", nargs="+", metavar="STEM", help="notebook stems to run")
    parser.add_argument("--workers", type=int, default=None, help="parallel processes")
    parser.add_argument("--timeout", type=int, default=1800, help="per-notebook seconds")
    parser.add_argument(
        "--summaries-only", action="store_true",
        help="rebuild CAPTIONS.md and All_Results.md from disk, run nothing",
    )
    args = parser.parse_args(argv)

    names = discover(tuple(args.only) if args.only else None)
    if not names:
        print("No notebooks found in notebooks/.")
        return 0

    if args.summaries_only:
        print(f"Rebuilding summaries from disk for: {', '.join(names)}")
        print(f"  captions  -> {write_captions(names)}")
        print(f"  summaries -> {write_all_results(names)}")
        return 0

    print(f"Running {len(names)} notebook(s): {', '.join(names)}")
    results = run_all(names, max_workers=args.workers, timeout=args.timeout)
    print(summarise(results))
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
