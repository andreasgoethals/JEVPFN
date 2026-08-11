"""Is this repository healthy? ONE command, three answers.

    python scripts/check.py              ruff, then every import, then pytest
    python scripts/check.py --fix        let ruff fix what it safely can, first
    python scripts/check.py --no-tests   lint and imports only — for a fast loop
    python scripts/check.py --quick      skip tests marked `slow`

ONE COMMAND, because three means one of them is the one nobody remembers — always the one that
would have caught the problem. CI runs this exact script, so the failure a reviewer sees is the
one the author can reproduce.

THE IMPORT CHECK IS ITS OWN STEP because ruff parses files without importing them, and pytest
only imports what a test touches. A circular import, or a typo'd import in an untested branch,
passes both and fails the first time it is used — usually twenty minutes into a cluster job.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Never linted, imported or collected. `tfm-library` is the read-only submodule: not our code,
#: and touching it is a rule violation.
EXCLUDE_DIRS = {"tfm-library", "output", "data", ".venv", "venv", "__pycache__", ".git"}


def _run(label: str, argv: list[str]) -> bool:
    """Run a step, stream its output, return whether it passed."""
    print(f"\n{'=' * 74}\n{label}\n{'=' * 74}", flush=True)
    try:
        completed = subprocess.run(argv, cwd=REPO_ROOT, check=False)
    except FileNotFoundError:
        # A missing tool is a setup problem, not a code problem — the common first-run
        # failure is a skipped `pip install -e ".[dev]"`, and saying so beats a traceback.
        print(f"  MISSING: {argv[0]} is not installed. Run:  pip install -e \".[dev]\"")
        return False
    return completed.returncode == 0


def src_modules() -> list[str]:
    """Every module under `src/`, as importable dotted names."""
    modules = []
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        rel = path.relative_to(REPO_ROOT).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts.pop()
        if parts:
            modules.append(".".join(parts))
    return modules


def check_imports() -> bool:
    """Import every `src` module in a fresh interpreter.

    A subprocess, not in-process `importlib`: a module that mutates global state on import
    (matplotlib's backend, a logging config) would leak into pytest and misattribute the failure.
    """
    modules = src_modules()
    if not modules:
        print("  no modules under src/ — nothing to import")
        return True
    program = (
        "import importlib, sys\n"
        f"mods = {modules!r}\n"
        "bad = []\n"
        "for m in mods:\n"
        "    try:\n"
        "        importlib.import_module(m)\n"
        "    except Exception as exc:\n"
        "        bad.append(f'{m}: {type(exc).__name__}: {exc}')\n"
        "for line in bad:\n"
        "    print('  FAILED', line)\n"
        "print(f'  {len(mods) - len(bad)}/{len(mods)} modules import cleanly')\n"
        "sys.exit(1 if bad else 0)\n"
    )
    return _run(f"IMPORTS  ({len(modules)} modules under src/)", [sys.executable, "-c", program])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fix", action="store_true", help="let ruff apply its safe fixes first")
    parser.add_argument("--no-tests", action="store_true", help="lint and imports only")
    parser.add_argument("--quick", action="store_true", help="skip tests marked `slow`")
    args = parser.parse_args(argv)

    results: list[tuple[str, bool]] = []

    if args.fix:
        _run("RUFF --fix", [sys.executable, "-m", "ruff", "check", "--fix", "."])

    results.append(("ruff", _run("RUFF", [sys.executable, "-m", "ruff", "check", "."])))
    results.append(("imports", check_imports()))

    if not args.no_tests:
        pytest_argv = [sys.executable, "-m", "pytest"]
        if args.quick:
            pytest_argv += ["-m", "not slow"]
        results.append(("pytest", _run("PYTEST", pytest_argv)))

    print(f"\n{'=' * 74}\nVERDICT\n{'=' * 74}")
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    healthy = all(ok for _, ok in results)
    print(f"\n{'Repository is healthy.' if healthy else 'NOT healthy — fix the FAILs above.'}")
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
