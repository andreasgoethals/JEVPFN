"""Finding the sibling projects. Shared by both `sync_*.py` tools.

One copy, because two copies of a filesystem walk drift — and the template's own rules say
anything used in more than one place belongs in one place. Not a package: `_template/` holds
loose scripts, and `python _template/sync_x.py` puts that directory on `sys.path`, so a plain
`import _projects` resolves.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

SUBMODULE = "tfm-library"

#: How deep to look for a git repository under the projects directory. 2 covers both
#: `Projects/<Project>` and `Projects/<N. Project>/<Project>`.
MAX_DEPTH = 2

#: Never descended into. Without this the walk enters every project's own submodule, virtual
#: environment and output tree — slow, and it finds nothing.
SKIP = frozenset({".git", ".venv", "venv", "env", "node_modules", "__pycache__", SUBMODULE,
                  "output", "data", "checkpoints", ".pytest_cache", ".ruff_cache"})


def git(repo: Path, *args: str) -> tuple[int, str, str]:
    """Run git in `repo`. Returns (returncode, stdout, stderr) — never raises."""
    try:
        result = subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        return 127, "", "git is not on PATH"
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def default_root() -> Path:
    """Where the projects live: the parent of the repository holding this script."""
    return Path(__file__).resolve().parents[2]


def find_projects(root: Path) -> list[Path]:
    """Every git repository under `root` (to MAX_DEPTH) that declares the submodule.

    Filtering on `.gitmodules` rather than on a hard-coded list is what makes a new project
    appear the moment it has the library — nothing to remember to update here.
    """
    found: list[Path] = []
    seen: set[Path] = set()

    def visit(directory: Path, depth: int) -> None:
        if depth > MAX_DEPTH:
            return
        try:
            children = sorted(directory.iterdir())
        except OSError:
            return  # a directory we cannot read is not a project
        for child in children:
            if not child.is_dir() or child.name in SKIP:
                continue
            if (child / ".git").exists():
                gitmodules = child / ".gitmodules"
                if gitmodules.is_file() and SUBMODULE in gitmodules.read_text(encoding="utf-8"):
                    resolved = child.resolve()
                    if resolved not in seen:
                        seen.add(resolved)
                        found.append(child)
                continue  # a git repo is a leaf: do not walk into it looking for more
            visit(child, depth + 1)

    visit(root, 1)
    return found


def add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    """`--only` / `--exclude` / `--pick` / `--all`, mutually exclusive.

    `--only` and `--exclude` cannot be combined: one says "these" and the other says "not
    these", and a command meaning both is a command whose author was unsure.
    """
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--only", nargs="+", metavar="NAME", help="do only these projects")
    scope.add_argument("--exclude", nargs="+", metavar="NAME", help="do all but these")
    scope.add_argument("--pick", action="store_true", help="choose from a numbered list")
    scope.add_argument("--all", action="store_true", help="every project (the default)")


def select(projects: list[Path], args: argparse.Namespace) -> list[Path]:
    """Apply `--only` / `--exclude` / `--pick`. Default is every project.

    Names match case-insensitively on the directory name.
    """
    if getattr(args, "only", None):
        wanted = {n.lower() for n in args.only}
        chosen = [p for p in projects if p.name.lower() in wanted]
        missing = wanted - {p.name.lower() for p in chosen}
        if missing:
            # Loudly, not silently: a typo in `--only` would otherwise look like "that project
            # does not have the submodule", and you would go looking in the wrong place.
            raise SystemExit(
                f"--only named {sorted(missing)}, which is not among the projects found: "
                f"{[p.name for p in projects]}"
            )
        return chosen

    if getattr(args, "exclude", None):
        unwanted = {n.lower() for n in args.exclude}
        missing = unwanted - {p.name.lower() for p in projects}
        if missing:
            print(f"note: --exclude named {sorted(missing)}, which was not found anyway")
        return [p for p in projects if p.name.lower() not in unwanted]

    if getattr(args, "pick", False):
        return pick_interactively(projects)

    return projects


def pick_interactively(projects: list[Path]) -> list[Path]:
    """Print the numbered list and ask which to drop or keep."""
    print("Projects found:")
    for i, p in enumerate(projects, start=1):
        print(f"  {i:>2}. {p.name}")
    print(
        "\nEnter nothing for ALL of them,\n"
        "      `2,4`   to do only those two,\n"
        "      `-3`    to do everything except the third.\n"
    )
    try:
        answer = input("selection: ").strip()
    except EOFError:
        # No stdin (a scheduled task, a CI run). Silently doing all of them would be a
        # surprising default for a command whose whole point is choosing, so refuse.
        raise SystemExit(
            "--pick needs an interactive terminal; use --only or --exclude instead"
        ) from None

    if not answer:
        return projects

    drop = answer.lstrip().startswith("-")
    try:
        numbers = {int(part.strip().lstrip("-")) for part in answer.split(",") if part.strip()}
    except ValueError:
        raise SystemExit(f"could not read {answer!r} as a list of numbers") from None
    bad = [n for n in numbers if not 1 <= n <= len(projects)]
    if bad:
        raise SystemExit(f"{bad} out of range 1..{len(projects)}")

    if drop:
        return [p for i, p in enumerate(projects, start=1) if i not in numbers]
    return [p for i, p in enumerate(projects, start=1) if i in numbers]
