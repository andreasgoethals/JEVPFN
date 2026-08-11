"""Bump the `tfm-library/` pin across EVERY project at once. Reports by default.

    python _template/sync_tfm_library.py                        every project, report only
    python _template/sync_tfm_library.py --only CreditICL CreditPFN
    python _template/sync_tfm_library.py --exclude TabPFN       all but these
    python _template/sync_tfm_library.py --pick                 choose interactively
    python _template/sync_tfm_library.py --update               move each working tree
    python _template/sync_tfm_library.py --update --commit      ...and record each new pin
    python _template/sync_tfm_library.py --projects-dir "D:/Projects"

SELECTION. Default is all of them. `--only` narrows to a list, `--exclude` removes a few from
the whole set, and `--pick` prints the numbered list and asks — enter nothing to keep them all,
`2,4` for only those two, `-3` for everything except the third. Names match
case-insensitively on the directory name. Shared with `sync_template_rules.py` via
`_projects.py`, so both tools select the same way.

THIS LIVES IN `_template/` because it is the one tool whose job spans projects — a project has
no business bumping its siblings. Each project also ships `scripts/update_tfm_library.py`,
which does the same thing for itself alone.

HOW PROJECTS ARE FOUND: every git repository under the projects directory, searched two levels
deep so a layout like `Projects/4. CreditICL/CreditICL` is found, then filtered to the ones
whose `.gitmodules` declares `tfm-library`. Nothing is hard-coded, so a new project is picked
up the moment it has the submodule.

IT NEVER PUSHES, and it never writes anything inside `tfm-library/` — it only moves which
commit a project points at.

WHY `--commit` IS SEPARATE FROM `--update`: `git submodule update --remote` moves the working
tree but does NOT record the pin, and the whole reason the literature is pinned is that a
result stays reproducible against the sources as they stood when it was produced. Moving that
silently across five repositories is exactly what the pin exists to prevent.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import _projects
from _projects import MAX_DEPTH, SUBMODULE, git


@dataclass
class Project:
    path: Path
    pin: str = ""
    other_staged: bool = False
    note: str = ""

    @property
    def name(self) -> str:
        return self.path.name


def inspect(project: Project) -> Project:
    """Fill in the pin, and the state that makes committing unsafe."""
    code, out, err = git(project.path, "submodule", "status", SUBMODULE)
    if code != 0:
        project.note = err.splitlines()[0] if err else "git submodule status failed"
        return project

    line = next((ln.strip() for ln in out.splitlines() if SUBMODULE in ln), "")
    project.pin = line
    if line.startswith("-"):
        # A `-` prefix means registered but never initialised: the folder is there and empty,
        # which reads as "the library is missing" if you do not know this.
        project.note = "not initialised — `git submodule update --init` there first"
    elif line.startswith("+"):
        project.note = "tree already moved, pin NOT recorded"

    # Anything else staged. A commit here must only ever move the pin, so this is checked
    # before acting, not after.
    code, staged, _ = git(project.path, "diff", "--cached", "--name-only")
    if code == 0 and staged:
        project.other_staged = any(
            f.strip() and f.strip() != SUBMODULE for f in staged.splitlines()
        )
        if project.other_staged:
            project.note = (project.note + "; " if project.note else "") + "other changes staged"
    return project


def report(projects: list[Project], total: int) -> None:
    print(f"{'#':>3}  {'project':<26} {'pin':<45} note")
    print("-" * 108)
    for i, p in enumerate(projects, start=1):
        print(f"{i:>3}. {p.name:<26} {(p.pin or '(unknown)')[:45]:<45} {p.note}")
    skipped = total - len(projects)
    print()
    print(f"{len(projects)} of {total} project(s) carrying {SUBMODULE}/"
          + (f" — {skipped} deselected" if skipped else ""))


def update(project: Project, commit: bool) -> str:
    """Move this project's working tree, and optionally record the pin. Returns a verdict."""
    if project.note.startswith("not initialised"):
        return "SKIPPED — submodule not initialised"

    code, _, err = git(project.path, "submodule", "update", "--remote", SUBMODULE)
    if code != 0:
        return f"FAILED to fetch — {err.splitlines()[0] if err else 'unknown error'}"

    _, unstaged, _ = git(project.path, "diff", "--", SUBMODULE)
    _, staged, _ = git(project.path, "diff", "--cached", "--", SUBMODULE)
    if not unstaged and not staged:
        return "already at the newest commit"

    after = next(
        (ln.strip() for ln in git(project.path, "submodule", "status", SUBMODULE)[1].splitlines()
         if SUBMODULE in ln),
        "",
    )
    if not commit:
        return f"tree moved to {after[:44]} — pin NOT recorded (add --commit)"

    if project.other_staged:
        # Refusing is the only safe option: the alternative is silently folding someone's
        # half-finished work into a commit labelled "Bump tfm-library pin".
        return "REFUSED to commit — other changes are staged here; commit those first"

    git(project.path, "add", SUBMODULE)
    # The `-- tfm-library` pathspec is not cosmetic: a bare `git commit` would sweep in
    # whatever else is in the index, and this tool must only ever move the pin.
    code, _, err = git(project.path, "commit", "-m", "Bump tfm-library pin", "--", SUBMODULE)
    if code != 0:
        return f"FAILED to commit — {err.splitlines()[0] if err else 'unknown error'}"
    return f"pin recorded at {after[:44]} — not pushed"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--projects-dir", default=None,
        help="where the projects live (default: the parent of this repository)",
    )
    _projects.add_selection_arguments(parser)
    parser.add_argument("--update", action="store_true", help="move each working tree")
    parser.add_argument("--commit", action="store_true",
                        help="record each new pin (needs --update)")
    args = parser.parse_args(argv)

    if args.commit and not args.update:
        raise SystemExit("--commit needs --update: there is nothing to record until the tree moves")

    root = Path(args.projects_dir).expanduser().resolve() if args.projects_dir         else _projects.default_root()
    if not root.is_dir():
        raise SystemExit(f"{root} is not a directory")

    print(f"searching {root} (depth {MAX_DEPTH}) for projects with {SUBMODULE}/\n")
    all_projects = [Project(path=p) for p in _projects.find_projects(root)]
    if not all_projects:
        print(f"No project under {root} declares {SUBMODULE}/ in its .gitmodules.")
        return 0

    selected_paths = _projects.select([p.path for p in all_projects], args)
    chosen = [inspect(Project(path=p)) for p in selected_paths]
    if not chosen:
        print("Nothing selected.")
        return 0

    report(chosen, total=len(all_projects))

    if not args.update:
        print("\nNothing changed. Re-run with --update to move the working trees.")
        print(f"Read {SUBMODULE}/CHANGELOG.md in any one project first — moving the pin")
        print("changes which literature a result was checked against.")
        return 0

    print(f"\n{'=' * 74}\nUPDATING\n{'=' * 74}")
    verdicts = [(p.name, update(p, args.commit)) for p in chosen]
    for name, verdict in verdicts:
        print(f"  {name:<26} {verdict}")

    if args.commit:
        print("\nNothing was pushed. Push each project yourself when you are ready.")
    print("Record the new pin in each project's docs/CHANGELOG.md.")
    return 1 if any(v.startswith(("FAILED", "REFUSED")) for _, v in verdicts) else 0


if __name__ == "__main__":
    raise SystemExit(main())
