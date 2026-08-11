"""Push a changed rule down into every existing project. Reports by default.

    python _template/sync_template_rules.py                    what is out of date, where
    python _template/sync_template_rules.py --diff             show what actually differs
    python _template/sync_template_rules.py --apply            update docs/TEMPLATE.md + hash
    python _template/sync_template_rules.py --apply --also scripts/check.py
    python _template/sync_template_rules.py --only CreditICL --apply

WHY. A project made with "Use this template" has no upstream to merge from, so a rule change
would otherwise be propagated by hand: copy `docs/TEMPLATE.md`, re-compute its SHA-256, paste it
into that project's compliance test, repeat. Four steps, five repositories, and the fourth is the
one that gets skipped. This is why forking is unnecessary — `git pull upstream main` looks like
it solves the same problem, but the files a template change touches most are exactly the ones
`init_project.py` rewrote per project, so every pull conflicts where you least want it.

TWO TIERS, split by who owns a file after initialisation.

  APPLIED   `docs/TEMPLATE.md` plus its hash. Safe to overwrite unasked: the template owns it
            verbatim, and the compliance test already proves the project has not edited it — if
            it had, that project's suite would be red.
  REPORTED  every other file the template ships unchanged. Shown as differing, never written: a
            project may *extend* some (a new cleanup category) and is *expected* to delete
            others (the example notebook). `--diff` to look, `--also <path>` to take one.

IT NEVER COMMITS AND NEVER PUSHES. It writes files; `git diff` afterwards is the review step.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import re
from pathlib import Path

import _projects

REPO_ROOT = Path(__file__).resolve().parents[1]

COMPLIANCE_TEST = Path("tests") / "test_template_compliance.py"
TEMPLATE_DOC = Path("docs") / "TEMPLATE.md"

#: Overwritten by `--apply`. The template owns these byte for byte.
APPLIED = (TEMPLATE_DOC,)

#: Reported when they differ, never written without `--also`. A difference means either the
#: template moved on or the project extended it, and this tool cannot tell which.
REPORTED = (
    Path(".gitattributes"),
    Path(".vscode") / "settings.json",
    Path(".vscode") / "extensions.json",
    Path(".github") / "workflows" / "check.yml",
    Path("scripts") / "check.py",
    Path("scripts") / "clean_run.py",
    Path("scripts") / "run_notebooks.py",
    Path("scripts") / "update_tfm_library.py",
    Path("src") / "utils" / "config.py",
    Path("src") / "utils" / "logging_setup.py",
    Path("src") / "utils" / "run_artifacts.py",
    Path("src") / "utils" / "run_notebooks.py",
    Path("src") / "visualize" / "figures.py",
    Path("tests") / "conftest.py",
    Path("tests") / "test_config.py",
    Path("tests") / "test_figures.py",
    Path("tests") / "test_paths.py",
    Path("tests") / "test_run_artifacts.py",
    Path("tests") / "test_run_notebooks.py",
    Path("tests") / "test_style.py",
    COMPLIANCE_TEST,
)

#: Never compared: each holds a substituted placeholder, a per-project registry, or content the
#: project replaces outright, so "differs from the template" is the correct state.
PROJECT_OWNED = (
    "README.md", "AGENTS.md", "LICENSE", "CITATION.cff", "pyproject.toml", ".gitignore",
    ".gitmodules", "docs/CHANGELOG.md", "docs/AGENTS_MEMORY.md", "docs/VSC.md",
    "src/__init__.py", "src/utils/paths.py", "src/visualize/style.py", "src/data/loaders.py",
    "scripts/slurm/job.slurm", "config/example.yaml", "notebooks/example_analysis.ipynb",
)

HASH_LINE_RE = re.compile(r'^(EXPECTED_TEMPLATE_SHA256\s*=\s*)".*"$', re.M)


def read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def compare(project: Path, relative: Path) -> str:
    """`same`, `differs`, `missing in project`, or `unreadable`."""
    theirs = read(project / relative)
    if theirs is None:
        return "missing in project" if not (project / relative).exists() else "unreadable"
    ours = read(REPO_ROOT / relative)
    if ours is None:
        return "missing in template"
    if relative == COMPLIANCE_TEST:
        # The hash line differs BY DESIGN — the one project-specific line in a shared file.
        # Blank it on both sides so the comparison is about the checks.
        ours = HASH_LINE_RE.sub(r'\1"<hash>"', ours)
        theirs = HASH_LINE_RE.sub(r'\1"<hash>"', theirs)
    return "same" if ours == theirs else "differs"


def rebake_hash(project: Path, sha: str) -> str:
    """Write the new template hash into this project's compliance test."""
    target = project / COMPLIANCE_TEST
    text = read(target)
    if text is None:
        return f"cannot read {COMPLIANCE_TEST}"
    updated = HASH_LINE_RE.sub(rf'\1"{sha}"', text, count=1)
    if updated == text:
        # Already correct, or the line is missing entirely. Both are worth saying.
        return "hash already current" if sha in text else "no EXPECTED_TEMPLATE_SHA256 line found"
    target.write_text(updated, encoding="utf-8", newline="")
    return "hash re-baked"


def show_diff(project: Path, relative: Path) -> None:
    ours = (read(REPO_ROOT / relative) or "").splitlines(keepends=True)
    theirs = (read(project / relative) or "").splitlines(keepends=True)
    lines = list(difflib.unified_diff(
        theirs, ours,
        fromfile=f"{project.name}/{relative.as_posix()}",
        tofile=f"template/{relative.as_posix()}",
        n=2,
    ))
    if not lines:
        return
    print(f"\n    --- {relative.as_posix()} ---")
    # Capped: the point is to decide whether to look properly, not to review it in a terminal.
    for line in lines[:40]:
        print(f"    {line.rstrip()}")
    if len(lines) > 40:
        print(f"    ... {len(lines) - 40} more diff lines")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--projects-dir", default=None,
                        help="where the projects live (default: the parent of this repository)")
    _projects.add_selection_arguments(parser)
    parser.add_argument("--apply", action="store_true",
                        help=f"write {TEMPLATE_DOC.as_posix()} and re-bake the hash")
    parser.add_argument("--also", nargs="+", default=(), metavar="PATH",
                        help="additionally apply these reported files, e.g. scripts/check.py")
    parser.add_argument("--diff", action="store_true", help="show what differs")
    args = parser.parse_args(argv)

    template_doc = REPO_ROOT / TEMPLATE_DOC
    if not template_doc.is_file():
        raise SystemExit(f"{template_doc} is missing — it is the governing document.")
    sha = hashlib.sha256(template_doc.read_bytes()).hexdigest()

    also = tuple(Path(p) for p in args.also)
    unknown = [p.as_posix() for p in also if p not in REPORTED]
    if unknown:
        raise SystemExit(
            f"--also {unknown} is not a file the template owns. Choose from:\n  "
            + "\n  ".join(p.as_posix() for p in REPORTED)
        )

    root = Path(args.projects_dir).expanduser().resolve() if args.projects_dir \
        else _projects.default_root()
    print(f"template: {REPO_ROOT}")
    print(f"docs/TEMPLATE.md sha256: {sha}")
    print(f"searching {root} for projects\n")

    # The template turns up in the walk too; copying TEMPLATE.md onto itself is a no-op, but
    # reporting it as a project is confusing.
    found = [p for p in _projects.find_projects(root) if p.resolve() != REPO_ROOT.resolve()]
    if not found:
        print(f"No project under {root} to update.")
        return 0

    chosen = _projects.select(found, args)
    if not chosen:
        print("Nothing selected.")
        return 0

    failures = 0
    for project in chosen:
        print(f"{'=' * 74}\n{project.name}\n{'=' * 74}")

        doc_state = compare(project, TEMPLATE_DOC)
        print(f"  {TEMPLATE_DOC.as_posix():<44} {doc_state}")

        differing = [r for r in REPORTED if compare(project, r) == "differs"]
        for relative in REPORTED:
            state = compare(project, relative)
            if state != "same":
                print(f"  {relative.as_posix():<44} {state}")
        if not differing and doc_state == "same":
            print("  up to date with the template.")

        if args.diff:
            for relative in ([TEMPLATE_DOC] if doc_state == "differs" else []) + differing:
                show_diff(project, relative)

        if not args.apply:
            continue

        print("  applying:")
        for relative in (*APPLIED, *also):
            source = REPO_ROOT / relative
            target = project / relative
            if not source.is_file():
                print(f"    SKIP  {relative.as_posix()} — not in the template")
                failures += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            print(f"    wrote {relative.as_posix()}")
        print(f"    {rebake_hash(project, sha)}")

    print(f"\n{'=' * 74}")
    if not args.apply:
        print("Nothing was written. Re-run with --apply to update docs/TEMPLATE.md and the hash.")
        print("Use --diff first if you want to see what changed.")
    else:
        print("Files written; nothing was committed or pushed. In each project:")
        print("  git diff                      review it")
        print("  python scripts/check.py       the compliance test must pass")
        print("  note the rule change in that project's docs/CHANGELOG.md")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
