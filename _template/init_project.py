"""Turn this checkout into a real project. Run it once, right after "Use this template".

    python _template/init_project.py CreditICL
    python _template/init_project.py CreditICL --dry-run
    python _template/init_project.py CreditICL --repo-url https://github.com/me/CreditICL

No copying: this repository already *is* the template, so the whole job is to replace every
`{{PLACEHOLDER}}` in every text file with a real value, and print the steps it deliberately does
not take.

IT RUNS NOTHING ELSE — no install, no git, no push, and it does not delete `_template/` for you.
Each of those reaches outside this directory or throws files away, so each is printed instead.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

#: `parents[1]`, because this file is `<repo>/_template/init_project.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DOC = REPO_ROOT / "docs" / "TEMPLATE.md"

DEFAULT_AUTHOR = "Andreas Goethals"
DEFAULT_EMAIL = "andreas.goethals@kuleuven.be"
DEFAULT_LIBRARY_URL = "https://github.com/andreasgoethals/TFM_Library.git"
GITHUB_OWNER = "andreasgoethals"

#: File types whose text is substituted. The guard is for the day someone adds an image — a
#: binary quietly mangled by a regex surfaces much later. `""` covers dotfiles.
TEXT_SUFFIXES = frozenset(
    {".py", ".md", ".toml", ".yaml", ".yml", ".cff", ".txt", ".ipynb", ".slurm",
     ".sh", ".cfg", ".ini", ""}
)

#: Never touched. `_template/` is about to be deleted, and its own documentation quotes
#: placeholders as examples, which substitution would destroy.
SKIP_DIRS = frozenset({".git", "_template", "tfm-library", ".venv", "venv", "__pycache__",
                       ".pytest_cache", ".ruff_cache", "output", "data"})

#: Upper-case names only, never after a `$`: GitHub Actions expressions look like
#: `${{ matrix.python-version }}`, which a naive substitution would destroy.
PLACEHOLDER_RE = re.compile(r"(?<!\$)\{\{([A-Z][A-Z0-9_]*)\}\}")

NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


def git(*args: str) -> str:
    """Run git and return stdout, or "" on failure. Never raises: this may not be a git checkout
    at all (someone downloaded a zip), and that must not stop the initialiser."""
    try:
        result = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def detected_repo_url() -> str:
    """This checkout's `origin`, as an https URL, or "".

    The default rather than a name-based guess because the new repository is cloned, so `origin`
    is *already* the right answer — including when its name differs from the project name.
    """
    url = git("remote", "get-url", "origin")
    if not url:
        return ""
    # Normalise the SSH form so the value that lands in pyproject.toml is clickable.
    if url.startswith("git@") and ":" in url:
        host, _, path = url.partition(":")
        url = f"https://{host.split('@', 1)[1]}/{path}"
    return url.removesuffix(".git")


def submodule_state() -> tuple[str, str]:
    """(state, advice) for `tfm-library`. Reported, never acted on. Distinguishes the three cases
    that look alike from outside and need different fixes."""
    if not (REPO_ROOT / ".git").exists():
        return "unknown", "not a git checkout yet — `git init`, or clone the fork instead"

    gitlink = git("ls-files", "-s", "tfm-library")
    if not gitlink.startswith("160000"):
        return (
            "MISSING",
            f"no submodule pin recorded. Add it once:\n"
            f"       git submodule add {DEFAULT_LIBRARY_URL} tfm-library",
        )
    library = REPO_ROOT / "tfm-library"
    if library.is_dir() and any(library.iterdir()):
        return "populated", "nothing to do"
    return (
        "pinned, empty",
        "the pin came with the repository; fetch its 749 MB when you want it:\n"
        "       git submodule update --init",
    )


def derive(project: str, args: argparse.Namespace) -> dict[str, str]:
    """Every placeholder value, derived from the project name plus the overrides."""
    today = date.today()
    return {
        "PROJECT_NAME": project,
        # The distribution name: lower-case, alphanumerics only. PEP 508 forbids a leading
        # digit and pip's error for a bad name is obscure, so it is normalised here.
        "PACKAGE_NAME": re.sub(r"[^a-z0-9]", "", project.lower()),
        # Environment-variable prefix, e.g. CREDITICL_STAGING_ROOT.
        "PROJECT_UPPER": re.sub(r"[^A-Z0-9]", "_", project.upper()),
        "AUTHOR": args.author,
        "EMAIL": args.email,
        # One sentence, reused in the README's opening line and pyproject's `description` —
        # so an agent initialises in one command instead of editing two files and getting them
        # slightly different.
        "DESCRIPTION": args.description or f"{project} — PhD research, KU Leuven: machine "
                                           f"learning on tabular data.",
        "YEAR": str(today.year),
        # DD-MM-YYYY: the changelog convention. Not ISO, deliberately — one format everywhere.
        "DATE": today.strftime("%d-%m-%Y"),
        "REPO_URL": args.repo_url or detected_repo_url()
                    or f"https://github.com/{GITHUB_OWNER}/{project}",
        "TFM_LIBRARY_URL": args.library_url,
    }


def text_files() -> list[Path]:
    """Every substitutable file in the repository, `_template/` excluded."""
    found = []
    for path in sorted(REPO_ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(REPO_ROOT).parts
        if SKIP_DIRS & set(relative_parts[:-1]):
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name.startswith("."):
            found.append(path)
    return found


def substitute(values: dict[str, str], *, dry_run: bool) -> tuple[list[str], list[str]]:
    """Fill in every placeholder. Returns (files changed, leftover placeholders)."""
    changed: list[str] = []
    leftovers: list[str] = []
    for path in text_files():
        try:
            original = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue  # binary despite the name; leave it exactly as it is
        replaced = PLACEHOLDER_RE.sub(lambda m: values.get(m.group(1), m.group(0)), original)
        relative = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
        if replaced != original:
            changed.append(relative)
            if not dry_run:
                # newline="" keeps the LF endings .gitattributes normalises to. Python would
                # otherwise write CRLF on Windows, and a CRLF SLURM script fails on the
                # cluster with a bare "$'\\r': command not found" that reads like a real bug.
                path.write_text(replaced, encoding="utf-8", newline="")
        leftovers += [f"{relative}: {{{{{m.group(1)}}}}}" for m in PLACEHOLDER_RE.finditer(replaced)]
    return changed, leftovers


def next_steps(project: str, values: dict[str, str]) -> str:
    """The steps this script deliberately does not take, because each reaches outside."""
    state, advice = submodule_state()
    return f"""
{"=" * 74}
{project} initialised in {REPO_ROOT}
{"=" * 74}

Nothing was installed, committed, pushed, or deleted. Run these yourself, in order.
Windows PowerShell — ONE COMMAND PER LINE, `&&` is a parser error there.

  1. Delete the template-only folder. It is not part of your project.

     Remove-Item -Recurse -Force _template

  2. Environment

     python -m venv .venv
     .\\.venv\\Scripts\\Activate.ps1
     pip install -e ".[dev]"

  3. The literature submodule — READ-ONLY in every project
     state: {state} — {advice}

     Copy-Item tfm-library\\PROJECT_SPECIFIC.template.md tfm-library\\PROJECT_SPECIFIC.md

  4. Check it runs.

     python -m pytest -q
     python -m src.utils.run_notebooks

  5. Make it yours

     README.md          replace everything ABOVE the last chapter. Keep that chapter.
     docs/VSC.md        fill in the TODOs: partition, walltime, credit account.
     scripts/slurm/job.slurm    the same TODOs.
     config/                    real configs; delete example.yaml.
     src/visualize/style.py     register this project's series names, once.
     notebooks/example_analysis.ipynb   the pattern to copy, then delete.
     docs/CHANGELOG.md          the first entry is already dated {values["DATE"]}.

  6. First commit

     git add -A
     git commit -m "Initialise {project} from the repository template"

docs/TEMPLATE.md is a starting point, not a contract. Deviate where the work needs it — but
deliberately, and say so. Generic rule changes belong at the template source.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("project", help="the project name, e.g. CreditICL")
    parser.add_argument(
        "--description", default=None,
        help="one sentence: what the project is and what question it answers. Lands in "
             "README.md and pyproject.toml.",
    )
    parser.add_argument("--author", default=DEFAULT_AUTHOR)
    parser.add_argument("--email", default=DEFAULT_EMAIL)
    parser.add_argument("--repo-url", default=None,
                        help="default: this checkout's `origin`, else "
                             f"https://github.com/{GITHUB_OWNER}/<project>")
    parser.add_argument("--library-url", default=DEFAULT_LIBRARY_URL)
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = parser.parse_args(argv)

    if not NAME_RE.match(args.project):
        # The name becomes a directory on two cluster tiers, a distribution name and an
        # environment-variable prefix. A space or a dot fails in one of the three, usually last.
        raise SystemExit(
            f"invalid project name {args.project!r}: start with a letter, then letters, "
            f"digits, '-' or '_' only."
        )
    if not TEMPLATE_DOC.is_file():
        raise SystemExit(f"{TEMPLATE_DOC} is missing — it is the governing document.")
    values = derive(args.project, args)
    print(f"{'DRY RUN — nothing will be written' if args.dry_run else 'initialising'} "
          f"in {REPO_ROOT}\n")
    for key, value in values.items():
        print(f"  {key:<18} {value}")

    changed, leftovers = substitute(values, dry_run=args.dry_run)
    print(f"\n{len(changed)} files updated:")
    for relative in changed:
        print(f"  {relative}")

    if leftovers:
        # A template bug, not a user error: a key exists that `derive()` does not produce.
        print("\nWARNING: unsubstituted placeholders remain — this is a template bug:")
        for item in leftovers:
            print(f"  {item}")

    if args.dry_run:
        print("\nDry run. Re-run without --dry-run to apply.")
        return 0

    print(next_steps(args.project, values))
    return 1 if leftovers else 0


if __name__ == "__main__":
    sys.exit(main())
