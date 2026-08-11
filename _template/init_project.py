"""Turn this checkout into a real project. Run it once, right after "Use this template".

    python _template/init_project.py CreditICL
    python _template/init_project.py CreditICL --dry-run
    python _template/init_project.py CreditICL --repo-url https://github.com/me/CreditICL

WHAT IT DOES. There is no copying — this repository already *is* the template, so the whole
job is filling in the blanks:

  1. replaces every `{{PLACEHOLDER}}` in every text file with a real value
  2. hashes `docs/TEMPLATE.md` and writes that hash into
     `tests/test_template_compliance.py`, which is what makes a later edit to the template
     fail the test suite
  3. prints the steps it deliberately does NOT take

WHY THE HASH IS BAKED IN rather than fetched from the template repository at test time: the
compliance test has to run on a fresh clone with no network and no submodule, so the
expected value has to already be in the file.

IT RUNS NOTHING ELSE. No `pip install`, no `git` command, no push, and it does not delete
`_template/` for you — each of those changes something outside this directory or throws away
files, so each is printed for you to run.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

#: `parents[1]`, because this file is `<repo>/_template/init_project.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DOC = REPO_ROOT / "docs" / "TEMPLATE.md"
TEMPLATE_ONLY_DIR = REPO_ROOT / "_template"

DEFAULT_AUTHOR = "Andreas Goethals"
DEFAULT_EMAIL = "andreas.goethals@kuleuven.be"
DEFAULT_LIBRARY_URL = "https://github.com/andreasgoethals/TFM_Library.git"
GITHUB_OWNER = "andreasgoethals"

#: File types whose text is substituted. Everything the template ships is text; the guard is
#: for the day someone adds an image — a binary file quietly mangled by a regex is a bug that
#: surfaces much later. The empty string covers dotfiles like `.gitignore`, whose whole name
#: pathlib reports as the stem.
TEXT_SUFFIXES = frozenset(
    {".py", ".md", ".toml", ".yaml", ".yml", ".cff", ".txt", ".ipynb", ".slurm",
     ".sh", ".cfg", ".ini", ""}
)

#: Never touched. `.git` is obvious; `_template/` is about to be deleted and its own
#: documentation quotes placeholders as examples, which substitution would destroy.
SKIP_DIRS = frozenset({".git", "_template", "tfm-library", ".venv", "venv", "__pycache__",
                       ".pytest_cache", ".ruff_cache", "output", "data"})

#: `{{NAME}}` for upper-case names only, and never after a `$`. WHY: GitHub Actions
#: expressions look like `${{ matrix.python-version }}`, and a naive `{{...}}` substitution
#: would destroy them and a naive leftover-check would flag them.
PLACEHOLDER_RE = re.compile(r"(?<!\$)\{\{([A-Z][A-Z0-9_]*)\}\}")

NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")

#: Where the template hash lives, and the line that holds it.
COMPLIANCE_TEST = REPO_ROOT / "tests" / "test_template_compliance.py"
HASH_LINE_RE = re.compile(r'^(EXPECTED_TEMPLATE_SHA256\s*=\s*)".*"$', re.M)


def git(*args: str) -> str:
    """Run git in the repository and return stdout, or "" if it fails.

    Never raises: this repository may not be a git checkout at all (someone downloaded a
    zip), and that must not stop the initialiser.
    """
    try:
        result = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def detected_repo_url() -> str:
    """This checkout's own `origin`, as an https URL, or "".

    WHY this is the default rather than a name-based guess: the intended workflow is to fork
    the template on GitHub and clone the fork, so `origin` is *already* the correct answer —
    and it is right even when the repository name differs from the project name, which a
    guess built from the project name never is.
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
    """(state, advice) for `tfm-library`. Reported, never acted on.

    Distinguishes the three cases that look alike from the outside and need different fixes:
    the pin is missing entirely, the pin is there but the folder is empty, or it is populated.
    """
    if not (REPO_ROOT / ".git").exists():
        return "unknown", "not a git checkout yet — `git init`, or clone the fork instead"

    gitlink = git("ls-files", "-s", "tfm-library")
    if not gitlink.startswith("160000"):
        return (
            "MISSING",
            f"no submodule pin recorded. Add it once:\n"
            f"       git submodule add {DEFAULT_LIBRARY_URL} tfm-library",
        )
    if any((REPO_ROOT / "tfm-library").iterdir()) if (REPO_ROOT / "tfm-library").is_dir() else False:
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
        # One sentence, reused in three places that all want the same answer: the README's
        # opening line, pyproject's `description`, and CITATION.cff's abstract. Passing it
        # here is what lets an agent initialise the repository in a single command instead of
        # editing three files afterwards and getting two of them slightly different.
        "DESCRIPTION": args.description or f"{project} — PhD research, KU Leuven: machine "
                                           f"learning on tabular data.",
        "YEAR": str(today.year),
        # DD-MM-YYYY: the changelog convention. Not ISO, deliberately — one format everywhere.
        "DATE": today.strftime("%d-%m-%Y"),
        # ISO only for CITATION.cff, whose format requires it.
        "DATE_ISO": today.isoformat(),
        "REPO_URL": args.repo_url or detected_repo_url()
                    or f"https://github.com/{GITHUB_OWNER}/{project}",
        "TFM_LIBRARY_URL": args.library_url,
        "TEMPLATE_SHA256": hashlib.sha256(TEMPLATE_DOC.read_bytes()).hexdigest(),
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


def bake_hash(sha: str, *, dry_run: bool) -> bool:
    """Write the template hash into the compliance test. Returns whether it changed."""
    text = COMPLIANCE_TEST.read_text(encoding="utf-8")
    updated = HASH_LINE_RE.sub(rf'\1"{sha}"', text, count=1)
    if updated == text:
        return False
    if not dry_run:
        COMPLIANCE_TEST.write_text(updated, encoding="utf-8", newline="")
    return True


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

     A rule change reaches this project later via the template's own
     `_template/sync_template_rules.py`, not from anything in here.

  2. Environment

     python -m venv .venv
     .\\.venv\\Scripts\\Activate.ps1
     pip install -e ".[dev]"

  3. The literature submodule — READ-ONLY in every project
     state: {state} — {advice}

     Copy-Item tfm-library\\PROJECT_SPECIFIC.template.md tfm-library\\PROJECT_SPECIFIC.md

  4. Is it healthy? This must pass before the first commit.

     python scripts/check.py

  5. Make it yours

     README.md          replace everything ABOVE the last chapter. Keep that chapter.
     docs/VSC.md        fill in the TODOs: partition, walltime, credit account.
     scripts/slurm/job.slurm    the same TODOs.
     config/example.yaml        copy per experiment; delete the example.
     src/visualize/style.py     register this project's series names, once.
     notebooks/example_analysis.ipynb   the pattern to copy, then delete.
     docs/CHANGELOG.md          the first entry is already dated {values["DATE"]}.

  6. First commit

     git add -A
     git commit -m "Initialise {project} from repo-template"

Do NOT edit docs/TEMPLATE.md. Its SHA-256 is now baked into
tests/test_template_compliance.py, so an edit fails the test suite — which is the point.
Change a rule at the template source and pull it down.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("project", help="the project name, e.g. CreditICL")
    parser.add_argument(
        "--description", default=None,
        help="one sentence: what the project is and what question it answers. Lands in "
             "README.md, pyproject.toml and CITATION.cff.",
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
        # environment-variable prefix. A space or a dot fails in one of those three, usually
        # the last, long after this script ran.
        raise SystemExit(
            f"invalid project name {args.project!r}: start with a letter, then letters, "
            f"digits, '-' or '_' only."
        )
    if not TEMPLATE_DOC.is_file():
        raise SystemExit(f"{TEMPLATE_DOC} is missing — it is the governing document.")
    if not COMPLIANCE_TEST.is_file():
        raise SystemExit(f"{COMPLIANCE_TEST} is missing — it is where the hash goes.")

    values = derive(args.project, args)
    print(f"{'DRY RUN — nothing will be written' if args.dry_run else 'initialising'} "
          f"in {REPO_ROOT}\n")
    for key, value in values.items():
        print(f"  {key:<18} {value}")

    changed, leftovers = substitute(values, dry_run=args.dry_run)
    print(f"\n{len(changed)} files updated:")
    for relative in changed:
        print(f"  {relative}")

    if bake_hash(values["TEMPLATE_SHA256"], dry_run=args.dry_run):
        print(f"\ntemplate hash baked into {COMPLIANCE_TEST.relative_to(REPO_ROOT)}")
    else:
        # Already baked means this script has run before, which is worth saying out loud:
        # a second run is harmless but it also did nothing, and silence would look like success.
        print("\ntemplate hash was already up to date — has this already been initialised?")

    if leftovers:
        # A leftover is a template bug, not a user error: a key exists in a file that
        # `derive()` does not produce.
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
