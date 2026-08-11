"""The template's rules, enforced. This is what stops `docs/TEMPLATE.md` being advice.

Every check here corresponds to a rule in `docs/TEMPLATE.md` § The compliance test. If a
check and the document disagree, the document is right and this file is a bug.

TWO PROPERTIES THIS FILE DELIBERATELY HAS:

1. **It never imports the project.** It reads files. So it runs on a fresh clone, before
   `pip install -e .`, before the submodule is populated, and inside CI with nothing set
   up — which is exactly when a structural mistake is cheapest to find.
2. **It fails rather than warns.** A warning in a test suite is a line of output nobody
   reads. Every rule below is either enforced or is not a rule.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Rule 1 — docs/TEMPLATE.md is the template, byte for byte.
# ---------------------------------------------------------------------------

#: SHA-256 of the template as it stood when this repository was initialised. Written by
#: `_template/init_project.py`; the placeholder means it has not run yet.
EXPECTED_TEMPLATE_SHA256 = "{{TEMPLATE_SHA256}}"

TEMPLATE_DOC = REPO_ROOT / "docs" / "TEMPLATE.md"
#: A root copy, if this repository keeps one. Compared only when it exists.
ROOT_TEMPLATE_DOC = REPO_ROOT / "TEMPLATE.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_uninitialised_template() -> bool:
    """True when this is still the template repository, name not yet filled in.

    WHY the signal is the placeholder in `src/utils/paths.py` and not the presence of
    `_template/`: a project that ran the initialiser but forgot to delete that folder must
    still have its template hash enforced. The project name is what the initialiser writes,
    so its placeholder is the one honest marker of "no project here yet".

    Read as text rather than imported, because this file never imports the project.

    The pattern's braces are backslash-escaped, which also keeps the initialiser from
    substituting this very line: it looks for two literal adjacent braces, and `\\{\\{` has a
    backslash between them.
    """
    paths_module = REPO_ROOT / "src" / "utils" / "paths.py"
    if not paths_module.is_file():
        return False
    text = paths_module.read_text(encoding="utf-8")
    return bool(re.search(r"PROJECT_NAME\s*=\s*[\"']\{\{[A-Z_]+\}\}[\"']", text))


def test_template_doc_exists() -> None:
    assert TEMPLATE_DOC.is_file(), (
        f"{TEMPLATE_DOC} is missing. It is the governing document for this repository's "
        f"structure; copy it from the template source."
    )


def test_template_doc_is_unmodified() -> None:
    """`docs/TEMPLATE.md` must hash to the template it was copied from.

    WHY a hash and not a diff: the rule is not "roughly the same", it is "the same file".
    The template is shared verbatim across every project, so a local edit either gets
    overwritten on the next sync or quietly makes this repository's rules diverge from
    everyone else's. Project-specific rules go in README.md or a new docs/<NAME>.md.
    """
    if is_uninitialised_template():
        pytest.skip(
            "this is the un-initialised template repository — docs/TEMPLATE.md IS the source, "
            "so there is nothing to compare it against. The check switches on as soon as "
            "`python _template/init_project.py <ProjectName>` has run."
        )
    if EXPECTED_TEMPLATE_SHA256.startswith("{{"):
        pytest.fail(
            "EXPECTED_TEMPLATE_SHA256 is still the placeholder, but this repository has a "
            "project name — so the initialiser ran without baking the hash, or the line was "
            "reverted. Re-run `python _template/init_project.py <ProjectName>`."
        )
    actual = sha256(TEMPLATE_DOC)
    assert actual == EXPECTED_TEMPLATE_SHA256, (
        f"docs/TEMPLATE.md has been edited.\n"
        f"  expected sha256 {EXPECTED_TEMPLATE_SHA256}\n"
        f"  actual   sha256 {actual}\n"
        f"Revert it. Change the rule at the template source and pull it down instead."
    )


def test_template_doc_matches_root_copy() -> None:
    """When a root `TEMPLATE.md` exists, `docs/TEMPLATE.md` must be identical to it."""
    if not ROOT_TEMPLATE_DOC.is_file():
        pytest.skip("this repository keeps no root TEMPLATE.md — docs/TEMPLATE.md is the only copy")
    assert sha256(TEMPLATE_DOC) == sha256(ROOT_TEMPLATE_DOC), (
        "docs/TEMPLATE.md and TEMPLATE.md differ. They are the same document; one of the "
        "two has been edited in place."
    )


# ---------------------------------------------------------------------------
# Rule 2 — the structure exists.
# ---------------------------------------------------------------------------

REQUIRED_DIRS = (
    "config",
    "data/raw",
    "data/processed",
    "docs",
    "notebooks",
    "output",
    "output/figures",
    "output/logs",
    "output/manifests",
    "output/results",
    "scripts",
    "scripts/slurm",
    "src",
    "src/data",
    "src/utils",
    "src/visualize",
    "tests",
)

REQUIRED_FILES = (
    ".gitattributes",
    ".gitignore",
    ".gitmodules",
    "AGENTS.md",
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "docs/TEMPLATE.md",
    "docs/CHANGELOG.md",
    "docs/AGENTS_MEMORY.md",
    "docs/VSC.md",
    "src/utils/paths.py",
    "src/utils/run_notebooks.py",
    "src/utils/run_artifacts.py",
    "src/visualize/style.py",
    "src/visualize/figures.py",
    "scripts/check.py",
    "scripts/clean_run.py",
    "tests/test_template_compliance.py",
)


@pytest.mark.parametrize("relative", REQUIRED_DIRS)
def test_required_directory_exists(relative: str) -> None:
    assert (REPO_ROOT / relative).is_dir(), (
        f"required directory {relative}/ is missing. The empty ones are kept by a tracked "
        f".gitkeep so a fresh clone still has somewhere to write."
    )


@pytest.mark.parametrize("relative", REQUIRED_FILES)
def test_required_file_exists(relative: str) -> None:
    assert (REPO_ROOT / relative).is_file(), f"required file {relative} is missing"


def test_literature_submodule_is_declared() -> None:
    """`tfm-library/` must be registered as a submodule.

    The DIRECTORY is not required to be populated: it is empty until
    `git submodule update --init`, and CI checks out without submodules on purpose. What
    must be true is that the repository declares it, so every consumer is pinned to one
    exact commit of the literature.
    """
    gitmodules = (REPO_ROOT / ".gitmodules").read_text(encoding="utf-8")
    assert "tfm-library" in gitmodules, (
        ".gitmodules does not declare tfm-library. Every project carries the shared "
        "literature library as a pinned, read-only submodule:\n"
        "  git submodule add <library repo url> tfm-library"
    )


# ---------------------------------------------------------------------------
# Rule 3 — nothing generated is written outside output/.
# ---------------------------------------------------------------------------

#: Extensions and names that only ever exist because code produced them.
GENERATED_SUFFIXES = frozenset(
    {".pdf", ".png", ".svg", ".eps", ".csv", ".tsv", ".log", ".jsonl", ".npy", ".npz",
     ".pkl", ".pickle", ".joblib", ".ckpt", ".pt", ".pth", ".safetensors", ".parquet"}
)
GENERATED_NAMES = frozenset({"CAPTIONS.md", "All_Results.md", "_figures.json", "_stdout.txt"})

#: Where a generated-looking file is legitimate.
#:   output/       the one place generated files belong
#:   data/         inputs and the processed cache — not generated *by a run*
#:   checkpoints/  weights: downloaded, or written by a training run to its own tier
#:   tfm-library/  the literature submodule; full of PDFs, and not ours
#:   docs/         hand-written documentation may include a diagram
#:   tests/        fixtures
GENERATED_ALLOWED_ROOTS = ("output", "data", "checkpoints", "tfm-library", "docs", "tests")

SKIP_TREE_PARTS = frozenset(
    {".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache", ".ruff_cache",
     ".mypy_cache", ".ipynb_checkpoints", "node_modules", ".idea", ".vscode",
     "site-packages", ".eggs", "build", "dist"}
)


def _walk_repo() -> list[Path]:
    """Every file in the working tree, minus caches, environments and the submodule."""
    return [
        p for p in REPO_ROOT.rglob("*")
        if p.is_file() and not (SKIP_TREE_PARTS & set(p.relative_to(REPO_ROOT).parts))
    ]


def test_no_generated_files_outside_output() -> None:
    """A generated artefact anywhere but `output/` means some code wrote to the wrong place.

    This is the check that catches it after the fact, and it is the one that never gives a
    false negative: whatever path the code built, the file is either under `output/` or it
    is not. `output/` is the one root, so "what did this run produce?" and "what can I
    delete?" each have exactly one answer.
    """
    offenders = []
    for path in _walk_repo():
        rel = path.relative_to(REPO_ROOT)
        if rel.parts[0] in GENERATED_ALLOWED_ROOTS:
            continue
        if path.suffix.lower() in GENERATED_SUFFIXES or path.name in GENERATED_NAMES:
            offenders.append(str(rel).replace("\\", "/"))
    assert not offenders, (
        "generated files found outside output/:\n  "
        + "\n  ".join(sorted(offenders))
        + "\nEverything the code produces goes under output/. Build the destination with "
          "src/utils/paths.py instead of a literal path."
    )


#: Calls whose FIRST argument is the destination path. Chosen deliberately narrow: a check
#: that guesses produces false positives, and a compliance test people disable is worse
#: than no compliance test.
WRITE_CALLS_ARG0 = frozenset(
    {"open", "savefig", "to_csv", "to_parquet", "to_json", "to_pickle", "to_feather",
     "savetxt", "imsave", "write_image"}
)

#: `src/utils/paths.py` is the resolver — naming other roots is its entire job. Tests write
#: to `tmp_path`. Both are exempt by design, not by oversight.
LITERAL_PATH_EXEMPT = ("src/utils/paths.py",)


def _source_files() -> list[Path]:
    files = []
    for folder in ("src", "scripts"):
        files += [
            p for p in sorted((REPO_ROOT / folder).rglob("*.py"))
            if not (SKIP_TREE_PARTS & set(p.parts))
        ]
    return files


def _literal(node: ast.expr) -> str | None:
    """The string value of a literal argument, or None if it is not a literal."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):  # f-string: the literal parts are enough to judge
        return "".join(v.value for v in node.values if isinstance(v, ast.Constant))
    return None


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _is_write_open(node: ast.Call) -> bool:
    """`open(...)` in a writing mode. A read-mode open is not our business."""
    if _call_name(node) != "open":
        return False
    mode = None
    if len(node.args) > 1:
        mode = _literal(node.args[1])
    for kw in node.keywords:
        if kw.arg == "mode":
            mode = _literal(kw.value)
    return bool(mode) and any(c in mode for c in "wax")


def test_no_hardcoded_write_paths_outside_output() -> None:
    """A write call with a literal destination must name `output/` — or not be literal.

    WHY this and not only the filesystem check above: a path assembled at the call site is
    correct on a laptop and wrong on the cluster, where `output/` is on a different tier
    entirely. The filesystem check catches it after a local run; this catches it before
    anyone has run it on the cluster, which is where it costs a job.
    """
    offenders = []
    for path in _source_files():
        rel = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
        if rel in LITERAL_PATH_EXEMPT:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            name = _call_name(node)
            if name not in WRITE_CALLS_ARG0:
                continue
            if name == "open" and not _is_write_open(node):
                continue
            literal = _literal(node.args[0])
            # No literal means the destination came from somewhere — normally paths.py.
            # A literal with no separator and no extension is not a path (a format name,
            # a buffer, a mode).
            if literal is None or ("/" not in literal and "\\" not in literal and "." not in literal):
                continue
            normalised = literal.replace("\\", "/").lstrip("./")
            if normalised.startswith("output/"):
                continue
            offenders.append(f"{rel}:{node.lineno}  {name}({literal!r})")
    assert not offenders, (
        "write calls with a hard-coded destination outside output/:\n  "
        + "\n  ".join(offenders)
        + "\nAsk src/utils/paths.py for the destination — it is the only module that builds "
          "a path, because on the cluster output/ is not where it is locally."
    )


# ---------------------------------------------------------------------------
# Rule 4 — everything in scripts/ is runnable.
# ---------------------------------------------------------------------------


def _scripts() -> list[Path]:
    """`.py` files directly in `scripts/`. `slurm/` holds shell scripts, not Python."""
    return sorted(p for p in (REPO_ROOT / "scripts").glob("*.py") if p.name != "__init__.py")


def test_scripts_directory_is_not_empty() -> None:
    assert _scripts(), "scripts/ has no Python entry points; at least check.py must be there"


@pytest.mark.parametrize("script", _scripts(), ids=lambda p: p.name)
def test_script_is_runnable(script: Path) -> None:
    """Every file in `scripts/` is an ENTRY POINT, so it has a `__main__` block.

    WHY: a module in `scripts/` cannot be imported or tested — that is the point of the
    directory. Something importable that lives there is untestable by construction, and it
    belongs in `src/`.
    """
    tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    has_main = any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        for node in tree.body
    )
    assert has_main, (
        f"scripts/{script.name} has no `if __name__ == \"__main__\":` block. Only runnable "
        f"entry points live in scripts/; anything importable belongs in src/."
    )


# ---------------------------------------------------------------------------
# Rules 5 and 6 — notebooks are thin, and they end by printing.
# ---------------------------------------------------------------------------


def _notebooks() -> list[Path]:
    return sorted(
        p for p in (REPO_ROOT / "notebooks").rglob("*.ipynb")
        if ".ipynb_checkpoints" not in p.parts
    )


def _code_cells(notebook: Path) -> list[str]:
    cells = json.loads(notebook.read_text(encoding="utf-8")).get("cells", [])
    return ["".join(c.get("source", [])) for c in cells if c.get("cell_type") == "code"]


@pytest.mark.parametrize("notebook", _notebooks(), ids=lambda p: p.stem)
def test_notebook_defines_no_logic(notebook: Path) -> None:
    """A notebook contains no `def` and no `class`. All logic lives in `src/`.

    WHY: a function defined in a notebook cannot be imported, cannot be tested, and cannot
    be reused by the next notebook — so it gets copied, and then the two copies diverge and
    two figures disagree for a reason nobody can find.
    """
    offenders = []
    for i, source in enumerate(_code_cells(notebook), start=1):
        for lineno, line in enumerate(source.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith(("def ", "async def ", "class ")):
                offenders.append(f"cell {i}, line {lineno}: {stripped[:70]}")
    assert not offenders, (
        f"{notebook.name} defines logic in the notebook:\n  " + "\n  ".join(offenders)
        + "\nMove it into src/ and import it. A notebook only calls."
    )


@pytest.mark.parametrize("notebook", _notebooks(), ids=lambda p: p.stem)
def test_notebook_ends_by_printing_a_summary(notebook: Path) -> None:
    """The LAST code cell prints a text summary of everything the notebook showed.

    WHY: `output/All_Results.md` is assembled from what the notebooks print. A notebook
    that ends on a plot contributes nothing to it, so the run's findings exist only as
    images — unreadable in a diff, ungreppable, and invisible to an agent.
    """
    cells = [c for c in _code_cells(notebook) if c.strip()]
    assert cells, f"{notebook.name} has no non-empty code cells"
    last = cells[-1]
    assert "print(" in last, (
        f"{notebook.name}'s last code cell does not print anything:\n"
        f"---\n{last.strip()[:300]}\n---\n"
        f"Every notebook ends by printing a text summary of what it showed; that text is "
        f"what output/All_Results.md is built from."
    )


# ---------------------------------------------------------------------------
# Rule 7 — .gitignore rules that must be root-anchored are.
# ---------------------------------------------------------------------------

#: Directory names that occur at more than one depth in the template's tree. A bare
#: `figures/` matches `output/figures/` as well as a root `figures/`, so for these names an
#: unanchored rule silently ignores files that are supposed to be tracked.
AMBIGUOUS_DIR_NAMES = frozenset(
    {"figures", "logs", "manifests", "results", "runs", "raw", "processed", "data",
     "output", "config", "docs", "src", "tests", "notebooks", "scripts", "slurm",
     "checkpoints", "utils", "visualize"}
)


def test_gitignore_ambiguous_rules_are_anchored() -> None:
    """A rule naming an ambiguous directory must be anchored or explicitly pathed.

    Git matches a pattern with no slash in it against that name at EVERY depth. So a bare
    `results/` intended for a root directory also hides `output/results/`, and the small
    result summaries that are supposed to be committed silently never reach git — noticed
    only when a clone comes up empty. Anchor with a leading slash, or write the full
    relative path.

    Cheap check for a single case: `git check-ignore -v <the path you expect tracked>`.
    """
    lines = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    offenders = []
    for lineno, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pattern = line[1:] if line.startswith("!") else line
        # A pattern containing a slash anywhere is already path-scoped, and git treats it
        # as relative to the .gitignore's directory — which is what we want.
        if "/" in pattern.rstrip("/"):
            continue
        name = pattern.rstrip("/")
        if name in AMBIGUOUS_DIR_NAMES:
            offenders.append(f"line {lineno}: {line!r} -> anchor it as '/{line.lstrip('!')}'")
    assert not offenders, (
        "unanchored .gitignore rules that also match deeper directories:\n  "
        + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Further rules from the document, cheap enough to enforce here.
# ---------------------------------------------------------------------------


def test_docs_holds_only_capitalised_markdown() -> None:
    """`docs/` holds only `.md` files, named in CAPITALS.

    One convention means a documentation file is recognisable as one from its name alone,
    and it keeps code, notebooks and data out of a directory that is meant to be readable
    top to bottom.
    """
    offenders = []
    for path in sorted((REPO_ROOT / "docs").rglob("*")):
        if path.is_dir() or ".ipynb_checkpoints" in path.parts:
            continue
        rel = path.relative_to(REPO_ROOT / "docs")
        if path.suffix != ".md":
            offenders.append(f"{rel} — not a .md file")
        elif path.stem != path.stem.upper():
            offenders.append(f"{rel} — name is not in CAPITALS")
    assert not offenders, "docs/ must hold only CAPITALISED .md files:\n  " + "\n  ".join(offenders)


def test_config_has_no_subfolders() -> None:
    """`config/` is flat: one YAML per experiment, no subfolders and no inheritance.

    A config file is read top to bottom and that is the whole story. Nesting is how
    "what did this run actually use?" becomes a question you answer by simulating a merge.
    """
    subdirs = [p.name for p in (REPO_ROOT / "config").iterdir() if p.is_dir() and p.name != "__pycache__"]
    assert not subdirs, f"config/ must be flat, found subfolders: {subdirs}"


def test_config_yaml_sweep_block_is_first() -> None:
    """When a config has a `sweep:` block it is the FIRST key in the file.

    Everything below the sweep is a single value. Putting the sweep at the top means the
    number of runs a file describes is visible without reading to the bottom.
    """
    offenders = []
    for path in sorted((REPO_ROOT / "config").glob("*.yaml")):
        keys = [
            re.match(r"^([A-Za-z_][\w-]*):", line).group(1)
            for line in path.read_text(encoding="utf-8").splitlines()
            if re.match(r"^([A-Za-z_][\w-]*):", line)
        ]
        if "sweep" in keys:
            before = keys[: keys.index("sweep")]
            # `name` and `seed` identify the experiment rather than parameterise it, so they
            # are allowed above the sweep block.
            unexpected = [k for k in before if k not in ("name", "seed")]
            if unexpected:
                offenders.append(f"{path.name}: {unexpected} appear above `sweep:`")
    assert not offenders, "\n".join(offenders)


def test_changelog_dates_are_dd_mm_yyyy_newest_first() -> None:
    """`CHANGELOG.md` and `AGENTS_MEMORY.md`: `## DD-MM-YYYY` headings, newest at the top.

    One date format, sortable by eye, and newest first because a reader wants the current
    state rather than the archaeology.
    """
    if is_uninitialised_template():
        # In the un-initialised template both files carry a `{{DATE}}` placeholder that the
        # initialiser fills in with today. There is no date to check yet, by design.
        pytest.skip("dates are still placeholders — run _template/init_project.py")
    for name in ("CHANGELOG.md", "AGENTS_MEMORY.md"):
        text = (REPO_ROOT / "docs" / name).read_text(encoding="utf-8")
        headings = re.findall(r"^##\s+(\S+)\s*$", text, flags=re.M)
        dates = [h for h in headings if re.fullmatch(r"\d{2}-\d{2}-\d{4}", h)]
        bad = [h for h in headings if not re.fullmatch(r"\d{2}-\d{2}-\d{4}", h)]
        assert dates, f"docs/{name} has no `## DD-MM-YYYY` chapter"
        assert not bad, f"docs/{name} has `## ` headings that are not DD-MM-YYYY dates: {bad}"
        sortable = [(d[6:], d[3:5], d[:2]) for d in dates]
        assert sortable == sorted(sortable, reverse=True), (
            f"docs/{name} chapters are not newest-first: {dates}"
        )


def test_readme_ends_with_the_template_chapter() -> None:
    """`README.md` ends with the "Based on the repository template" chapter.

    It is the last chapter and nothing follows it, so a project's own content grows
    downward from the top and the provenance stays where every project keeps it.
    """
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    headings = re.findall(r"^##\s+(.+?)\s*$", text, flags=re.M)
    assert headings, "README.md has no `## ` chapters"
    assert headings[-1].lower().startswith("based on the repository template"), (
        f"README.md's last chapter is {headings[-1]!r}, not 'Based on the repository "
        f"template'. Keep that chapter at the bottom and add this project's own above it."
    )
