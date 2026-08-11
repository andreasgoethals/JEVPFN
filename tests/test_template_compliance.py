"""The template's rules, enforced. This is what stops `docs/TEMPLATE.md` being advice.

Every check maps to a rule in `docs/TEMPLATE.md` § The compliance test. If a check and the
document disagree, the document is right and this file is a bug.

Two deliberate properties: it **never imports the project** — it reads files, so it runs on a
fresh clone before `pip install`, before the submodule is populated, and in CI with nothing set
up. And it **fails rather than warns**: a warning in a test suite is output nobody reads.
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

#: SHA-256 of the template when this repository was initialised, written by
#: `_template/init_project.py`. The placeholder means it has not run yet.
EXPECTED_TEMPLATE_SHA256 = "{{TEMPLATE_SHA256}}"

TEMPLATE_DOC = REPO_ROOT / "docs" / "TEMPLATE.md"
#: A root copy, if this repository keeps one. Compared only when it exists.
ROOT_TEMPLATE_DOC = REPO_ROOT / "TEMPLATE.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_uninitialised_template() -> bool:
    """True when this is still the template repository, name not yet filled in.

    The signal is the placeholder in `paths.py`, NOT the presence of `_template/`: a project that
    ran the initialiser but forgot to delete that folder must still have its hash enforced. Read
    as text, because this file never imports the project. The pattern's braces are escaped, which
    also stops the initialiser substituting this line — it looks for two literal adjacent braces.
    """
    module = REPO_ROOT / "src" / "utils" / "paths.py"
    if not module.is_file():
        return False
    return bool(re.search(r"PROJECT_NAME\s*=\s*[\"']\{\{[A-Z_]+\}\}[\"']",
                          module.read_text(encoding="utf-8")))


def test_template_doc_exists() -> None:
    assert TEMPLATE_DOC.is_file(), f"{TEMPLATE_DOC} is missing — it governs this repository"


def test_template_doc_is_unmodified() -> None:
    """`docs/TEMPLATE.md` must hash to the template it came from.

    A hash and not a diff, because the rule is not "roughly the same" but "the same file": the
    template is shared verbatim, so a local edit either gets overwritten on the next sync or
    quietly makes this repository's rules diverge from everyone else's. Project-specific rules go
    in `README.md` or a new `docs/<NAME>.md`.
    """
    if is_uninitialised_template():
        pytest.skip(
            "this is the un-initialised template — docs/TEMPLATE.md IS the source, so there is "
            "nothing to compare it against. Enforced once init_project.py has run."
        )
    if EXPECTED_TEMPLATE_SHA256.startswith("{{"):
        pytest.fail(
            "EXPECTED_TEMPLATE_SHA256 is still the placeholder but this repository has a project "
            "name — re-run `python _template/init_project.py <ProjectName>`."
        )
    actual = sha256(TEMPLATE_DOC)
    assert actual == EXPECTED_TEMPLATE_SHA256, (
        f"docs/TEMPLATE.md has been edited.\n  expected {EXPECTED_TEMPLATE_SHA256}\n"
        f"  actual   {actual}\nRevert it; change the rule at the template source instead."
    )


def test_template_doc_matches_root_copy() -> None:
    """When a root `TEMPLATE.md` exists too, the two must be identical."""
    if not ROOT_TEMPLATE_DOC.is_file():
        pytest.skip("this repository keeps no root TEMPLATE.md — docs/TEMPLATE.md is the only copy")
    assert sha256(TEMPLATE_DOC) == sha256(ROOT_TEMPLATE_DOC), (
        "docs/TEMPLATE.md and TEMPLATE.md differ. They are the same document; one was edited."
    )


# ---------------------------------------------------------------------------
# Rule 2 — the structure exists.
# ---------------------------------------------------------------------------

REQUIRED_DIRS = (
    "config", "data/raw", "data/processed", "docs", "notebooks",
    "output", "output/figures", "output/logs", "output/manifests", "output/results",
    "scripts", "scripts/slurm", "src", "src/data", "src/utils", "src/visualize", "tests",
)

REQUIRED_FILES = (
    ".gitattributes", ".gitignore", ".gitmodules", "AGENTS.md", "LICENSE", "README.md",
    "pyproject.toml",
    "docs/TEMPLATE.md", "docs/CHANGELOG.md", "docs/AGENTS_MEMORY.md", "docs/VSC.md",
    "src/utils/paths.py", "src/utils/run_notebooks.py", "src/utils/run_artifacts.py",
    "src/visualize/style.py", "src/visualize/figures.py",
    "scripts/check.py", "scripts/clean_run.py", "tests/test_template_compliance.py",
)


@pytest.mark.parametrize("relative", REQUIRED_DIRS)
def test_required_directory_exists(relative: str) -> None:
    assert (REPO_ROOT / relative).is_dir(), (
        f"{relative}/ is missing. Empty ones are kept by a tracked .gitkeep so a fresh clone "
        f"still has somewhere to write."
    )


@pytest.mark.parametrize("relative", REQUIRED_FILES)
def test_required_file_exists(relative: str) -> None:
    assert (REPO_ROOT / relative).is_file(), f"required file {relative} is missing"


def test_literature_submodule_is_declared() -> None:
    """`tfm-library/` must be registered as a submodule.

    The DIRECTORY need not be populated — it is empty until `git submodule update --init`, and CI
    checks out without submodules on purpose. What must hold is that the repository declares it,
    so every consumer is pinned to one exact commit of the literature.
    """
    assert "tfm-library" in (REPO_ROOT / ".gitmodules").read_text(encoding="utf-8"), (
        ".gitmodules does not declare tfm-library. Every project carries the shared literature "
        "as a pinned, read-only submodule:\n  git submodule add <url> tfm-library"
    )


# ---------------------------------------------------------------------------
# Rule 3 — nothing generated is written outside output/.
# ---------------------------------------------------------------------------

#: Extensions and names that only exist because code produced them.
GENERATED_SUFFIXES = frozenset(
    {".pdf", ".png", ".svg", ".eps", ".csv", ".tsv", ".log", ".jsonl", ".npy", ".npz",
     ".pkl", ".pickle", ".joblib", ".ckpt", ".pt", ".pth", ".safetensors", ".parquet"}
)
GENERATED_NAMES = frozenset({"CAPTIONS.md", "All_Results.md", "_figures.json", "_stdout.txt"})

#: Where a generated-looking file is legitimate: `output/` (the one place they belong), `data/`
#: (inputs and the cache — not produced by a run), `checkpoints/` (weights), `tfm-library/` (full
#: of PDFs, and not ours), `docs/` (a hand-made diagram), `tests/` (fixtures).
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
    """A generated artefact anywhere but `output/` means code wrote to the wrong place.

    This is the check with no false negatives: whatever path the code built, the file either is
    under `output/` or is not. One root means "what did this run produce?" and "what can I
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
        "generated files found outside output/:\n  " + "\n  ".join(sorted(offenders))
        + "\nEverything the code produces goes under output/. Build the destination with "
          "src/utils/paths.py instead of a literal path."
    )


#: Calls whose FIRST argument is the destination. Deliberately narrow: a check that guesses
#: produces false positives, and a compliance test people disable is worse than none.
WRITE_CALLS_ARG0 = frozenset(
    {"open", "savefig", "to_csv", "to_parquet", "to_json", "to_pickle", "to_feather",
     "savetxt", "imsave", "write_image"}
)

#: Exempt by design: `paths.py` is the resolver, and naming other roots is its entire job.
LITERAL_PATH_EXEMPT = ("src/utils/paths.py",)


def _source_files() -> list[Path]:
    files = []
    for folder in ("src", "scripts"):
        files += [p for p in sorted((REPO_ROOT / folder).rglob("*.py"))
                  if not (SKIP_TREE_PARTS & set(p.parts))]
    return files


def _literal(node: ast.expr) -> str | None:
    """The string value of a literal argument, or None if it is not one."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):  # f-string: the literal parts are enough to judge
        return "".join(v.value for v in node.values if isinstance(v, ast.Constant))
    return None


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    return func.id if isinstance(func, ast.Name) else ""


def _is_write_open(node: ast.Call) -> bool:
    """`open(...)` in a writing mode. A read-mode open is not our business."""
    if _call_name(node) != "open":
        return False
    mode = _literal(node.args[1]) if len(node.args) > 1 else None
    for kw in node.keywords:
        if kw.arg == "mode":
            mode = _literal(kw.value)
    return bool(mode) and any(c in mode for c in "wax")


def test_no_hardcoded_write_paths_outside_output() -> None:
    """A write call with a literal destination must name `output/` — or not be literal.

    The filesystem check above catches this after a local run; this catches it before anyone has
    run it on the cluster, where `output/` is on a different tier entirely and the mistake costs
    a job.
    """
    offenders = []
    for path in _source_files():
        rel = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
        if rel in LITERAL_PATH_EXEMPT:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            name = _call_name(node)
            if name not in WRITE_CALLS_ARG0 or (name == "open" and not _is_write_open(node)):
                continue
            literal = _literal(node.args[0])
            # No literal means the destination came from somewhere — normally paths.py. A literal
            # with no separator and no extension is not a path (a format name, a buffer, a mode).
            if literal is None or not any(c in literal for c in "/\\."):
                continue
            if literal.replace("\\", "/").lstrip("./").startswith("output/"):
                continue
            offenders.append(f"{rel}:{node.lineno}  {name}({literal!r})")
    assert not offenders, (
        "write calls with a hard-coded destination outside output/:\n  " + "\n  ".join(offenders)
        + "\nAsk src/utils/paths.py — it is the only module that builds a path, because on the "
          "cluster output/ is not where it is locally."
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

    A module in `scripts/` cannot be imported or tested — that is the point of the directory — so
    something importable there is untestable by construction and belongs in `src/`.
    """
    tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    has_main = any(
        isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__"
        for node in tree.body
    )
    assert has_main, (
        f'scripts/{script.name} has no `if __name__ == "__main__":` block. Only runnable entry '
        f"points live in scripts/; anything importable belongs in src/."
    )


# ---------------------------------------------------------------------------
# Rules 5 and 6 — notebooks are thin, and they end by printing.
# ---------------------------------------------------------------------------


def _notebooks() -> list[Path]:
    return sorted(p for p in (REPO_ROOT / "notebooks").rglob("*.ipynb")
                  if ".ipynb_checkpoints" not in p.parts)


def _code_cells(notebook: Path) -> list[str]:
    cells = json.loads(notebook.read_text(encoding="utf-8")).get("cells", [])
    return ["".join(c.get("source", [])) for c in cells if c.get("cell_type") == "code"]


@pytest.mark.parametrize("notebook", _notebooks(), ids=lambda p: p.stem)
def test_notebook_defines_no_logic(notebook: Path) -> None:
    """A notebook contains no `def` and no `class`. All logic lives in `src/`.

    A function defined in a notebook cannot be imported, cannot be tested and cannot be reused by
    the next notebook — so it gets copied, the copies diverge, and two figures disagree for a
    reason nobody can find.
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

    `output/All_Results.md` is assembled from what the notebooks print, so a notebook ending on a
    plot contributes nothing to it: its findings exist only as images — ungreppable, invisible in
    a diff, and invisible to an agent.
    """
    cells = [c for c in _code_cells(notebook) if c.strip()]
    assert cells, f"{notebook.name} has no non-empty code cells"
    assert "print(" in cells[-1], (
        f"{notebook.name}'s last code cell does not print anything:\n---\n"
        f"{cells[-1].strip()[:300]}\n---\nEvery notebook ends by printing a text summary; that "
        f"text is what output/All_Results.md is built from."
    )


# ---------------------------------------------------------------------------
# Rule 7 — .gitignore rules that must be root-anchored are.
# ---------------------------------------------------------------------------

#: Directory names occurring at more than one depth in the tree. For these, an unanchored rule
#: silently ignores files that are supposed to be tracked.
AMBIGUOUS_DIR_NAMES = frozenset(
    {"figures", "logs", "manifests", "results", "runs", "raw", "processed", "data",
     "output", "config", "docs", "src", "tests", "notebooks", "scripts", "slurm",
     "checkpoints", "utils", "visualize"}
)


def test_gitignore_ambiguous_rules_are_anchored() -> None:
    """A rule naming an ambiguous directory must be anchored or explicitly pathed.

    Git matches a pattern with no slash against that name at EVERY depth, so a bare `results/`
    meant for a root directory also hides `output/results/` — and the small result summaries that
    should be committed silently never reach git, noticed only when a clone comes up empty.
    Cheap check for one case: `git check-ignore -v <path you expect tracked>`.
    """
    offenders = []
    for lineno, raw in enumerate((REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
                                 .splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pattern = line[1:] if line.startswith("!") else line
        # A pattern containing a slash is already path-scoped, relative to the .gitignore.
        if "/" in pattern.rstrip("/"):
            continue
        if pattern.rstrip("/") in AMBIGUOUS_DIR_NAMES:
            offenders.append(f"line {lineno}: {line!r} -> anchor it as '/{line.lstrip('!')}'")
    assert not offenders, (
        "unanchored .gitignore rules that also match deeper directories:\n  "
        + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Further rules from the document, cheap enough to enforce here.
# ---------------------------------------------------------------------------


def test_docs_holds_only_capitalised_markdown() -> None:
    """`docs/` holds only `.md` files, named in CAPITALS — so a documentation file is
    recognisable from its name alone, and code and data stay out of it."""
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
    """`config/` is flat: one YAML per experiment, no inheritance. Nesting is how "what did this
    run actually use?" becomes a question you answer by simulating a merge."""
    subdirs = [p.name for p in (REPO_ROOT / "config").iterdir()
               if p.is_dir() and p.name != "__pycache__"]
    assert not subdirs, f"config/ must be flat, found subfolders: {subdirs}"


def test_config_yaml_sweep_block_is_first() -> None:
    """A `sweep:` block is the FIRST key, so the number of runs a file describes is visible
    without reading to the bottom. Everything below the sweep is a single value."""
    offenders = []
    for path in sorted((REPO_ROOT / "config").glob("*.yaml")):
        keys = [m.group(1) for m in (re.match(r"^([A-Za-z_][\w-]*):", line)
                                     for line in path.read_text(encoding="utf-8").splitlines())
                if m]
        if "sweep" in keys:
            # `name` and `seed` identify the experiment rather than parameterise it, so they are
            # allowed above the sweep block.
            unexpected = [k for k in keys[: keys.index("sweep")] if k not in ("name", "seed")]
            if unexpected:
                offenders.append(f"{path.name}: {unexpected} appear above `sweep:`")
    assert not offenders, "\n".join(offenders)


def test_changelog_dates_are_dd_mm_yyyy_newest_first() -> None:
    """Both dated logs use `## DD-MM-YYYY` headings, newest at the top: one format, sortable by
    eye, and newest first because a reader wants the current state, not the archaeology."""
    if is_uninitialised_template():
        # Both files carry a `{{DATE}}` placeholder the initialiser fills in. No date to check yet.
        pytest.skip("dates are still placeholders — run _template/init_project.py")
    for name in ("CHANGELOG.md", "AGENTS_MEMORY.md"):
        headings = re.findall(r"^##\s+(\S+)\s*$",
                              (REPO_ROOT / "docs" / name).read_text(encoding="utf-8"), flags=re.M)
        dates = [h for h in headings if re.fullmatch(r"\d{2}-\d{2}-\d{4}", h)]
        bad = [h for h in headings if h not in dates]
        assert dates, f"docs/{name} has no `## DD-MM-YYYY` chapter"
        assert not bad, f"docs/{name} has `## ` headings that are not DD-MM-YYYY dates: {bad}"
        sortable = [(d[6:], d[3:5], d[:2]) for d in dates]
        assert sortable == sorted(sortable, reverse=True), (
            f"docs/{name} chapters are not newest-first: {dates}"
        )


def test_readme_ends_with_the_template_chapter() -> None:
    """`README.md` ends with the "Based on the repository template" chapter and nothing after it,
    so a project's own content grows downward from the top and the provenance stays put."""
    headings = re.findall(r"^##\s+(.+?)\s*$",
                          (REPO_ROOT / "README.md").read_text(encoding="utf-8"), flags=re.M)
    assert headings, "README.md has no `## ` chapters"
    assert headings[-1].lower().startswith("based on the repository template"), (
        f"README.md's last chapter is {headings[-1]!r}, not 'Based on the repository template'. "
        f"Keep that chapter at the bottom and add this project's own above it."
    )
