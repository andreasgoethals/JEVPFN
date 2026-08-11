"""Is the TEMPLATE itself okay? Run this after changing anything in this repository.

    python _template/check_template.py             the full sweep
    python _template/check_template.py --fast      skip the notebook run
    python _template/check_template.py --keep      leave the throwaway copy for inspection

NOT `scripts/check.py`, which answers "is this repository healthy?" for a project. This answers
"does this template still produce a healthy project?", and the only honest way is to make one:
copy the repository to a temp directory, run `init_project.py` there, then in the copy —

  1. `scripts/check.py`         ruff, every import, pytest
  2. `scripts/run_notebooks.py` and a check that it really wrote figures and both documents
  3. one violation at a time    every rule INJECTED into a fresh copy, compliance test must FAIL

Step 3 is the point. A compliance test only ever seen to pass is decoration. Every rule added to
`docs/TEMPLATE.md` gets an entry in VIOLATIONS below; if it cannot be made to fail, it is not
enforced.

Nothing here touches this repository — it reads it, and writes inside temp directories.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Not copied: `.git` would share history, and the caches are large and worthless.
IGNORE = shutil.ignore_patterns(
    ".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".venv", "venv",
    "*.pyc", "output", "tfm-library",
)


def fresh_copy(label: str) -> Path:
    """A copy of this repository, initialised as a project, in a new temp directory.

    UNIQUE every time, never a reused path: Windows will not remove a directory while any process
    holds a handle inside it, and a just-finished pytest subprocess often does.
    """
    target = Path(tempfile.mkdtemp(prefix=f"tmplchk_{label}_")) / "DemoProj"
    shutil.copytree(REPO_ROOT, target, ignore=IGNORE)
    # output/ is excluded from the copy (it may hold a previous local run), so rebuild the
    # tracked skeleton the compliance test requires.
    for relative in ("output", "output/figures", "output/logs", "output/manifests", "output/results"):
        directory = target / relative
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ".gitkeep").write_text("", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "_template/init_project.py", "DemoProj"],
        cwd=target, capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"init_project.py failed in the copy:\n{result.stdout}\n{result.stderr}")
    return target


def run_step(label: str, argv: list[str], cwd: Path) -> bool:
    print(f"\n{'-' * 74}\n{label}\n{'-' * 74}", flush=True)
    result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, check=False)
    ok = result.returncode == 0
    if not ok:
        print(result.stdout[-4000:])
        print(result.stderr[-2000:])
    else:
        print("  OK")
    return ok


# ---------------------------------------------------------------------------
# The violations, one per rule. Each function breaks the copy in exactly ONE way and returns the
# test that must catch it — injecting two at once lets one over-broad check take both credits.
# ---------------------------------------------------------------------------


def edit_template_doc(repo: Path) -> str:
    path = repo / "docs" / "TEMPLATE.md"
    path.write_text(path.read_text(encoding="utf-8") + "\na local note\n", encoding="utf-8")
    return "test_template_doc_is_unmodified"


def remove_required_dir(repo: Path) -> str:
    # Renamed rather than deleted: a fresh copytree on Windows often still holds the handle.
    (repo / "output" / "manifests").rename(repo / "output" / "manifests_gone")
    return "test_required_directory_exists"


def remove_required_file(repo: Path) -> str:
    (repo / "docs" / "AGENTS_MEMORY.md").unlink()
    return "test_required_file_exists"


def generated_file_outside_output(repo: Path) -> str:
    (repo / "figures").mkdir(exist_ok=True)
    (repo / "figures" / "stray.pdf").write_bytes(b"%PDF-1.4 not really a pdf")
    return "test_no_generated_files_outside_output"


def hardcoded_write_path(repo: Path) -> str:
    (repo / "src" / "utils" / "sloppy.py").write_text(
        'def dump(rows):\n    with open("results/scores.csv", "w") as fh:\n'
        '        fh.write(rows)\n',
        encoding="utf-8",
    )
    return "test_no_hardcoded_write_paths_outside_output"


def script_without_main(repo: Path) -> str:
    (repo / "scripts" / "helper.py").write_text(
        "VALUE = 1\n\n\ndef helper(x):\n    return x * VALUE\n", encoding="utf-8"
    )
    return "test_script_is_runnable"


def _edit_notebook(repo: Path, mutate) -> None:
    path = repo / "notebooks" / "example_analysis.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    mutate(notebook)
    path.write_text(json.dumps(notebook), encoding="utf-8")


def notebook_with_a_def(repo: Path) -> str:
    _edit_notebook(repo, lambda nb: nb["cells"].insert(
        1, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": ["def helper(x):\n", "    return x * 2\n"]}
    ))
    return "test_notebook_defines_no_logic"


def notebook_not_ending_in_print(repo: Path) -> str:
    _edit_notebook(repo, lambda nb: nb["cells"].append(
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
         "source": ["total = 1 + 1\n"]}
    ))
    return "test_notebook_ends_by_printing_a_summary"


def unanchored_gitignore_rule(repo: Path) -> str:
    path = repo / ".gitignore"
    path.write_text(path.read_text(encoding="utf-8") + "\nfigures/\n", encoding="utf-8")
    return "test_gitignore_ambiguous_rules_are_anchored"


def lowercase_docs_file(repo: Path) -> str:
    (repo / "docs" / "notes.md").write_text("# a lower-case name\n", encoding="utf-8")
    return "test_docs_holds_only_capitalised_markdown"


def config_subfolder(repo: Path) -> str:
    (repo / "config" / "nested").mkdir()
    (repo / "config" / "nested" / "base.yaml").write_text("name: base\n", encoding="utf-8")
    return "test_config_has_no_subfolders"


def changelog_out_of_order(repo: Path) -> str:
    path = repo / "docs" / "CHANGELOG.md"
    path.write_text(path.read_text(encoding="utf-8") + "\n## 01-01-2030\n\n- from the future\n",
                    encoding="utf-8")
    return "test_changelog_dates_are_dd_mm_yyyy_newest_first"


def chapter_after_the_template_chapter(repo: Path) -> str:
    path = repo / "README.md"
    path.write_text(path.read_text(encoding="utf-8") + "\n## An extra chapter at the end\n",
                    encoding="utf-8")
    return "test_readme_ends_with_the_template_chapter"


VIOLATIONS = [
    ("rule 1  edited docs/TEMPLATE.md", edit_template_doc),
    ("rule 2  missing required directory", remove_required_dir),
    ("rule 2  missing required file", remove_required_file),
    ("rule 3  generated file outside output/", generated_file_outside_output),
    ("rule 3  hard-coded write path in src/", hardcoded_write_path),
    ("rule 4  scripts/ file with no __main__", script_without_main),
    ("rule 5  notebook containing a def", notebook_with_a_def),
    ("rule 6  notebook not ending in a print", notebook_not_ending_in_print),
    ("rule 7  unanchored .gitignore rule", unanchored_gitignore_rule),
    ("docs    file not CAPITALISED", lowercase_docs_file),
    ("config  subfolder", config_subfolder),
    ("logs    changelog not newest-first", changelog_out_of_order),
    ("readme  chapter after the template one", chapter_after_the_template_chapter),
]


def check_violation(index: int, label: str, inject, keep: bool) -> bool:
    repo = fresh_copy(f"v{index:02d}")
    expected = inject(repo)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_template_compliance.py",
         "-q", "--no-header", "-x", "-k", expected],
        cwd=repo, capture_output=True, text=True, check=False,
    )
    caught = result.returncode != 0 and "failed" in result.stdout
    print(f"  {'CAUGHT ' if caught else 'MISSED!'}  {label:<42} -> {expected}")
    if not caught:
        print(result.stdout[-1500:])
    if not keep:
        shutil.rmtree(repo.parent, ignore_errors=True)
    return caught


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--fast", action="store_true", help="skip the notebook execution")
    parser.add_argument("--keep", action="store_true", help="leave the temp copies behind")
    args = parser.parse_args(argv)

    print(f"{'=' * 74}\nCHECKING THE TEMPLATE\n{'=' * 74}")
    print(f"source: {REPO_ROOT}")

    healthy = fresh_copy("main")
    print(f"copy:   {healthy}")

    results = [
        ("scripts/check.py in a fresh project",
         run_step("scripts/check.py", [sys.executable, "scripts/check.py"], healthy)),
    ]
    if not args.fast:
        results.append((
            "scripts/run_notebooks.py in a fresh project",
            run_step("scripts/run_notebooks.py", [sys.executable, "scripts/run_notebooks.py"], healthy),
        ))
        # A green runner that produced nothing is the failure mode worth guarding: a notebook
        # can silently stop saving and the exit code stays 0.
        produced = list((healthy / "output" / "figures").rglob("*.pdf"))
        captions = healthy / "output" / "figures" / "CAPTIONS.md"
        summaries = healthy / "output" / "All_Results.md"
        artefacts_ok = bool(produced) and captions.is_file() and summaries.is_file()
        print(f"\n  {len(produced)} PDFs, CAPTIONS.md {'yes' if captions.is_file() else 'MISSING'}, "
              f"All_Results.md {'yes' if summaries.is_file() else 'MISSING'}")
        results.append(("the notebooks actually produced their artefacts", artefacts_ok))

    if not args.keep:
        shutil.rmtree(healthy.parent, ignore_errors=True)

    print(f"\n{'-' * 74}\nEVERY RULE, INJECTED INTO A FRESH COPY\n{'-' * 74}")
    caught = [check_violation(i, label, inject, args.keep)
              for i, (label, inject) in enumerate(VIOLATIONS)]
    results.append((f"all {len(VIOLATIONS)} violations caught", all(caught)))

    print(f"\n{'=' * 74}\nVERDICT\n{'=' * 74}")
    for label, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    print(f"  {sum(caught)}/{len(caught)} violations caught")
    everything = all(ok for _, ok in results)
    print(f"\n{'The template is okay.' if everything else 'TEMPLATE IS BROKEN — see the FAILs.'}")
    return 0 if everything else 1


if __name__ == "__main__":
    raise SystemExit(main())
