# Came with the template, and worth keeping: `src/utils/run_notebooks.py` is identical in every
# project, and these pin the two documented contracts — notebooks discovered alphabetically, and
# `All_Results.md` sorted alphabetically with each block verbatim.
"""`src/utils/run_notebooks.py` — the runner and the two summary documents.

The end-to-end test executes a real one-cell notebook in a subprocess. It is marked `slow`
because it is, and it is here anyway: the runner's whole job is that a notebook produces
the same files whether a person or a script runs it, and only an actual execution shows that.
"""

from __future__ import annotations

import json

import pytest

from src.utils import run_notebooks as rn


def make_notebook(path, cells: list[str]) -> None:
    """Write a minimal but valid .ipynb."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "code", "source": [c], "metadata": {}, "outputs": [],
                     "execution_count": None}
                    for c in cells
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )


def test_discovery_is_alphabetical(tmp_path, monkeypatch) -> None:
    """Alphabetical order is also the order both summary documents use, so it has to be
    stable — and discovered rather than listed, because a hard-coded list stops covering a
    notebook someone added."""
    monkeypatch.setattr(rn, "notebooks_dir", lambda: tmp_path)
    for name in ("zeta", "alpha", "mid"):
        make_notebook(tmp_path / f"{name}.ipynb", ["print(1)"])
    assert rn.discover() == ("alpha", "mid", "zeta")


def test_discovery_can_be_overridden_for_a_partial_rerun(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(rn, "notebooks_dir", lambda: tmp_path)
    assert rn.discover(("only_this",)) == ("only_this",)


def test_magics_are_stripped_from_the_flattened_script(tmp_path) -> None:
    """`%matplotlib inline` is a syntax error in a plain interpreter, and a notebook that
    needs a magic to run cannot be executed non-interactively at all."""
    nb = tmp_path / "nb.ipynb"
    make_notebook(nb, ["%matplotlib inline\nprint('hello')", "!ls\nprint('two')"])
    script = rn._build_script(nb, tmp_path / "out.txt")
    assert "%matplotlib" not in script and "!ls" not in script
    assert "print('hello')" in script and "print('two')" in script


def test_markdown_cells_are_skipped(tmp_path) -> None:
    nb = tmp_path / "nb.ipynb"
    nb.write_text(
        json.dumps({
            "cells": [
                {"cell_type": "markdown", "source": ["# a heading"]},
                {"cell_type": "code", "source": ["print('code')"], "outputs": [],
                 "execution_count": None, "metadata": {}},
            ],
            "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
        }),
        encoding="utf-8",
    )
    script = rn._build_script(nb, tmp_path / "out.txt")
    assert "a heading" not in script and "print('code')" in script


def test_captions_are_grouped_per_notebook_in_order(isolated_output, monkeypatch) -> None:
    """ONE CAPTIONS.md for the project, built from each notebook's manifest — so it can be
    regenerated after an interactive run without executing anything."""
    from src.utils.paths import figures_dir

    for name, entries in {
        "b_second": [{"index": 1, "stem": "01_x", "name": "x", "caption": "Caption X."}],
        "a_first": [
            {"index": 1, "stem": "01_p", "name": "p", "caption": "Caption P."},
            {"index": 2, "stem": "02_q", "name": "q", "caption": "Caption Q."},
        ],
    }.items():
        folder = figures_dir(name)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "_figures.json").write_text(json.dumps(entries), encoding="utf-8")

    text = rn.write_captions(("a_first", "b_second")).read_text(encoding="utf-8")
    assert text.index("## a_first") < text.index("## b_second")
    assert text.index("01_p") < text.index("02_q")
    for caption in ("Caption P.", "Caption Q.", "Caption X."):
        assert caption in text


def test_a_missing_caption_is_flagged_not_skipped(isolated_output) -> None:
    """A gap should be visible in the document that is supposed to contain it."""
    from src.utils.paths import figures_dir

    folder = figures_dir("nb")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "_figures.json").write_text(
        json.dumps([{"index": 1, "stem": "01_x", "name": "x", "caption": ""}]), encoding="utf-8"
    )
    assert "MISSING CAPTION" in rn.write_captions(("nb",)).read_text(encoding="utf-8")


def test_a_notebook_with_no_figures_still_gets_a_section(isolated_output) -> None:
    assert "_No figures produced._" in rn.write_captions(("empty",)).read_text(encoding="utf-8")


def test_all_results_is_sorted_alphabetically_by_notebook(isolated_output) -> None:
    """One block per notebook, verbatim, alphabetical — even when passed out of order."""
    from src.utils.paths import figures_dir

    for name, text in (("a", "SUMMARY A"), ("b", "SUMMARY B")):
        folder = figures_dir(name)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / rn.STDOUT_FILE).write_text(text, encoding="utf-8")

    # Passed b-then-a on purpose: the file must still come out a-then-b.
    written = rn.write_all_results(("b", "a")).read_text(encoding="utf-8")
    assert written.index("SUMMARY A") < written.index("SUMMARY B")
    assert written.index("## a") < written.index("## b")


def test_a_missing_notebook_is_reported_not_raised(isolated_output, monkeypatch) -> None:
    """One bad notebook must not take the other eleven down with it."""
    result = rn.run_one("does_not_exist")
    assert result.ok is False and "not found" in result.error


def test_summarise_says_where_everything_went() -> None:
    results = [
        rn.NotebookResult("ok_one", True, 1.2, 3),
        rn.NotebookResult("broken", False, 0.4, 0, "ValueError: nope"),
    ]
    text = rn.summarise(results)
    assert "OK" in text and "FAILED" in text and "ValueError: nope" in text
    assert "1/2 notebooks OK" in text
    assert rn.summarise([]) == "No notebooks found in notebooks/."


@pytest.mark.slow
def test_end_to_end_a_notebook_saves_its_own_figure(isolated_output, monkeypatch) -> None:
    """The property the whole design rests on: the NOTEBOOK writes the files, so a runner
    execution and an interactive Run All produce the same thing.

    `run_one` plus the two writers rather than `run_all`: the process pool starts a fresh
    interpreter that does not inherit monkeypatches, so a redirected `notebooks_dir` would
    be invisible to it. Environment variables ARE inherited, which is how `isolated_output`
    still keeps the figures out of the repository. `run_all` is exactly these three calls
    plus the pool, and each is covered.
    """
    from src.utils.paths import REPO_ROOT, figures_dir

    nb_dir = isolated_output / "notebooks"
    monkeypatch.setattr(rn, "notebooks_dir", lambda: nb_dir)
    make_notebook(
        nb_dir / "smoke.ipynb",
        [
            "import sys\n"
            f"sys.path.insert(0, r{str(REPO_ROOT)!r})\n"
            "import matplotlib.pyplot as plt\n"
            "from src.visualize import figures, style\n"
            "style.apply()\n"
            "save = figures.FigureSaver('smoke')\n"
            "fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_HALF))\n"
            "ax.plot([0, 1], [0, 1])\n"
            "save(fig, 'line', caption='A line from (0,0) to (1,1).')\n"
            "print('SMOKE SUMMARY: 1 figure')\n"
        ],
    )
    result = rn.run_one("smoke")
    assert result.ok, result.error
    assert result.n_figures == 1
    folder = figures_dir("smoke")
    assert (folder / "01_line.pdf").is_file()
    assert not list(folder.glob("*.png"))  # PDF only

    from src.utils.paths import all_results_path, captions_path

    rn.write_captions(("smoke",))
    rn.write_all_results(("smoke",))
    assert "A line from (0,0) to (1,1)." in captions_path().read_text(encoding="utf-8")
    assert "SMOKE SUMMARY: 1 figure" in all_results_path().read_text(encoding="utf-8")
