# Retained from the template and adapted to the layout registry and automatic concurrency.
# These pin the documented contracts — notebooks discovered in registry order from
# `notebooks/<group>/`, and `All_Results.md` sorted alphabetically with each block verbatim.
"""`src/utils/run_notebooks.py` — the runner and the two summary documents.

The end-to-end test executes a real one-cell notebook in a subprocess. It is marked `slow`
because it is, and it is here anyway: the runner's whole job is that a notebook produces
the same files whether a person or a script runs it, and only an actual execution shows that.
"""

from __future__ import annotations

import json

import pytest

from src.utils import run_notebooks as rn


def test_automatic_workers_use_cpu_and_slurm_limits(monkeypatch):
    monkeypatch.setattr(rn.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(rn.os, "sched_getaffinity", lambda _: set(range(8)), raising=False)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    assert rn.worker_count(4) == 4
    assert rn.worker_count(12) == 8
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    assert rn.worker_count(12) == 2
    assert rn.worker_count(12, 1) == 1
    assert rn.worker_count(12, 20) == 2
    with pytest.raises(ValueError):
        rn.worker_count(4, 0)


def make_notebook(path, cells: list[str]) -> None:
    """Write a minimal but valid .ipynb."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": [c],
                        "metadata": {},
                        "outputs": [],
                        "execution_count": None,
                    }
                    for c in cells
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )


def test_synthesis_reads_figures_from_the_current_run(isolated_output, monkeypatch):
    names = ("general/5_synthesis", "B/exp04_decision_features")
    monkeypatch.setattr(rn, "discover", lambda notebooks=None: names)
    figures = isolated_output / "figures_ready"

    def run_one(name, timeout, scratch):
        if name == "general/5_synthesis":
            assert figures.exists()
        else:
            figures.touch()
        return rn.NotebookResult(name, True, 0.1, 0)

    monkeypatch.setattr(rn, "run_one", run_one)
    assert tuple(r.name for r in rn.run_all(max_workers=1)) == names


def test_discovery_follows_registry_order(tmp_path, monkeypatch) -> None:
    """Registry order (general, then A, B, C) is the order CAPTIONS.md uses, so it has to be
    stable — and discovered rather than listed, because a hard-coded list stops covering a
    notebook someone added."""
    monkeypatch.setattr(rn, "notebooks_dir", lambda: tmp_path)
    for name in ("C/exp11_adaptation", "B/exp09_data_efficiency", "general/5_synthesis",
                 "A/A_synthesis", "general/1_data_and_questions", "B/exp04_decision_features"):
        make_notebook(tmp_path / f"{name}.ipynb", ["print(1)"])
    (tmp_path / "paper").mkdir()
    (tmp_path / "paper" / ".gitkeep").write_text("")
    assert rn.discover() == ("general/1_data_and_questions", "general/5_synthesis", "A/A_synthesis",
                             "B/exp04_decision_features", "B/exp09_data_efficiency",
                             "C/exp11_adaptation")


def test_discovery_can_be_overridden_for_a_partial_rerun(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(rn, "notebooks_dir", lambda: tmp_path)
    assert rn.discover(("only_this",)) == ("only_this",)


def test_discovery_uses_group_folders_and_rejects_stray_notebooks(tmp_path, monkeypatch):
    """Notebooks live at `notebooks/<group>/<name>.ipynb` as the registry declares; a flat,
    unregistered or deeper-nested notebook would be silently skipped, so it is refused."""
    monkeypatch.setattr(rn, "notebooks_dir", lambda: tmp_path)
    for name in ("B/exp04_decision_features", "general/1_data_and_questions",
                 "B/exp09_data_efficiency", "general/.ipynb_checkpoints/x-checkpoint"):
        make_notebook(tmp_path / f"{name}.ipynb", ["print(1)"])
    assert rn.discover() == ("general/1_data_and_questions", "B/exp04_decision_features",
                             "B/exp09_data_efficiency")
    # Previous IDs (flat, older and renumbered) and stems resolve to the current notebook.
    for alias in ("rq2/01_decisions", "exp04_decision_features/decisions", "decisions",
                  "exp04_decision_features", "B/exp04_decision_features"):
        assert rn.discover((alias,)) == ("B/exp04_decision_features",), alias
    assert rn.discover(("00_data",)) == ("general/1_data_and_questions",)
    assert rn.discover(("general/01_data",)) == ("general/1_data_and_questions",)
    assert rn.discover(("exp11_data_efficiency",)) == ("B/exp09_data_efficiency",)
    with pytest.raises(ValueError):
        rn.discover(("../outside",))
    for stray in ("exp01_outputs.ipynb", "A/unregistered.ipynb", "B/sub/exp04_decision_features.ipynb",
                  "D/exp04_decision_features.ipynb", "A/exp04_decision_features.ipynb"):
        make_notebook(tmp_path / stray, ["print(1)"])
        with pytest.raises(ValueError, match="registered"):
            rn.discover()
        (tmp_path / stray).unlink()
    assert len(rn.discover()) == 3


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
    from src.utils.paths import notebook_artifacts

    for name, text in (("a", "SUMMARY A"), ("b", "SUMMARY B")):
        folder = notebook_artifacts(name, kind="logs")
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
@pytest.mark.notebook
def test_end_to_end_a_notebook_saves_its_own_figure(isolated_output, monkeypatch, capsys) -> None:
    """The property the whole design rests on: the NOTEBOOK writes the files, so a runner
    execution and an interactive Run All produce the same thing.

    Environment variables reach the notebook subprocess, which is how `isolated_output`
    keeps its figures out of the repository. The coordinator can redirect discovery in
    this process while the notebook itself executes in an isolated interpreter.
    """
    from src.utils.paths import REPO_ROOT, figures_dir

    nb_dir = isolated_output / "notebooks"
    monkeypatch.setattr(rn, "notebooks_dir", lambda: nb_dir)
    make_notebook(
        nb_dir / "general" / "4_study_audit.ipynb",
        [
            "import sys\n"
            f"sys.path.insert(0, r{str(REPO_ROOT)!r})\n"
            "import matplotlib.pyplot as plt\n"
            "from src.visualize import figures, style\n"
            "style.apply()\n"
            "save = figures.FigureSaver('general/4_study_audit')\n"
            "fig, ax = plt.subplots(figsize=style.figsize(style.WIDTH_HALF))\n"
            "ax.plot([0, 1], [0, 1])\n"
            "save(fig, 'line', caption='A line from (0,0) to (1,1).')\n"
            "from IPython.display import display\n"
            "display(fig)\n"
            "print('SMOKE SUMMARY: 1 figure')\n"
        ],
    )
    result = rn.run_one("general/4_study_audit")
    assert result.ok, result.error
    assert result.n_figures == 1
    executed = json.loads((nb_dir / "general" / "4_study_audit.ipynb").read_text(encoding="utf-8"))
    assert executed["cells"][0]["execution_count"] == 1
    assert any("image/png" in output.get("data", {}) for output in executed["cells"][0]["outputs"])
    assert any(
        "SMOKE SUMMARY" in "".join(output.get("text", ""))
        for output in executed["cells"][0]["outputs"]
    )
    progress = capsys.readouterr().out
    assert "cell 1/1" in progress and "saved executed outputs" in progress

    folder = figures_dir("general/4_study_audit")
    assert folder.parts[-3:] == ("figures", "general", "4_study_audit")
    assert (folder / "01_line.pdf").is_file()
    assert not list(folder.glob("*.png"))  # PDF only

    from src.utils.paths import all_results_path, captions_path

    rn.write_captions(("general/4_study_audit",))
    rn.write_all_results(("general/4_study_audit",))
    assert "A line from (0,0) to (1,1)." in captions_path().read_text(encoding="utf-8")
    assert "SMOKE SUMMARY: 1 figure" in all_results_path().read_text(encoding="utf-8")


@pytest.mark.slow
@pytest.mark.notebook
def test_cell_failure_saves_partial_outputs_and_error(isolated_output, monkeypatch):
    nb_dir = isolated_output / "notebooks"
    monkeypatch.setattr(rn, "notebooks_dir", lambda: nb_dir)
    target = nb_dir / "general" / "4_study_audit.ipynb"
    make_notebook(
        target,
        ["print('before error')", "raise ValueError('expected failure')", "print('must not run')"],
    )
    result = rn.run_one("general/4_study_audit")
    assert not result.ok
    executed = json.loads(target.read_text(encoding="utf-8"))
    assert "before error" in "".join(executed["cells"][0]["outputs"][0]["text"])
    assert executed["cells"][1]["outputs"][0]["output_type"] == "error"
    assert executed["cells"][2]["execution_count"] is None
