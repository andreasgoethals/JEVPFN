# Came with the template, and worth keeping: `src/visualize/figures.py` is identical in every
# project. These pin the behaviour the layout depends on — PDF only, the folder cleared before
# drawing, and only ever this notebook's own folder.
"""`src/visualize/figures.py` — the saver.

Every test redirects `output/` into `tmp_path` via `isolated_output`, so the suite never
writes a real figure into the repository.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pytest

from src.visualize import figures


@pytest.fixture
def fig():
    f, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    yield f
    plt.close(f)


def test_a_pdf_and_only_a_pdf_is_written(isolated_output, fig) -> None:
    """PDF only: the paper uses it, and the notebook displays the figure inline instead of
    keeping a second raster copy on disk to go stale."""
    save = figures.FigureSaver("nb")
    written = save(fig, "first figure", caption="A line from (0,0) to (1,1).")
    assert written.suffix == ".pdf"
    assert written.is_file() and written.stat().st_size > 0
    assert not list(save.folder.glob("*.png"))


def test_filenames_are_numbered_in_drawing_order(isolated_output, fig) -> None:
    """Alphabetical order of the files is the order the notebook drew them, which is what
    makes CAPTIONS.md rebuildable from disk without re-executing anything."""
    save = figures.FigureSaver("nb")
    save(fig, "alpha", caption="c")
    save(fig, "beta", caption="c")
    stems = sorted(p.stem for p in save.folder.glob("*.pdf"))
    assert stems == ["01_alpha", "02_beta"]


def test_the_folder_is_cleared_before_anything_is_drawn(isolated_output, fig) -> None:
    """A stale PDF beside a fresh one is how a paper ends up with a figure that no longer
    matches the code that made it."""
    first = figures.FigureSaver("nb")
    first(fig, "old", caption="c")
    assert (first.folder / "01_old.pdf").is_file()

    figures.FigureSaver("nb")  # constructing it clears
    assert not (first.folder / "01_old.pdf").exists()


def test_clearing_touches_only_this_notebook(isolated_output, fig) -> None:
    """A notebook deletes ITS OWN figures — never another notebook's."""
    a = figures.FigureSaver("nb_a")
    a(fig, "keep", caption="c")
    figures.FigureSaver("nb_b")
    assert (a.folder / "01_keep.pdf").is_file()


def test_clearing_leaves_unrecognised_files_alone(isolated_output, fig) -> None:
    """A cleaner that removes what it does not recognise eventually removes something
    irreplaceable."""
    save = figures.FigureSaver("nb")
    stranger = save.folder / "notes.txt"
    stranger.write_text("mine", encoding="utf-8")
    figures.FigureSaver("nb")
    assert stranger.is_file()


def test_the_manifest_records_the_caption(isolated_output, fig) -> None:
    save = figures.FigureSaver("nb")
    save(fig, "hist", caption="Histogram of the target, 40 bins, n = 1,000.")
    entries = json.loads(figures.manifest_path("nb").read_text(encoding="utf-8"))
    assert entries[0]["name"] == "hist"
    assert entries[0]["caption"].startswith("Histogram")
    assert figures.read_manifest("nb") == entries


def test_the_manifest_survives_a_notebook_that_dies_halfway(isolated_output, fig) -> None:
    """Rewritten after every figure, so a crashed run still describes what it did produce."""
    save = figures.FigureSaver("nb")
    save(fig, "one", caption="c")
    assert len(figures.read_manifest("nb")) == 1


def test_a_truncated_manifest_reads_as_missing_not_as_a_crash(isolated_output) -> None:
    """A run killed mid-write leaves invalid JSON; the figures are still on disk."""
    figures.FigureSaver("nb")
    figures.manifest_path("nb").write_text('[{"index": 1,', encoding="utf-8")
    assert figures.read_manifest("nb") == []


def test_a_figure_name_cannot_escape_the_folder(isolated_output, fig) -> None:
    """A `..` or a separator in a figure name must not put a generated file outside
    `output/` — the one rule the whole layout rests on.

    Slugification neutralises it before it can become a path, so the file lands inside the
    folder rather than raising. `_guard` behind it is defence in depth: it asserts the
    invariant even if slugification is ever loosened.
    """
    save = figures.FigureSaver("nb")
    pdf = save(fig, "../../escaped", caption="c")
    assert pdf.parent == save.folder
    assert ".." not in pdf.name


def test_the_write_guard_rejects_a_path_outside_the_folder(isolated_output) -> None:
    """The guard itself, tested directly — it is unreachable through `save()` by design."""
    save = figures.FigureSaver("nb")
    with pytest.raises(ValueError, match="outside"):
        figures._guard(save.folder.parent / "elsewhere.pdf", save.folder)


def test_names_are_slugified_but_stay_readable(isolated_output, fig) -> None:
    save = figures.FigureSaver("nb")
    pdf = save(fig, "Target Distribution (LGD)", caption="c")
    assert pdf.stem == "01_target_distribution__lgd"


def test_summary_flags_a_missing_caption(isolated_output, fig) -> None:
    """A missing caption must be visible in the notebook's own printed summary, not only in
    the document that is supposed to contain it."""
    save = figures.FigureSaver("nb")
    save(fig, "uncaptioned")
    assert "NO CAPTION" in save.summary()
    save2 = figures.FigureSaver("nb2")
    save2(fig, "captioned", caption="A line.")
    assert "NO CAPTION" not in save2.summary()


def test_the_figure_stays_open_so_jupyter_still_shows_it(isolated_output, fig) -> None:
    """The inline render IS the raster copy, so the figure must never be closed on save."""
    save = figures.FigureSaver("nb")
    save(fig, "shown", caption="c")
    assert plt.fignum_exists(fig.number)


def test_clear_returns_how_many_files_went(isolated_output, fig) -> None:
    save = figures.FigureSaver("nb")
    save(fig, "a", caption="c")
    save(fig, "b", caption="c")
    assert figures.clear("nb") == 3  # 2 pdfs + the manifest
    assert figures.clear("never_existed") == 0
