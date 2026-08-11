"""Saving figures. One folder per notebook, PDF + PNG, and the notebook clears its own.

    output/figures/<notebook>/01_<name>.pdf     vector, 300 dpi raster elements — the paper
    output/figures/<notebook>/01_<name>.png     110 dpi — small enough to commit and review
    output/figures/<notebook>/_figures.json     what was drawn, in order, with captions

WHY THE NOTEBOOK SAVES ITS OWN FIGURES, and the runner does not do it for it: a runner
that captures figures on the notebook's behalf only works inside the runner. *Run All* in
Jupyter — where figures are actually iterated on — then produces nothing, and the two
execution paths silently disagree. Here both paths run the same code, so what you see
interactively is exactly what lands on disk.

WHY BOTH FORMATS: the PDF goes in the paper (vector, text embedded as TrueType so journal
systems accept it); the PNG is committed so a figure can be reviewed in a diff and looked
at on GitHub without cloning and re-running anything.

WHY THE FOLDER IS CLEARED FIRST, by the notebook, on construction: a stale PDF next to a
fresh one is how a paper ends up with a figure that no longer matches the code that made
it. Clearing happens BEFORE anything is drawn, and only ever inside this notebook's own
folder — never another's.

THE NUMBERED PREFIX is what makes `CAPTIONS.md` reproducible: alphabetical order of the
files is the order the notebook drew them, so the captions file can be rebuilt from disk
without re-executing anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.utils.paths import figures_dir

#: Extensions written for every figure. PDF first — it is the one that matters.
FORMATS = ("pdf", "png")

#: DPI per format. The PDF is vector, but heatmaps and scatter clouds inside it rasterise,
#: so it still needs a print DPI. The PNG's is chosen so a few dozen of them do not bloat
#: the repository.
DPI = {"pdf": 300, "png": 110}

#: The runner's and the saver's own bookkeeping files, and the two figure formats. Only
#: these are ever deleted from a notebook's folder — anything else a person put there
#: survives, because a cleaner that removes files it does not recognise is a cleaner that
#: eventually removes something irreplaceable.
_OWNED = ("*.pdf", "*.png", "_figures.json", "_stdout.txt")

MANIFEST = "_figures.json"


def manifest_path(notebook: str) -> Path:
    return figures_dir(notebook) / MANIFEST


def read_manifest(notebook: str) -> list[dict]:
    """What this notebook drew last time it ran, in order. `[]` if it never has."""
    path = manifest_path(notebook)
    if not path.is_file():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # A run killed mid-write leaves truncated JSON. That is a missing manifest, not a
        # crash: the figures are still on disk and the next run rewrites it.
        return []


def clear(notebook: str) -> int:
    """Delete this notebook's own figures and manifest. Returns how many files went.

    Scoped to one notebook's folder and to the extensions above, and non-recursive, so it
    cannot reach a sibling notebook's figures or anything nested.
    """
    folder = figures_dir(notebook)
    if not folder.is_dir():
        return 0
    removed = 0
    for pattern in _OWNED:
        for path in folder.glob(pattern):
            if path.is_file():
                path.unlink()
                removed += 1
    return removed


class FigureSaver:
    """Saves every figure of ONE notebook, in order, with its caption.

    Construct it in the notebook's setup cell, before anything is drawn:

        from src.visualize import figures, style
        style.apply()
        save = figures.FigureSaver("example_analysis")   # clears its own folder here

        fig, ax = plt.subplots()
        ...
        save(fig, "target_distribution",
             caption="Histogram of the target, 40 bins, n = 12,043.")

    CAPTIONS ARE PURE DESCRIPTION: what is plotted, on what axes, from how much data. No
    interpretation, no conclusion, no "this shows that" — exactly what would sit under the
    figure in a journal. The argument belongs in the body text, and a caption that argues
    is a caption that has to be rewritten when the argument changes.

    A caption is required at save time rather than kept in a central registry, so it lives
    next to the figure it describes and cannot go stale when the figure is renamed.
    """

    def __init__(self, notebook: str, *, clear_first: bool = True) -> None:
        self.notebook = notebook
        self.folder = figures_dir(notebook)
        self.folder.mkdir(parents=True, exist_ok=True)
        #: `clear_first=False` exists for one case only: re-running a single cell mid-session
        #: without wiping the figures the earlier cells already wrote. Never pass it in the
        #: setup cell — that is the call that guarantees no stale figure survives.
        if clear_first:
            clear(notebook)
        self.entries: list[dict] = [] if clear_first else read_manifest(notebook)

    # -- saving --------------------------------------------------------------

    def __call__(self, fig: plt.Figure, name: str, caption: str = "", **kwargs) -> list[Path]:
        return self.save(fig, name, caption=caption, **kwargs)

    def save(
        self,
        fig: plt.Figure,
        name: str,
        *,
        caption: str = "",
        close: bool = False,
    ) -> list[Path]:
        """Write `<NN>_<name>.pdf` and `.png`, and record the caption.

        `close=False` by default so the figure still displays in Jupyter — the interactive
        run has to look the same as the runner's, and a closed figure shows nothing.
        """
        index = len(self.entries) + 1
        stem = f"{index:02d}_{_slug(name)}"
        written = []
        for fmt in FORMATS:
            path = self.folder / f"{stem}.{fmt}"
            _guard(path, self.folder)
            fig.savefig(path, format=fmt, dpi=DPI[fmt])
            written.append(path)
        self.entries.append(
            {"index": index, "stem": stem, "name": name, "caption": caption.strip()}
        )
        self._write_manifest()
        if close:
            plt.close(fig)
        return written

    def _write_manifest(self) -> None:
        """Rewritten after every figure, so a notebook that dies halfway still has a
        manifest describing the figures it did produce."""
        manifest_path(self.notebook).write_text(
            json.dumps(self.entries, indent=2), encoding="utf-8"
        )

    # -- the notebook's closing summary -------------------------------------

    def summary(self) -> str:
        """A text listing of what was saved, for the notebook's final `print`.

        Every notebook ends by printing a summary, and the figure list is part of it: it
        makes `All_Results.md` say what the run drew, not only what it computed.
        """
        if not self.entries:
            return f"{self.notebook}: no figures saved."
        lines = [f"{self.notebook}: {len(self.entries)} figures -> {self.folder}"]
        for e in self.entries:
            lines.append(f"  {e['index']:02d}  {e['name']}")
            if not e["caption"]:
                lines.append("      NO CAPTION — add one; CAPTIONS.md will flag it.")
        return "\n".join(lines)


def _slug(name: str) -> str:
    """A filename-safe version of a figure name. Keeps it readable, not opaque."""
    keep = [c if (c.isalnum() or c in "-_") else "_" for c in name.strip().lower()]
    return "".join(keep).strip("_") or "figure"


def _guard(path: Path, folder: Path) -> None:
    """Refuse to write outside this notebook's own figure folder.

    A figure name with a `..` or an absolute path in it would otherwise put a generated
    file outside `output/`, which is the one rule the whole layout rests on. Cheap check,
    caught once in a template rather than in every project.
    """
    resolved = path.resolve()
    root = folder.resolve()
    if root != resolved.parent:
        raise ValueError(
            f"figure would be written to {resolved}, outside {root}. Figure names are "
            f"plain names, not paths."
        )
