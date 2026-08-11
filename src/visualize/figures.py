"""Saving figures. One folder per notebook, one PDF per figure, cleared before drawing.

    output/figures/<notebook>/01_<name>.pdf     the figure — vector, for the paper
    output/figures/<notebook>/_figures.json     what was drawn, in order, with captions

PDF ONLY, AND SIZED FOR A4. The PDF is what the paper uses: vector, text embedded as TrueType so
journal systems accept it, drawn at the width it will occupy on the A4 page (see
`src/visualize/style.py`). The notebook *displays* each figure inline, so a reader sees them by
scrolling the notebook — there is no second raster copy on disk to go stale.

THE NOTEBOOK SAVES ITS OWN FIGURES, not the runner: a runner that captures them on the notebook's
behalf only works inside the runner, so *Run All* in Jupyter — where figures are actually iterated
on — produces nothing, and the two paths silently disagree.

THE FOLDER IS CLEARED ON CONSTRUCTION, before anything is drawn, and only ever this notebook's
own: a stale PDF beside a fresh one is how a paper ends up with a figure that no longer matches
the code that made it.

THE NUMBERED PREFIX makes alphabetical order equal drawing order, so `CAPTIONS.md` is rebuildable
from disk without re-executing anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.utils.paths import figures_dir

#: Vector already, but heatmaps and scatter clouds inside a PDF rasterise, so it still needs a
#: print DPI.
DPI = 300

#: The only things ever deleted from a notebook's folder. Anything else a person put there
#: survives: a cleaner that removes what it does not recognise eventually removes something
#: irreplaceable.
_OWNED = ("*.pdf", "_figures.json", "_stdout.txt")

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
        # A run killed mid-write leaves truncated JSON. That is a missing manifest, not a crash:
        # the figures are still on disk and the next run rewrites it.
        return []


def clear(notebook: str) -> int:
    """Delete this notebook's own figures and manifest; returns how many went.

    Scoped to one folder, to the patterns above, and non-recursive, so it cannot reach a sibling
    notebook's figures.
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

    THE CAPTION IS THE PAPER'S CAPTION. Write it to be pasted straight under the figure in the
    manuscript: pure description — what is plotted, on what axes, from how much data. No
    interpretation, no conclusion. The argument goes in the body text, and a caption that argues has
    to be rewritten when the argument changes.

    Passed at save time rather than kept in a central registry, so a caption lives next to the
    figure it describes and cannot go stale when that figure is renamed.
    """

    def __init__(self, notebook: str, *, clear_first: bool = True) -> None:
        self.notebook = notebook
        self.folder = figures_dir(notebook)
        self.folder.mkdir(parents=True, exist_ok=True)
        #: `clear_first=False` is for one case only: re-running a single cell mid-session without
        #: wiping what the earlier cells wrote. Never pass it in the setup cell.
        if clear_first:
            clear(notebook)
        self.entries: list[dict] = [] if clear_first else read_manifest(notebook)

    def __call__(self, fig: plt.Figure, name: str, caption: str = "", **kwargs) -> Path:
        return self.save(fig, name, caption=caption, **kwargs)

    def save(self, fig: plt.Figure, name: str, *, caption: str = "") -> Path:
        """Write `<NN>_<name>.pdf` and record the caption.

        The figure is left open so it still displays in Jupyter — that inline render is the only
        raster copy there is, and the interactive run has to look the same as the runner's.
        """
        index = len(self.entries) + 1
        stem = f"{index:02d}_{_slug(name)}"
        path = self.folder / f"{stem}.pdf"
        _guard(path, self.folder)
        fig.savefig(path, format="pdf", dpi=DPI)
        self.entries.append(
            {"index": index, "stem": stem, "name": name, "caption": caption.strip()}
        )
        # Rewritten after every figure, so a notebook that dies halfway still has a manifest
        # describing the figures it did produce.
        manifest_path(self.notebook).write_text(
            json.dumps(self.entries, indent=2), encoding="utf-8"
        )
        return path

    def summary(self) -> str:
        """What was saved, for the notebook's final `print` — so `All_Results.md` says what the run
        drew, not only what it computed."""
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
    """Refuse to write outside this notebook's own folder — a `..` in a figure name would put a
    generated file outside `output/`, the one rule the layout rests on."""
    if folder.resolve() != path.resolve().parent:
        raise ValueError(
            f"figure would be written to {path.resolve()}, outside {folder.resolve()}. "
            f"Figure names are plain names, not paths."
        )
