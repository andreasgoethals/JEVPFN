"""Run every notebook in parallel, then rebuild the two summary documents.

    output/figures/<notebook>/*.pdf, *.png     written by the notebooks themselves
    output/figures/CAPTIONS.md                 ONE file, all notebooks, notebook order
    output/All_Results.md                      every notebook's printed text summary

WHY SEPARATE PROCESSES AND NOT THREADS: matplotlib's figure registry is global state. Two
notebooks in one interpreter would capture each other's figures, and the corruption is
silent — you get plausible figures attributed to the wrong notebook.

WHY A FLATTENED SCRIPT AND NOT A JUPYTER KERNEL: nothing extra to install (no nbclient, no
nbformat), it runs identically on the cluster, and a traceback points at a readable line
number instead of a cell index. The cost is that IPython magics are stripped, which is
deliberate — a notebook that needs a magic to run is a notebook that cannot be executed
non-interactively.

THE RUNNER DOES NOT SAVE FIGURES. Each notebook does that itself through
`src.visualize.figures.FigureSaver`, so an interactive *Run All* produces exactly the same
files. All the runner adds is parallelism and the two concatenated documents.

NOTEBOOKS ARE DISCOVERED, NOT LISTED, and run in alphabetical order — which is also the
order they appear in both summary documents. A hard-coded list is a list that silently
stops covering a notebook someone added.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from src.utils.paths import (
    REPO_ROOT,
    all_results_path,
    captions_path,
    figures_dir,
    notebooks_dir,
)

#: Per-notebook wall-clock limit. A notebook is a summary of a finished computation, not
#: the computation — one that needs longer than this is doing work that belongs in a script.
DEFAULT_TIMEOUT = 1800

#: Where a notebook's captured stdout is parked between execution and assembly. Removed
#: afterwards; `_figures.json` is kept, because CAPTIONS.md must be rebuildable from disk
#: after an interactive run without re-executing anything.
STDOUT_FILE = "_stdout.txt"


@dataclass
class NotebookResult:
    name: str
    ok: bool
    seconds: float
    n_figures: int
    error: str = ""


def discover(names: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """Notebook stems, alphabetical. `names` overrides discovery for a partial rerun."""
    if names:
        return tuple(names)
    return tuple(sorted(p.stem for p in notebooks_dir().glob("*.ipynb")))


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _prelude() -> str:
    """Injected above every flattened notebook.

    `Agg` because a compute node has no display and the default backend would either fail
    or block. stdout is captured so `All_Results.md` can be assembled without the notebook
    knowing it is being run by anything.
    """
    return (
        "import matplotlib\n"
        'matplotlib.use("Agg")\n'
        "import io as _io\n"
        "from contextlib import redirect_stdout as _redirect\n"
        "_TEXT = _io.StringIO()\n"
    )


def _build_script(nb_path: Path, text_path: Path) -> str:
    """Flatten a notebook's code cells into one script under the capture prelude."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    parts = [_prelude(), "\nwith _redirect(_TEXT):\n"]
    for i, cell in enumerate(nb.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        # Strip IPython magics and shell escapes: they are syntax errors in a plain
        # interpreter. A notebook that depends on one cannot be run non-interactively,
        # which the compliance rules already forbid.
        source = re.sub(r"^\s*[%!].*$", "", source, flags=re.M)
        body = "\n".join(f"    {line}" for line in source.split("\n"))
        parts.append(f"\n    # ---- cell {i} ----\n{body}\n")
    parts.append(
        "\nimport pathlib as _pl\n"
        f"_pl.Path(r{str(text_path)!r}).write_text(_TEXT.getvalue(), encoding='utf-8')\n"
    )
    return "".join(parts)


def run_one(name: str, timeout: int = DEFAULT_TIMEOUT) -> NotebookResult:
    """Execute one notebook in a fresh process. Never raises — it reports."""
    started = time.time()
    nb_path = notebooks_dir() / f"{name}.ipynb"
    if not nb_path.is_file():
        return NotebookResult(name, False, 0.0, 0, f"{nb_path} not found")

    out_dir = figures_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    text_path = out_dir / STDOUT_FILE

    # The generated script goes to the system temp dir, NOT into the figure folder: the
    # notebook clears that folder as its first act, and on Windows a directory cannot be
    # modified while it holds the script currently being executed from it.
    tmp = Path(tempfile.gettempdir()) / f"nbrun_{name}.py"
    tmp.write_text(_build_script(nb_path, text_path), encoding="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, str(tmp)],
            cwd=str(REPO_ROOT),   # so `from src...` resolves without an install
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return NotebookResult(name, False, time.time() - started, 0, f"timed out after {timeout}s")
    finally:
        tmp.unlink(missing_ok=True)

    n_figs = len(list(out_dir.glob("*.pdf")))
    if proc.returncode != 0:
        # Only the tail: a full traceback from twelve notebooks buries the one that matters.
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-12:])
        return NotebookResult(name, False, time.time() - started, n_figs, tail)
    return NotebookResult(name, True, time.time() - started, n_figs)


# ---------------------------------------------------------------------------
# The two summary documents
# ---------------------------------------------------------------------------


def _captured_text(name: str) -> str:
    path = figures_dir(name) / STDOUT_FILE
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def write_captions(notebooks: tuple[str, ...]) -> Path:
    """ONE CAPTIONS.md for the whole project, grouped per notebook, in notebook order.

    Built from each notebook's `_figures.json`, so it can be regenerated from disk after an
    interactive run without executing anything. A figure saved without a caption is listed
    with a loud placeholder rather than skipped — a missing caption should be visible in
    the document that is supposed to contain it.
    """
    from src.visualize.figures import read_manifest

    lines = [
        "# Figure captions",
        "",
        "Generated by `python scripts/run_notebooks.py`. Grouped by notebook, figures in",
        "the order that notebook drew them. Caption text is passed to",
        "`FigureSaver.save(..., caption=...)` in the notebook — edits here are overwritten.",
        "",
        "Captions are pure description: what is plotted, on what axes, from how much data.",
        "",
    ]
    for name in notebooks:
        entries = read_manifest(name)
        lines += [f"## {name}", ""]
        if not entries:
            lines += ["_No figures produced._", ""]
            continue
        for e in entries:
            lines.append(f"**{e['stem']}** — `{e['name']}`")
            lines.append("")
            lines.append(e["caption"] or "> MISSING CAPTION. Add one at the `save()` call.")
            lines.append("")
    path = captions_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_all_results(notebooks: tuple[str, ...]) -> Path:
    """Every notebook's printed text summary, concatenated in alphabetical order."""
    lines = [
        "# All results",
        "",
        "Every notebook's printed text summary, concatenated in alphabetical notebook",
        "order. Generated by `python scripts/run_notebooks.py`.",
        "",
    ]
    for name in notebooks:
        text = _captured_text(name).strip()
        lines += ["---", "", f"## {name}", "", "```", text or "(no output captured)", "```", ""]
    path = all_results_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _cleanup(notebooks: tuple[str, ...]) -> None:
    """Drop the captured-stdout scratch files once folded into `All_Results.md`."""
    for name in notebooks:
        (figures_dir(name) / STDOUT_FILE).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# The one entry point
# ---------------------------------------------------------------------------


def run_all(
    notebooks: tuple[str, ...] | None = None,
    max_workers: int | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[NotebookResult]:
    """Run every notebook in parallel, then rebuild CAPTIONS.md and All_Results.md.

    The two documents are rebuilt even when a notebook failed, using whatever the
    successful ones wrote. A half-updated summary is more useful than none, and the
    failure is reported separately rather than by leaving a stale file behind.
    """
    names = discover(notebooks)
    if not names:
        return []
    # Capped at 4: notebooks are numpy-heavy and each already uses several threads, so more
    # workers than this trades parallelism for cache thrashing.
    workers = max_workers or min(len(names), 4)

    results: list[NotebookResult] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, name, timeout): name for name in names}
        for fut in as_completed(futures):
            results.append(fut.result())

    write_captions(names)
    write_all_results(names)
    _cleanup(names)
    return sorted(results, key=lambda r: names.index(r.name))


def summarise(results: list[NotebookResult]) -> str:
    if not results:
        return "No notebooks found in notebooks/."
    lines = ["", "=" * 74, "NOTEBOOK RUN SUMMARY", "=" * 74]
    for r in results:
        lines.append(
            f"  {'OK    ' if r.ok else 'FAILED'} {r.name:<32} "
            f"{r.seconds:6.1f}s  {r.n_figures:2d} figures"
        )
        if not r.ok:
            lines += [f"           {line}" for line in r.error.splitlines()]
    ok = sum(1 for r in results if r.ok)
    lines += [
        "",
        f"{ok}/{len(results)} notebooks OK, {sum(r.n_figures for r in results)} figures",
        f"  figures   -> {figures_dir()}",
        f"  captions  -> {captions_path()}",
        f"  summaries -> {all_results_path()}",
    ]
    return "\n".join(lines)
