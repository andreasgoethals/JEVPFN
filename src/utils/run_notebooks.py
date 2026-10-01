"""Run every notebook in parallel, then rebuild the two summary documents.

    python -m src.utils.run_notebooks                     every notebook in notebooks/
    python -m src.utils.run_notebooks --only exploration/01_data_exploration
    python -m src.utils.run_notebooks --summaries-only    rebuild the two .md files only

    output_JEVPFN/figures/<phase>/<notebook>/*.pdf   written by the notebooks themselves
    output_JEVPFN/Captions.md                       all figure captions
    output_JEVPFN/All Results.md             every notebook's printed summary, alphabetical

Each notebook executes in a separate Python subprocess: matplotlib's figure registry must
never be shared between notebooks. Lightweight coordinator threads launch those subprocesses,
avoiding an extra Python process pool and its Windows startup/memory overhead.

A FLATTENED SCRIPT, NOT A JUPYTER KERNEL: nothing extra to install, identical on the cluster,
and a traceback points at a line number instead of a cell index. Magics are stripped, which is
deliberate — a notebook needing one cannot be executed non-interactively at all.

THE RUNNER DOES NOT SAVE FIGURES; each notebook does, through `FigureSaver`, so an interactive
*Run All* produces exactly the same PDFs. The runner adds parallelism and the two documents.

NOTEBOOKS ARE DISCOVERED, NOT LISTED, alphabetically — which is also the order in both summary
documents. A hard-coded list silently stops covering a notebook someone added.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from src.utils.notebook_report import atomic_text
from src.utils.paths import (
    REPO_ROOT,
    all_results_path,
    captions_path,
    figures_dir,
    logs_dir,
    manifests_dir,
    notebook_phase,
    notebook_stem,
    notebooks_dir,
    reports_dir,
)
from src.utils.serialization import write_json

#: Per-notebook wall-clock limit. A notebook summarises a finished computation; one needing
#: longer is doing work that belongs in a script.
DEFAULT_TIMEOUT = 1800

#: Legacy capture filename, still readable by the summary builder. New reports are durable
#: under each phase's reports/, with process stdout/stderr in logs/.
STDOUT_FILE = "_stdout.txt"


def worker_count(n_notebooks: int, requested: int | None = None) -> int:
    """Use available CPUs automatically, including affinity and Slurm allocation limits."""
    available = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        # Some hosts expose affinity without allowing it to be queried.
        with suppress(OSError):
            available = min(available, len(os.sched_getaffinity(0)))
    allocated = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocated is not None:
        if not allocated.isdigit() or int(allocated) < 1:
            raise ValueError("SLURM_CPUS_PER_TASK must be a positive integer")
        available = min(available, int(allocated))
    if requested is not None and requested < 1:
        raise ValueError("workers must be positive")
    return max(1, min(n_notebooks, available, requested or available))


@dataclass
class NotebookResult:
    name: str
    ok: bool
    seconds: float
    n_figures: int
    error: str = ""


def discover(names: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """Recursively discover phase/name identifiers; accept an unambiguous legacy stem."""
    found = tuple(
        sorted(
            p.relative_to(notebooks_dir()).with_suffix("").as_posix()
            for p in notebooks_dir().rglob("*.ipynb")
            if ".ipynb_checkpoints" not in p.parts
        )
    )
    if not names:
        return found
    selected = []
    for name in names:
        notebook_phase(name)  # Reject absolute paths and traversal before resolution.
        matches = [n for n in found if n == name or notebook_stem(n) == name]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous notebook {name}; use phase/name: {matches}")
        selected.append(matches[0] if matches else name)
    return tuple(dict.fromkeys(selected))


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _prelude() -> str:
    """Injected above every flattened notebook. `Agg` because a compute node has no display, and
    stdout is captured so `All Results.md` can be built without the notebook knowing."""
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
    name = discover((name,))[0]
    stem = notebook_stem(name)
    nb_path = notebooks_dir() / f"{name}.ipynb"
    if not nb_path.is_file():
        return NotebookResult(name, False, 0.0, 0, f"{nb_path} not found")

    out_dir = figures_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    phase = notebook_phase(name)
    log_dir = logs_dir(phase)
    log_dir.mkdir(parents=True, exist_ok=True)
    text_path = log_dir / f"{stem}.stdout.txt"
    atomic_text(reports_dir(name), f"RUNNING: {name}; previous report superseded.")

    # The generated script goes to the system temp dir, NOT into the figure folder: the
    # notebook clears that folder as its first act, and on Windows a directory cannot be
    # modified while it holds the script currently being executed from it.
    with tempfile.NamedTemporaryFile(prefix=f"nbrun_{stem}_", suffix=".py", delete=False) as handle:
        tmp = Path(handle.name)
    tmp.write_text(_build_script(nb_path, text_path), encoding="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, str(tmp)],
            cwd=str(REPO_ROOT),  # so `from src...` resolves without an install
            capture_output=True,
            text=True,
            encoding="utf-8",
            env={
                **os.environ,
                "PYTHONIOENCODING": "utf-8",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            },
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        atomic_text(reports_dir(name), f"FAILED: {name}; timed out after {timeout}s")
        write_json(
            manifests_dir(phase) / f"{stem}_run.json",
            {"name": name, "ok": False, "error": "timeout"},
        )
        return NotebookResult(name, False, time.time() - started, 0, f"timed out after {timeout}s")
    finally:
        tmp.unlink(missing_ok=True)

    atomic_text(log_dir / f"{stem}.stderr.txt", proc.stderr or "")
    write_json(
        manifests_dir(phase) / f"{stem}_run.json",
        {
            "name": name,
            "ok": proc.returncode == 0,
            "seconds": time.time() - started,
            "interpreter": sys.executable,
            "phase": phase,
        },
    )
    n_figs = len(list(out_dir.glob("*.pdf")))
    if proc.returncode != 0:
        # Only the tail: a full traceback from twelve notebooks buries the one that matters.
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-12:])
        atomic_text(reports_dir(name), f"FAILED: {name}\n{tail}")
        return NotebookResult(name, False, time.time() - started, n_figs, tail)
    if reports_dir(name).read_text(encoding="utf-8").startswith("RUNNING:"):
        atomic_text(reports_dir(name), text_path.read_text(encoding="utf-8"))
    return NotebookResult(name, True, time.time() - started, n_figs)


# ---------------------------------------------------------------------------
# The two summary documents
# ---------------------------------------------------------------------------


def _captured_text(name: str) -> str:
    path = reports_dir(name)
    if not path.is_file():
        path = figures_dir(name) / STDOUT_FILE
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def write_captions(notebooks: tuple[str, ...], *, phase: str | None = None) -> Path:
    """ONE Captions.md for the project, grouped per notebook, in notebook order.

    Built from each `_figures.json`, so it regenerates from disk after an interactive run. A
    figure with no caption gets a loud placeholder rather than being skipped — a gap should be
    visible in the document meant to contain it.
    """
    from src.visualize.figures import read_manifest

    lines = [
        "# Figure captions",
        "",
        "Generated by `python -m src.utils.run_notebooks`. Grouped by notebook, figures in",
        "the order that notebook drew them. Caption text is passed to",
        "`FigureSaver.save(..., caption=...)` in the notebook — edits here are overwritten.",
        "",
        "These are the paper's captions: paste one straight under its figure. Pure description",
        "— what is plotted, on what axes, from how much data. No interpretation.",
        "",
        "Figures are PDFs. Inspection dashboards use a larger canvas for readable labels;",
        "do not shrink those dashboards to fit a paper page.",
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
    path = captions_path(phase)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_text(path, "\n".join(lines))
    if phase is None:
        for group in sorted({notebook_phase(name) for name in notebooks}):
            write_captions(
                tuple(name for name in notebooks if notebook_phase(name) == group), phase=group
            )
    return path


def write_all_results(notebooks: tuple[str, ...], *, phase: str | None = None) -> Path:
    """Every notebook's printed summary, concatenated. The shape is fixed:

    one block per notebook, **sorted alphabetically by notebook name**; each block is that
    notebook's printed summary **verbatim**, not a rewrite; and that summary follows the
    notebook's own section order, so the file and the notebook read the same way round.

    Verbatim matters: the moment this file paraphrases, the two disagree and the notebook wins —
    but this file is the one anybody actually reads.
    """
    names = tuple(sorted(notebooks))
    lines = [
        "# All Results",
        "",
        "Every notebook's printed summary, verbatim, one block per notebook in alphabetical",
        "order. Each block follows that notebook's own section order.",
        "Generated by `python -m src.utils.run_notebooks`.",
        "",
    ]
    for name in names:
        text = _captured_text(name).strip()
        lines += ["---", "", f"## {name}", "", "```", text or "(no output captured)", "```", ""]
    path = all_results_path(phase)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_text(path, "\n".join(lines))
    if phase is None:
        for group in sorted({notebook_phase(name) for name in notebooks}):
            write_all_results(
                tuple(name for name in notebooks if notebook_phase(name) == group), phase=group
            )
    return path


# ---------------------------------------------------------------------------
# The one entry point
# ---------------------------------------------------------------------------


def run_all(
    notebooks: tuple[str, ...] | None = None,
    max_workers: int | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[NotebookResult]:
    """Run every notebook in parallel, then rebuild both summary documents.

    Rebuilt even when a notebook failed, from whatever the successful ones wrote: a
    half-updated summary beats a stale one, and the failure is reported separately.
    """
    names = discover(notebooks)
    if not names:
        return []
    workers = worker_count(len(names), max_workers)

    results: list[NotebookResult] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, name, timeout): name for name in names}
        for fut in as_completed(futures):
            results.append(fut.result())

    from src.utils.locking import file_lock
    from src.utils.paths import outputs_dir

    with file_lock(outputs_dir() / ".reports.lock"):
        all_names = tuple(sorted(set(discover()) | set(names)))
        write_captions(all_names)
        write_all_results(all_names)
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


# ---------------------------------------------------------------------------
# Entry point. `--summaries-only` exists because both documents are built from what the notebooks
# left on disk (`_figures.json`), so after an interactive Jupyter session they can be regenerated
# without executing anything.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", nargs="+", metavar="STEM", help="notebook stems to run")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="optional cap; defaults to available CPUs/notebooks",
    )
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="seconds per notebook")
    parser.add_argument(
        "--summaries-only",
        action="store_true",
        help="rebuild both documents from disk, run nothing",
    )
    args = parser.parse_args(argv)

    names = discover(tuple(args.only) if args.only else None)
    if not names:
        print("No notebooks found in notebooks/.")
        return 0

    if args.summaries_only:
        from src.utils.locking import file_lock
        from src.utils.paths import outputs_dir

        print(f"Rebuilding summaries from disk for: {', '.join(names)}")
        all_names = tuple(sorted(set(discover()) | set(names)))
        with file_lock(outputs_dir() / ".reports.lock"):
            print(f"  captions  -> {write_captions(all_names)}")
            print(f"  summaries -> {write_all_results(all_names)}")
        return 0

    print(
        f"Running {len(names)} notebook(s) with {worker_count(len(names), args.workers)} parallel processes: {', '.join(names)}",
        flush=True,
    )
    results = run_all(names, max_workers=args.workers, timeout=args.timeout)
    print(summarise(results))
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
