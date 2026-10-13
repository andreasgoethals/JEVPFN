"""Execute and save a real local notebook using this interpreter's Jupyter kernel."""

from __future__ import annotations

import hashlib
import sys
import threading
import time
import traceback
from pathlib import Path

import nbformat
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpec, KernelSpecManager
from nbclient import NotebookClient

from src.utils.files import atomic_text


class LocalKernelSpecs(KernelSpecManager):
    """Use the active venv, without depending on a separately registered kernel."""

    def get_kernel_spec(self, kernel_name):
        return KernelSpec(
            argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            display_name="JEVPFN",
            language="python",
        )


def execute(path: Path, *, root: Path, capture: Path, backup: Path, timeout: int) -> None:
    original = path.read_bytes()
    atomic_text(backup, original.decode("utf-8"))
    notebook = nbformat.reads(original.decode("utf-8"), as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    name = (
        path.relative_to(root / "notebooks").with_suffix("").as_posix()
        if path.is_relative_to(root / "notebooks")
        else path.stem
    )
    total = sum(c.cell_type == "code" and bool(c.source.strip()) for c in notebook.cells)
    current = {"number": 0, "title": "kernel startup", "started": time.monotonic()}
    stop = threading.Event()

    def progress(message):
        print(f"[{name}] {message}", flush=True)

    def heartbeat():
        while not stop.wait(20):
            progress(
                f"still running {current['title']} ({time.monotonic() - current['started']:.0f}s)"
            )

    def cell_start(cell, cell_index, **_):
        if cell.cell_type != "code" or not cell.source.strip():
            return
        current["number"] += 1
        current["started"] = time.monotonic()
        preceding = [c for c in notebook.cells[:cell_index] if c.cell_type == "markdown"]
        current["title"] = (
            preceding[-1].source.splitlines()[0].lstrip("# ") if preceding else "setup"
        )
        progress(f"cell {current['number']}/{total}: {current['title']}")

    def cell_done(cell, **_):
        if cell.cell_type == "code" and cell.source.strip():
            progress(
                f"cell {current['number']}/{total} finished ({time.monotonic() - current['started']:.1f}s)"
            )

    manager = AsyncKernelManager(kernel_name="python3", kernel_spec_manager=LocalKernelSpecs())
    client = NotebookClient(
        notebook,
        km=manager,
        timeout=timeout,
        resources={"metadata": {"path": str(root)}},
        on_cell_start=cell_start,
        on_cell_executed=cell_done,
        extra_arguments=["--matplotlib=inline"],
    )
    threading.Thread(target=heartbeat, daemon=True).start()
    failure = None
    try:
        client.execute(cleanup_kc=True)
    except BaseException as exc:
        failure = exc
    finally:
        stop.set()
        streams = [
            o.get("text", "")
            for c in notebook.cells
            if c.cell_type == "code"
            for o in c.outputs
            if o.output_type == "stream"
        ]
        atomic_text(capture, "".join(streams))
        executed = nbformat.writes(notebook)
        # Never overwrite edits made by someone while a long notebook was running.
        if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(original).digest():
            recovered = capture.with_suffix(".executed.ipynb")
            atomic_text(recovered, executed)
            raise RuntimeError(
                f"Notebook changed during execution; saved outputs separately at {recovered}"
            )
        atomic_text(path, executed)
        progress(
            "saved executed outputs"
            if failure is None
            else "saved partial outputs, including the error"
        )
    if failure is not None:
        raise failure


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--timeout", type=int, required=True)
    args = parser.parse_args()
    try:
        execute(
            args.path,
            root=args.root,
            capture=args.capture,
            backup=args.backup,
            timeout=args.timeout,
        )
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
