"""Atomic output writes that tolerate transient Windows file-sharing conflicts."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path


def atomic_replace(temporary: Path, path: Path) -> None:
    """Retry sharing violations; retain both old output and recovery copy on failure."""
    for attempt in range(7):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as exc:
            if attempt == 6:
                raise PermissionError(
                    f"Cannot replace {path}. Close any application locking it and retry. "
                    f"Any previous output is intact; the new content is saved at {temporary}."
                ) from exc
            time.sleep(min(0.1 * 2**attempt, 1.0))


def atomic_text(path: Path, text: str) -> None:
    """Avoid rewriting identical content, otherwise replace with a completed local file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if path.is_file():
            with path.open(encoding="utf-8", newline="") as existing:
                if existing.read() == text:
                    return
    except PermissionError:
        # An external reader may temporarily block reading as well as replacement.
        pass
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".pending",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    atomic_replace(temporary, path)
