"""Small, persistent operational events; never log row text or provider response bodies.

DATA-tier logs stay compact (TEMPLATE `output/`): high-volume per-batch events (`TRACE_EVENTS`)
are appended only to the PROJECT counterpart of the log, `<same relative path>.trace.log` under
`paths.large_outputs_dir()`, while lifecycle events (started/completed/failed) stay in the DATA log.
Locally both tiers are the repository output folder.
"""

from datetime import UTC, datetime
from pathlib import Path

from src.utils.locking import file_lock
from src.utils.serialization import canonical_json

#: Per-batch progress: one line per encoder batch grew a DATA log to 116 MB on VSC.
TRACE_EVENTS = frozenset({"embedding_progress"})
TRACE_SUFFIX = ".trace.log"


def trace_path(path: Path) -> Path:
    """PROJECT counterpart of a DATA-tier log (same relative path, `.trace.log`)."""
    from src.utils import paths

    path = Path(path)
    name = path.name.removesuffix(".log") + TRACE_SUFFIX
    data = paths.outputs_dir()
    if not path.is_relative_to(data):
        return path.with_name(name)
    return paths.large_outputs_dir() / path.relative_to(data).with_name(name)


def append_event(path: Path, event: str, **fields) -> None:
    record = {"timestamp_utc": datetime.now(UTC).isoformat(), "event": event, **fields}
    if event in TRACE_EVENTS:
        try:
            path = trace_path(path)
        except ValueError:
            return  # PROJECT unresolved: drop the detailed trace, never grow the DATA log.
    _append(path, record)


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep operational coordination outside logs, which contain .log files only.
    log_root = next((p for p in path.parents if p.name == "logs"), None)
    lock = (log_root.parent / "manifests" / "event_locks" / path.relative_to(log_root)).with_suffix(".lock") if log_root else path.with_suffix(".lock")
    with file_lock(lock), path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(record) + "\n")
