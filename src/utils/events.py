"""Small, persistent operational events; never log row text or provider response bodies."""

from datetime import UTC, datetime
from pathlib import Path

from src.utils.locking import file_lock
from src.utils.serialization import canonical_json


def append_event(path: Path, event: str, **fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp_utc": datetime.now(UTC).isoformat(), "event": event, **fields}
    # Keep operational coordination outside logs, which contain .log files only.
    log_root = next((p for p in path.parents if p.name == "logs"), None)
    lock = (log_root.parent / "manifests" / "event_locks" / path.relative_to(log_root)).with_suffix(".lock") if log_root else path.with_suffix(".lock")
    with file_lock(lock), path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(record) + "\n")
