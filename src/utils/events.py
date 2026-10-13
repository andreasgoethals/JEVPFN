"""Small, persistent operational events; never log row text or provider response bodies."""

from datetime import UTC, datetime
from pathlib import Path

from src.utils.locking import file_lock
from src.utils.serialization import canonical_json


def append_event(path: Path, event: str, **fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp_utc": datetime.now(UTC).isoformat(), "event": event, **fields}
    with file_lock(path.with_suffix(".lock")), path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(record) + "\n")
