"""Reading inputs and caching the processed form. Replace the bodies; keep the contract.

THE CONTRACT this template asks a project's loaders to keep:

1. **Never build a path.** Ask `src.utils.paths`, so the same call works on a laptop and on
   both cluster tiers.
2. **`data/raw/` is read-only.** Nothing here writes to it, ever.
3. **A cache is only valid once its marker exists.** Write the marker LAST, so a run killed
   halfway leaves an incomplete cache that is correctly treated as absent rather than
   silently reused — a half-written cache that looks complete is a wrong result nobody
   investigates.
4. **Return the same object shape whether the cache hit or missed.** A caller that can tell
   the difference will eventually depend on it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.utils.logging_setup import get_logger
from src.utils.paths import data_search_paths, ensure, find_input, processed_dir

log = get_logger(__name__)

#: Written LAST when a cache is built, and the only thing that makes one count as complete.
CACHE_MARKER = "meta.json"

#: Formats a raw input may arrive in, in preference order.
RAW_SUFFIXES = (".parquet", ".csv")


def raw_path(dataset: str) -> Path:
    """The raw file for a dataset, searching the repo first and then project storage."""
    for suffix in RAW_SUFFIXES:
        found = find_input("raw", f"{dataset}{suffix}")
        if found is not None:
            return found
    # Name every root that was searched. "file not found" without the search path is the
    # least useful error in a two-tier setup, where the answer is usually "the wrong tier".
    searched = ", ".join(str(p) for p in data_search_paths("raw"))
    raise FileNotFoundError(
        f"no raw file for {dataset!r} as {' or '.join(RAW_SUFFIXES)} under: {searched}"
    )


def load_raw(dataset: str) -> pd.DataFrame:
    """Read a raw dataset. No cleaning, no renaming — exactly what is on disk."""
    path = raw_path(dataset)
    log.info("reading raw %s", path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def cache_dir(dataset: str) -> Path:
    """Where this dataset's processed form is written. Project storage on the cluster."""
    return processed_dir(dataset)


def is_cached(dataset: str) -> bool:
    return (cache_dir(dataset) / CACHE_MARKER).is_file()


def preprocess(frame: pd.DataFrame) -> pd.DataFrame:
    """PROJECT-SPECIFIC. Replace this with the real preprocessing.

    Kept as a separate pure function so it can be tested on a small frame without touching
    the filesystem, which is the only way preprocessing bugs get caught early.
    """
    return frame


def load(dataset: str, *, rebuild: bool = False) -> pd.DataFrame:
    """The processed dataset, from cache when valid and rebuilt when not."""
    folder = cache_dir(dataset)
    table = folder / "data.parquet"

    if is_cached(dataset) and not rebuild:
        log.info("cache hit %s", table)
        return pd.read_parquet(table)

    frame = preprocess(load_raw(dataset))
    ensure(table)
    frame.to_parquet(table, index=False)
    # The marker goes last. See contract point 3.
    (folder / CACHE_MARKER).write_text(
        json.dumps({"dataset": dataset, "rows": len(frame), "columns": list(frame.columns)}, indent=2),
        encoding="utf-8",
    )
    log.info("cached %d rows -> %s", len(frame), folder)
    return frame
