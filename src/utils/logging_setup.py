"""One logger, configured once. Writes to the console and to `output/logs/`.

WHY A FILE AS WELL AS THE CONSOLE: on the cluster the console is a SLURM `.out` file that lands
wherever the job script put it, and after a requeue it is a *different* file. A `.log` under
`output/logs/` is where the code decides, on the tier you can browse, and it survives the job that
produced it. `output/logs/` holds `.log` files and nothing else, so "read the logs" is unambiguous.

WHY `force=True` and an idempotence guard: a notebook cell that calls this twice would
otherwise attach a second handler and print every line twice — which reads like the code
running twice.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from src.utils.paths import logs_dir

_CONFIGURED = False

#: Level name on the left so a log is greppable by level, and the module name so a line
#: says which part of `src/` produced it. No colour codes: they become escape-sequence
#: noise in a SLURM log file.
_FORMAT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"
_DATEFMT = "%d-%m-%Y %H:%M:%S"


def setup(level: str = "INFO", *, run_id: str | None = None, to_file: bool = True) -> Path | None:
    """Configure the root logger once. Returns the log file path, or None.

    `run_id` names the file so a sweep's points do not overwrite each other. Without one the
    timestamp is the name, which is enough for an interactive session.
    """
    global _CONFIGURED
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    path: Path | None = None

    if to_file:
        logs_dir().mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = logs_dir() / (f"{run_id}_{stamp}.log" if run_id else f"{stamp}.log")
        handlers.append(logging.FileHandler(path, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=_FORMAT,
        datefmt=_DATEFMT,
        handlers=handlers,
        force=True,
    )
    _CONFIGURED = True
    return path


def get_logger(name: str) -> logging.Logger:
    """A module logger. Call as `get_logger(__name__)`.

    Configures logging on first use if nobody has, so importing a module and calling it
    from a REPL still produces visible output instead of silence.
    """
    if not _CONFIGURED:
        setup(to_file=False)
    return logging.getLogger(name)
