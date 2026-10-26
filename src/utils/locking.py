"""Cross-process file locks, automatically released by the OS after a crash."""

import errno
import os
import threading
import time
import weakref
from contextlib import contextmanager
from pathlib import Path

_mutex_guard = threading.Lock()
_mutexes = weakref.WeakValueDictionary()


def _thread_mutex(path):
    # POSIX record locks belong to a process. Serialize threads *before* opening
    # the lockfile: closing a second descriptor would release its process's lock.
    key = os.path.normcase(str(path.resolve()))
    with _mutex_guard:
        mutex = _mutexes.get(key)
        if mutex is None:
            mutex = threading.Lock()
            _mutexes[key] = mutex
        return mutex


def _after_fork():
    global _mutex_guard, _mutexes
    _mutex_guard = threading.Lock()
    _mutexes = weakref.WeakValueDictionary()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


@contextmanager
def file_lock(path: Path, *, timeout: float = 600, shared: bool = False):
    """POSIX record locks across nodes; Windows retains exclusive byte locks.

    BSD flock was observed to be node-local on the Mindwell GPFS mount. lockf
    requests POSIX record locks instead, still subject to the real two-node gate.
    Shared leases can overlap across processes; within one process we serialize.
    Never unlink a lockfile, and never mix old flock writers with this implementation.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mutex = _thread_mutex(path)
    started = time.monotonic()
    if not mutex.acquire(timeout=max(0, timeout)):
        raise TimeoutError(f"Timed out waiting for {path}")
    try:
        with _locked_handle(path, shared, timeout, started):
            yield
    finally:
        mutex.release()


@contextmanager
def _locked_handle(path, shared, timeout, started):
    with path.open("a+b") as handle:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"0")
            handle.flush()
        if os.name == "nt":
            import msvcrt

            def acquire():
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

            def release():
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            def acquire():
                fcntl.lockf(
                    handle.fileno(),
                    (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB,
                    0,
                    0,
                )

            def release():
                fcntl.lockf(handle.fileno(), fcntl.LOCK_UN, 0, 0)

        while True:
            try:
                acquire()
                break
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise
                if time.monotonic() - started >= timeout:
                    raise TimeoutError(f"Timed out waiting for {path}") from exc
                time.sleep(0.1)
        try:
            yield
        finally:
            release()
