import errno
import os
import subprocess
import sys
import threading

import pytest

from src.utils.locking import file_lock


def test_threads_cannot_enter_same_lock_and_timeout_does_not_release_holder(tmp_path):
    path = tmp_path / "writer.lock"
    errors = []

    def contender():
        try:
            with file_lock(path, timeout=0.05):
                errors.append("acquired")
        except TimeoutError:
            errors.append("blocked")

    with file_lock(path):
        for _ in range(2):
            thread = threading.Thread(target=contender)
            thread.start()
            thread.join(3)
            assert not thread.is_alive()
    assert errors == ["blocked", "blocked"]
    with file_lock(path, timeout=0):
        pass


def test_process_exclusion_and_release(tmp_path):
    path = tmp_path / "writer.lock"
    code = "from src.utils.locking import file_lock; import sys\ntry:\n with file_lock(sys.argv[1],timeout=.1): print('acquired')\nexcept TimeoutError: print('blocked')"
    with file_lock(path):
        result = subprocess.run(
            [sys.executable, "-c", code, str(path)], capture_output=True, text=True, check=True
        )
        assert result.stdout.strip() == "blocked"
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "acquired"


@pytest.mark.skipif(os.name == "nt", reason="POSIX lock backend is tested in VSC CPU suite")
def test_posix_uses_record_lock_and_does_not_hide_io_error(tmp_path, monkeypatch):
    import fcntl

    calls = []

    def acquire(*args):
        calls.append(args)
        raise OSError(errno.EIO, "filesystem error")

    monkeypatch.setattr(fcntl, "lockf", acquire)
    with pytest.raises(OSError, match="filesystem error"), file_lock(tmp_path / "lock", timeout=10):
        pytest.fail("Must not acquire")
    assert len(calls) == 1


@pytest.mark.skipif(os.name == "nt", reason="Windows serializes shared leases")
def test_shared_posix_readers_overlap_but_writer_blocks(tmp_path):
    path = tmp_path / "writer.lock"
    code = "from src.utils.locking import file_lock; import sys\ntry:\n with file_lock(sys.argv[1],shared=sys.argv[2]=='yes',timeout=.1): print('acquired')\nexcept TimeoutError: print('blocked')"
    with file_lock(path, shared=True):
        for shared, expected in (("yes", "acquired"), ("no", "blocked")):
            p = subprocess.run(
                [sys.executable, "-c", code, str(path), shared],
                capture_output=True,
                text=True,
                check=True,
            )
            assert p.stdout.strip() == expected
