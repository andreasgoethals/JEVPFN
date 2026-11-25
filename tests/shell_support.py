"""Select Bash for shell integration tests, including Git for Windows."""

import os
import shutil
import subprocess
from functools import cache
from pathlib import Path

import pytest


@cache
def _msys_access_denied(bash):
    probe = subprocess.run([bash, "--noprofile", "--norc", "-c", ":"],
                           capture_output=True, timeout=10)
    error = probe.stderr.lower()
    return bool(probe.returncode and b"ntcreatedirectoryobject" in error and b"0xc0000022" in error)


def bash_executable():
    bash = shutil.which("bash")
    if not bash and (git := shutil.which("git")):
        candidate = Path(git).parent.parent / "bin" / "bash.exe"
        if candidate.is_file():
            bash = str(candidate)
    if not bash:
        pytest.skip("Bash is required for shell environment integration checks")
    if os.name == "nt" and _msys_access_denied(bash):
        pytest.skip("Windows denied the MSYS process namespace before Bash could start (0xC0000022)")
    return bash
