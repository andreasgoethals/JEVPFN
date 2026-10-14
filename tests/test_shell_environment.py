"""Reproduce inherited-venv and shell-shadowing failures without installing anything."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def bash_executable():
    bash = shutil.which("bash")
    if not bash and (git := shutil.which("git")):
        candidate = Path(git).parent.parent / "bin" / "bash.exe"
        if candidate.is_file():
            bash = str(candidate)
    if not bash:
        pytest.skip("Bash is required for shell environment integration checks")
    return bash


@pytest.mark.parametrize("failure", ["none", "activation", "interpreter"])
def test_setup_selects_only_verified_conda_python(tmp_path, failure):
    bootstrap = tmp_path / "DATA" / "miniconda3" / "etc" / "profile.d"
    bootstrap.mkdir(parents=True)
    selected = tmp_path / "selected env"
    (selected / "bin").mkdir(parents=True)
    (bootstrap / "conda.sh").write_text(
        "conda() {\n"
        '  if [[ "$TEST_FAILURE" == activation ]]; then return 7; fi\n'
        '  export CONDA_PREFIX="$TEST_ENV"\n'
        "}\n",
        encoding="utf-8",
    )
    interpreter = selected / "bin" / "python"
    interpreter.write_text(
        "#!/bin/bash\n"
        'if [[ "$TEST_FAILURE" == interpreter ]]; then exit 8; fi\n'
        '[[ -z "${VIRTUAL_ENV:-}${PYTHONHOME:-}${PYTHONPATH:-}" ]] || exit 9\n'
        '[[ "$PYTHONNOUSERSITE" == 1 ]] || exit 10\n'
        'printf "verified-python\\n"\n',
        encoding="utf-8",
    )
    interpreter.chmod(0o755)
    driver = tmp_path / "check.sh"
    driver.write_text(
        'if [[ "$OSTYPE" == msys* ]]; then\n'
        '  export VSC_DATA="$(cygpath -u "$VSC_DATA")"\n'
        '  export TEST_ENV="$(cygpath -u "$TEST_ENV")"\n'
        "fi\n"
        "export VIRTUAL_ENV=old PYTHONHOME=old PYTHONPATH=old\n"
        'python() { printf "WRONG-PYTHON\\n"; return 99; }\n'
        'pip() { printf "WRONG-PIP\\n"; return 99; }\n'
        "shopt -s expand_aliases\n"
        'alias python="echo WRONG-ALIAS"\n'
        'if source "$TEST_ACTIVATION"; then\n'
        '  [[ "$JEVPFN_PYTHON" == "$TEST_ENV/bin/python" ]] || exit 11\n'
        "else\n"
        '  [[ -z "${JEVPFN_PYTHON:-}" ]] || exit 12\n'
        '  printf "blocked\\n"\n'
        "fi\n"
        'if [[ "$TEST_FAILURE" == none ]]; then python; fi\n',
        encoding="utf-8",
    )
    env = dict(os.environ)
    env.update(
        VSC_DATA=(tmp_path / "DATA").as_posix(),
        TEST_ENV=selected.as_posix(),
        TEST_FAILURE=failure,
        TEST_ACTIVATION=(ROOT / "scripts" / "activate_conda.sh").as_posix(),
    )
    result = subprocess.run(
        [bash_executable(), str(driver)],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "WRONG" not in result.stdout
    if failure == "none":
        assert result.stdout.count("verified-python") == 2
    else:
        assert result.stdout.strip() == "blocked"
