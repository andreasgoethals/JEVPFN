"""Reproduce inherited-venv and shell-shadowing failures without installing anything."""

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
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


@pytest.mark.parametrize("selection", ["packed", "corrupt", "native"])
def test_compact_runtime_selection_is_verified_before_activation(tmp_path, selection):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    stage = tmp_path / "project"
    packed = stage / "JEVPFN/containers/cloudflare"
    (packed / "bin").mkdir(parents=True)
    (packed / "_JEVPFN_CONTAINER.json").write_text("fixture")
    host = tmp_path / "host_python"
    host.write_text(
        '#!/bin/bash\n[[ "$TEST_CASE" != corrupt ]] || exit 9\nprintf "export JEVPFN_CONTAINER_PROVIDER=cloudflare\\nexport JEVPFN_CONTAINER_ID=verified\\n"\n'
    )
    host.chmod(0o755)
    wrapper = packed / "bin/python"
    wrapper.write_text('#!/bin/bash\nprintf "packed-python\\n"\n')
    wrapper.chmod(0o755)
    (repo / "scripts/activate_conda.sh").write_text(
        'export JEVPFN_PYTHON="$TEST_HOST"\nprintf "native-activation\\n"\n'
    )
    driver = tmp_path / "driver.sh"
    driver.write_text(
        "set -euo pipefail\n"
        'if [[ "$OSTYPE" == msys* ]]; then\n'
        '  TEST_REPO="$(cygpath -u "$TEST_REPO")"\n'
        '  TEST_HOST="$(cygpath -u "$TEST_HOST")"\n'
        '  JEVPFN_STAGING_ROOT="$(cygpath -u "$JEVPFN_STAGING_ROOT")"\nfi\n'
        'export TEST_HOST JEVPFN_STAGING_ROOT\ncd "$TEST_REPO"\n'
        'export JEVPFN_CONDA_ENV="$JEVPFN_STAGING_ROOT/JEVPFN/environments/cloudflare"\n'
        '[[ "$TEST_CASE" != native ]] || export JEVPFN_CONDA_ENV=JEVPFN\n'
        'source "$TEST_ACTIVATION"\n'
        'printf "selected-provider=%s\\n" "${JEVPFN_CONTAINER_PROVIDER:-none}"\n',
        encoding="utf-8",
    )
    env = dict(
        os.environ,
        TEST_CASE=selection,
        TEST_REPO=repo.as_posix(),
        TEST_HOST=host.as_posix(),
        JEVPFN_STAGING_ROOT=stage.as_posix(),
        TEST_ACTIVATION=(ROOT / "scripts/activate_runtime.sh").as_posix(),
    )
    result = subprocess.run(
        [bash_executable(), str(driver)], env=env, capture_output=True, text=True, timeout=30
    )
    if selection == "corrupt":
        assert result.returncode != 0 and "selected-provider" not in result.stdout
    else:
        assert result.returncode == 0, result.stderr
        assert (
            f"selected-provider={'cloudflare' if selection == 'packed' else 'none'}"
            in result.stdout
        )


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


@pytest.mark.parametrize(
    "script,args,expected",
    [
        (
            "experiment_0_decision_contracts",
            ["intern"],
            ["intern", "/lustre1/project/stg_00211/JEVPFN/environments/intern"],
        ),
        (
            "experiment_0_decision_pilot",
            ["cloudflare", "/project/env with spaces", "/project/custom"],
            ["cloudflare", "/project/env with spaces", "/project/custom"],
        ),
        (
            "feature_creation",
            ["liquid"],
            ["liquid", "/lustre1/project/stg_00211/JEVPFN/environments/liquid"],
        ),
        (
            "experiment_0_predictor",
            ["catboost", "/project/catboost"],
            ["catboost", "/project/catboost"],
        ),
        ("experiment_0_gpu", ["/project/gpu"], ["/project/gpu"]),
        ("experiment_0_tokens", [], ["JEVPFN", "/lustre1/project/stg_00211"]),
        ("experiment_0_storage", [], ["JEVPFN", "/lustre1/project/stg_00211"]),
        ("experiment_0_cpu", [], ["JEVPFN", "/lustre1/project/stg_00211"]),
    ],
)
def test_job_arguments_survive_clean_submission_environment(tmp_path, script, args, expected):
    # Execute the real script's setup prefix with no inherited project variables.
    source = (ROOT / "scripts/slurm" / f"{script}.slurm").read_text()
    prefix = source.split('cd "${VSC_DATA:?}/JEVPFN"')[0]
    driver = tmp_path / "job.sh"
    driver.write_text(
        prefix
        + '\nprintf "%s\\n" "${JEVPFN_PROVIDER:-}" "${JEVPFN_MODEL:-}" "$JEVPFN_CONDA_ENV" "$JEVPFN_STAGING_ROOT"\n',
        encoding="utf-8",
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith(("JEVPFN_", "VSC_", "SLURM_"))}
    result = subprocess.run(
        [bash_executable(), str(driver), *args], env=env, capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stderr
    assert all(value in result.stdout.splitlines() for value in expected)


def test_gpu_memory_profiles_fit_a100_and_token_job_matches_four_cores():
    for script in (ROOT / "scripts/slurm").glob("*.slurm"):
        text = script.read_text()
        if "#SBATCH --partition=gpu_a100" in text:
            memory = next(
                line.split("=")[1]
                for line in text.splitlines()
                if line.startswith("#SBATCH --mem=")
            )
            assert memory.endswith("M") and int(memory[:-1]) <= 126000
    assert "#SBATCH --mem=13600M" in (ROOT / "scripts/slurm/experiment_0_tokens.slurm").read_text()


def test_perplexity_requirements_guard_runs_before_packages_exist(tmp_path):
    # Execute the real shell's Python guard with site-packages disabled, as in a fresh env.
    source = (ROOT / "scripts/setup_decision_env.sh").read_text()
    command = next(line for line in source.splitlines() if "Publisher requirements changed" in line)
    code = shlex.split(command)[2]
    requirements = tmp_path / "requirements.txt"
    requirements.write_bytes(b"example==1.0\n")
    (tmp_path / "_JEVPFN_CHECKPOINT.json").write_text(
        json.dumps(
            {
                "files": {
                    "requirements.txt": hashlib.sha256(requirements.read_bytes()).hexdigest(),
                }
            }
        )
    )
    call = [sys.executable, "-S", "-c", code, str(requirements)]
    assert subprocess.run(call, capture_output=True, timeout=10).returncode == 0
    requirements.write_bytes(b"different==2.0\n")
    failed = subprocess.run(call, capture_output=True, text=True, timeout=10)
    assert failed.returncode != 0 and "Publisher requirements changed" in failed.stderr


def test_unknown_runtime_is_rejected_before_any_install():
    result = subprocess.run(
        [bash_executable(), str(ROOT / "scripts/setup_decision_env.sh"), "unknown"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2 and "Unknown selected" in result.stderr


@pytest.mark.parametrize("kind", ["missing", "empty", "nonempty", "linked"])
def test_container_retry_reuses_only_an_empty_destination(tmp_path, kind):
    source = (ROOT / "scripts/build_decision_container.sh").read_text()
    start = source.index('if [[ -d "$JEVPFN_BUILD_DEST"')
    guard = source[start:source.index("# The wICE", start)]
    destination = tmp_path / "container with spaces"
    if kind in {"empty", "nonempty"}:
        destination.mkdir()
    if kind == "nonempty":
        (destination / "image.sif").write_text("preserve")
    if kind == "linked":
        outside = tmp_path / "outside"
        outside.mkdir()
        try:
            destination.symlink_to(outside, target_is_directory=True)
        except OSError:
            pytest.skip("Symlink creation unavailable")
    env = dict(os.environ, JEVPFN_BUILD_DEST=destination.as_posix())
    result = subprocess.run(
        [bash_executable(), "-c", guard], env=env, capture_output=True, text=True, timeout=15
    )
    assert result.returncode == (2 if kind in {"nonempty", "linked"} else 0)
    if kind == "nonempty":
        assert (destination / "image.sif").read_text() == "preserve"
    if kind == "linked":
        assert destination.is_symlink() and outside.is_dir()
