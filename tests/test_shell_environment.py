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


def test_new_conda_container_cannot_inherit_host_environment_directory(tmp_path):
    image, host = tmp_path / "image root", tmp_path / "personal DATA"
    image.mkdir()
    host.mkdir()
    (host / "preserve").write_text("host remains untouched")
    driver = tmp_path / "preinstall.sh"
    driver.write_text(
        "set -euo pipefail\n"
        'if [[ "$OSTYPE" == msys* ]]; then CW_INSTALLATION_PATH="$(cygpath -u "$CW_INSTALLATION_PATH")"; fi\n'
        'source "$TEST_HOOK"\n'
        '[[ "$CONDA_ENVS_PATH" == "$CW_INSTALLATION_PATH/miniforge/envs" ]]\n'
        'mkdir "$CONDA_ENVS_PATH/env1"\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        [bash_executable(), str(driver)],
        env=dict(
            os.environ,
            CW_INSTALLATION_PATH=image.as_posix(),
            CONDA_ENVS_PATH=host.as_posix(),
            TEST_HOOK=(ROOT / "scripts/container_conda_pre_install.sh").as_posix(),
        ),
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert (image / "miniforge/envs/env1").is_dir()
    assert list(host.iterdir()) == [host / "preserve"]
    source = (ROOT / "scripts/build_decision_container.sh").read_text()
    assert '--pre-install "$JEVPFN_BUILD_REPO/scripts/container_conda_pre_install.sh"' in source


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
    wrapper.write_text(
        '#!/bin/bash\n'
        'for v in ${!CONDA_@} ${!_CONDA_@}; do echo "Stale host Conda stack: $v" >&2; exit 8; done\n'
        '[[ -z "${_CE_M+x}${_CE_CONDA+x}" ]] || exit 8\n'
        '[[ "$CUDA_VISIBLE_DEVICES" == 3 && "$SLURM_JOB_GPUS" == 3 ]] || exit 8\n'
        'printf "packed-python\\n"\n'
    )
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
        'export CONDA_SHLVL=2 CONDA_PREFIX=/host/env CONDA_PREFIX_1=/host/base\n'
        'export CONDA_DEFAULT_ENV=JEVPFN CONDA_EXE=/host/conda _CONDA_ROOT=/host/base\n'
        'export _CE_M=m _CE_CONDA=conda CUDA_VISIBLE_DEVICES=3 SLURM_JOB_GPUS=3\n'
        '[[ "$TEST_CASE" != native ]] || export JEVPFN_CONDA_ENV=JEVPFN\n'
        'source "$TEST_ACTIVATION"\n'
        '[[ "$JEVPFN_MONITOR_PYTHON" == "$TEST_HOST" ]]\n'
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


def test_gpu_memory_profiles_fit_h100_and_token_job_matches_four_cores():
    gpu_jobs = 0
    for script in (ROOT / "scripts/slurm").glob("*.slurm"):
        text = script.read_text()
        assert "#SBATCH --partition=gpu_a100" not in text
        if "#SBATCH --partition=gpu_h100" in text:
            gpu_jobs += 1
            memory = next(
                line.split("=")[1]
                for line in text.splitlines()
                if line.startswith("#SBATCH --mem=")
            )
            assert memory.endswith("M") and int(memory[:-1]) <= 187200
    assert gpu_jobs == 7
    assert "#SBATCH --mem=13600M" in (ROOT / "scripts/slurm/experiment_0_tokens.slurm").read_text()


def test_a100_override_stops_before_environment_activation(tmp_path):
    driver = tmp_path / "reject.sh"
    driver.write_text((ROOT / "scripts/slurm/_activate_env.sh").read_text(), encoding="utf-8")
    result = subprocess.run(
        [bash_executable(), str(driver)], cwd=tmp_path,
        env={**os.environ, "SLURM_JOB_PARTITION": "gpu_a100_debug"},
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 2
    assert "A100 allocations are disabled" in result.stderr
    assert "No such file" not in result.stderr


def test_all_monitored_jobs_separate_host_monitor_from_workload_interpreter():
    monitored = []
    for script in (ROOT / "scripts/slurm").glob("*.slurm"):
        calls = [line for line in script.read_text().splitlines() if "-m src.cluster.monitor" in line]
        for call in calls:
            assert call.startswith('"${JEVPFN_MONITOR_PYTHON:?}" -m src.cluster.monitor -- ')
            assert '-- "${JEVPFN_PYTHON:?}" -m ' in call
            monitored.append(script.name)
    assert len(monitored) == 8


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


@pytest.mark.parametrize(
    "case",
    [
        "path",
        "bundled",
        "absent",
        "broken",
        "config_error",
        "relative_config",
        "missing_wrapper",
        "module_error",
    ],
)
def test_container_tools_resolve_bundle_or_stop_before_build(tmp_path, case):
    bindir = tmp_path / "isolated bin"
    bundled = tmp_path / "libexec with spaces" / "apptainer/bin"
    bindir.mkdir()
    bundled.mkdir(parents=True)
    commands = {
        "wrap-install": "exit 0\n",
        "conda-containerize": "exit 0\n",
        "apptainer": (
            '[[ "$TEST_CASE" != config_error || "$1" != buildcfg ]] || exit 9\n'
            'if [[ "$1" == buildcfg ]]; then\n'
            '  if [[ "$TEST_CASE" == relative_config ]]; then printf "LIBEXECDIR=relative\\n";\n'
            '  else printf "LIBEXECDIR=%s\\n" "$TEST_LIBEXEC"; fi\n'
            'else printf "apptainer version fixture\\n"; fi\n'
        ),
    }
    for name, content in commands.items():
        executable = bindir / name
        executable.write_text("#!/bin/bash\n" + content, encoding="utf-8")
        executable.chmod(0o755)
    if case == "missing_wrapper":
        (bindir / "wrap-install").unlink()
    if case in {"path", "bundled", "broken"}:
        packer = (bundled if case == "bundled" else bindir) / "mksquashfs"
        packer.write_text(
            '#!/bin/bash\n[[ "$1" == -version ]] || exit 8\n'
            '[[ "$TEST_CASE" != broken ]] || exit 7\nprintf "packer fixture\\n"\n',
            encoding="utf-8",
        )
        packer.chmod(0o755)
    driver = tmp_path / "check.sh"
    driver.write_text(
        "set -euo pipefail\n"
        'if [[ "$OSTYPE" == msys* ]]; then\n'
        '  TEST_LIBEXEC="$(cygpath -u "$TEST_LIBEXEC")"\n'
        '  TEST_BIN="$(cygpath -u "$TEST_BIN")"\nfi\n'
        'export TEST_LIBEXEC PATH="$TEST_BIN"\n'
        'module() { [[ "$*" == "load hpc-container-wrapper/0.4.2" && "$TEST_CASE" != module_error ]]; }\n'
        'source "$TEST_CHECK"\n'
        'mksquashfs -version\nprintf "ready-to-build\\n"\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        [bash_executable(), str(driver)],
        env=dict(
            os.environ,
            TEST_CASE=case,
            TEST_BIN=bindir.as_posix(),
            TEST_LIBEXEC=bundled.parent.parent.as_posix(),
            TEST_CHECK=(ROOT / "scripts/check_container_tools.sh").as_posix(),
        ),
        capture_output=True,
        text=True,
        timeout=20,
    )
    if case in {"path", "bundled"}:
        assert result.returncode == 0, result.stderr
        assert "ready-to-build" in result.stdout
        assert result.stdout.count("packer fixture") == 2
    else:
        assert result.returncode != 0 and "ready-to-build" not in result.stdout
        if case in {"absent", "config_error", "relative_config"}:
            assert "mksquashfs is absent" in result.stderr
        if case == "broken":
            assert "exists but cannot execute" in result.stderr
    build = (ROOT / "scripts/build_decision_container.sh").read_text()
    assert build.index("source scripts/check_container_tools.sh") < build.index("verify_checkpoint")
    assert build.index("source scripts/check_container_tools.sh") < build.index("mktemp -d")


@pytest.mark.parametrize("kind", ["missing", "empty", "nonempty", "linked"])
def test_container_retry_reuses_only_an_empty_destination(tmp_path, kind):
    source = (ROOT / "scripts/build_decision_container.sh").read_text()
    start = source.index('if [[ -d "$JEVPFN_BUILD_DEST"')
    guard = source[start : source.index("# The wICE", start)]
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


@pytest.mark.parametrize("failure", ["none", "copy", "post_install"])
def test_private_wrap_template_preserves_contents_and_executable_bits(tmp_path, failure):
    source = tmp_path / "source with spaces"
    destination = tmp_path / "image with spaces"
    work = tmp_path / "work with spaces"
    for p in (source, destination, work):
        p.mkdir()
    (source / "package.txt").write_text("unchanged package bytes α\n", encoding="utf-8")
    (source / "empty").mkdir()
    (source / "tool").write_text("#!/bin/bash\nprintf 'executable works\\n'\n")
    (source / "tool").chmod(0o755)
    # The actual wrapper writes this hook into the image before copying the environment.
    (destination / "_post_install.sh").write_text(
        "exit 29\n" if failure == "post_install" else "printf 'post install works\\n'\n"
    )
    driver = tmp_path / "wrap.sh"
    driver.write_text(
        "set -euo pipefail\n"
        'if [[ "$OSTYPE" == msys* ]]; then\n'
        '  CW_SOURCE_MOUNT_POINT="$(cygpath -u "$CW_SOURCE_MOUNT_POINT")"\n'
        '  CW_INSTALLATION_PATH="$(cygpath -u "$CW_INSTALLATION_PATH")"\n'
        '  CW_WORKDIR="$(cygpath -u "$CW_WORKDIR")"\nfi\n'
        "cp() {\n"
        '  [[ "$TEST_FAILURE" != copy ]] || return 27\n'
        '  [[ "$1" == -R && "$2" == --preserve=links,timestamps ]] || return 28\n'
        '  command cp "$@"\n}\n'
        'source "$TEST_TEMPLATE"\n'
        '"$CW_INSTALLATION_PATH/tool"\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        [bash_executable(), str(driver)],
        env=dict(
            os.environ,
            CW_SOURCE_MOUNT_POINT=source.as_posix(),
            CW_INSTALLATION_PATH=destination.as_posix(),
            CW_WORKDIR=work.as_posix(),
            TEST_FAILURE=failure,
            TEST_TEMPLATE=(ROOT / "scripts/wrap_environment_template.sh").as_posix(),
        ),
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == {"none": 0, "copy": 27, "post_install": 29}[failure], result.stderr
    assert (source / "package.txt").read_text(encoding="utf-8") == "unchanged package bytes α\n"
    assert not (source / "_post_install.sh").exists()
    if failure != "copy":
        assert (destination / "package.txt").read_bytes() == (source / "package.txt").read_bytes()
        assert (destination / "empty").is_dir()
    if failure == "none":
        assert "post install works" in result.stdout and "executable works" in result.stdout
    else:
        assert "executable works" not in result.stdout


@pytest.mark.skipif(os.name == "nt", reason="POSIX link behavior needs a Linux filesystem")
def test_private_wrap_template_keeps_symlinks_and_hardlinks(tmp_path):
    source, dest = tmp_path / "source", tmp_path / "dest"
    source.mkdir()
    dest.mkdir()
    (source / "original").write_bytes(b"package bytes")
    (source / "alias").symlink_to("original")
    os.link(source / "original", source / "hardlink")
    (dest / "_post_install.sh").write_text(":\n")
    result = subprocess.run(
        [bash_executable(), "-e", str(ROOT / "scripts/wrap_environment_template.sh")],
        env=dict(
            os.environ,
            CW_SOURCE_MOUNT_POINT=str(source),
            CW_INSTALLATION_PATH=str(dest),
            CW_WORKDIR=str(tmp_path),
        ),
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert (dest / "alias").is_symlink() and os.readlink(dest / "alias") == "original"
    assert os.path.samefile(dest / "original", dest / "hardlink")


@pytest.mark.parametrize("changed_template", [False, True])
@pytest.mark.parametrize("existing_alias", [False, True])
def test_private_wrapper_patch_rejects_unknown_site_templates(
    tmp_path, changed_template, existing_alias
):
    script = (ROOT / "scripts/build_decision_container.sh").read_text()
    start = script.index("import sys\nfrom pathlib import Path\nroot,repo=")
    code = script[start : script.index("\nPY", start)]
    wrapper = tmp_path / "private wrapper"
    (wrapper / "templates").mkdir(parents=True)
    (wrapper / "frontends").mkdir()
    original = (
        "cp -a $CW_SOURCE_MOUNT_POINT/. $CW_INSTALLATION_PATH\n"
        "cd $CW_WORKDIR\nsource $CW_INSTALLATION_PATH/_post_install.sh\n"
    )
    template = wrapper / "templates/wrap.sh"
    template.write_text(original + ("unknown change\n" if changed_template else ""))
    alias = wrapper / "frontends/wrap-install"
    if existing_alias:
        alias.write_text("site dispatcher placeholder")
    result = subprocess.run(
        [sys.executable, "-c", code, str(wrapper), str(ROOT)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if changed_template:
        assert result.returncode != 0 and "Site wrap template changed" in result.stderr
        assert template.read_text() == original + "unknown change\n"
        if existing_alias:
            assert alias.read_text() == "site dispatcher placeholder"
        else:
            assert not alias.exists()
    else:
        # Windows may refuse the same native symlink that is available in a Linux allocation.
        if os.name == "nt" and "WinError 1314" in result.stderr:
            pytest.skip("Native symlink creation unavailable")
        assert result.returncode == 0, result.stderr
        assert template.read_bytes() == (ROOT / "scripts/wrap_environment_template.sh").read_bytes()
        assert alias.is_symlink() and os.readlink(alias) == "containerize"


@pytest.mark.parametrize("changed", [False, True])
def test_new_conda_template_uses_image_prefix_even_with_host_env1(tmp_path, changed):
    from src.cluster.container_templates import explicit_conda_prefix

    template = """if false; then
mamba $_EC create --name $CW_ENV_NAME $_FF environment.yaml
else
conda $_EC create --name $CW_ENV_NAME $_FF environment.yaml
fi
conda activate $CW_ENV_NAME
"""
    if changed:
        with pytest.raises(ValueError, match="template changed"):
            explicit_conda_prefix(template.replace("conda activate", "conda run"))
        return
    image = tmp_path / "image"
    (image / "miniforge/bin").mkdir(parents=True)
    executable = image / "miniforge/bin/conda"
    executable.write_text(
        "#!/bin/bash\nset -eu\n"
        '[[ "$1" == create && "$2" == --prefix && "$3" == "$env_root" ]] || exit 61\n'
        'mkdir -p "$3"\nprintf "created image prefix\\n"\n',
        encoding="utf-8",
        newline="\n",
    )
    executable.chmod(0o755)
    driver = tmp_path / "driver.sh"
    driver.write_text(
        "set -euo pipefail\n"
        'if [[ "$OSTYPE" == msys* ]]; then CW_INSTALLATION_PATH="$(cygpath -u "$CW_INSTALLATION_PATH")"; fi\n'
        'export env_root="$CW_INSTALLATION_PATH/miniforge/envs/env1"\n'
        'CW_ENV_NAME=env1; _EC=""; _FF=""\n'
        "# Name-based lookup would select the unrelated host env1: refuse it.\n"
        'conda() { [[ "$1" == activate && "$2" == "$env_root" && -d "$2" ]] || return 62; printf "activated image prefix\\n"; }\n'
        + explicit_conda_prefix(template),
        encoding="utf-8",
        newline="\n",
    )
    result = subprocess.run(
        [bash_executable(), str(driver)],
        env=dict(os.environ, CW_INSTALLATION_PATH=image.as_posix()),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["created image prefix", "activated image prefix"]
