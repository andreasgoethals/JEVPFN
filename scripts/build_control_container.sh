#!/bin/bash
jevpfn_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]//\\//}")" && pwd)"
source "$jevpfn_script_dir/storage_environment.sh" || return 2 2>/dev/null || exit 2
# Recreate the recorded CPU dependencies; no populated environment is relocated.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use a CPU Slurm build allocation.' >&2; exit 2; }
cd "${VSC_DATA:?}/JEVPFN"
export JEVPFN_STAGING_ROOT="${JEVPFN_DURABLE_ROOT:?Select the allocated durable PROJECT root}"
control_parent="$JEVPFN_DURABLE_ROOT/JEVPFN/containers"
if [[ -n "${JEVPFN_BUILD_PYTHON:-}" ]]; then
    control_python="$JEVPFN_BUILD_PYTHON"
elif [[ -f "$control_parent/control.active.json" ]]; then
    export JEVPFN_CONDA_ENV=JEVPFN
    source scripts/activate_control.sh "$control_parent/control-v1"
    control_python="${JEVPFN_PYTHON:?}"
else
    control_python="$(command -v python3)"
fi
"$control_python" -c 'from src.utils.config import load_yaml; load_yaml("0/control_runtime")' || {
    echo 'Choose a build Python with the project control dependencies using JEVPFN_BUILD_PYTHON.' >&2; exit 2; }
source scripts/check_container_tools.sh
if [[ -n "${JEVPFN_CONTROL_IMAGE:-}" ]]; then
    JEVPFN_BUILD_PYTHON="$control_python"
    source scripts/container_build_path.sh
fi
control_repo="$PWD"
control_name="${JEVPFN_CONTROL_BUILD_NAME:-control-v1}"
[[ "$control_name" =~ ^control-v1(-[a-zA-Z0-9_-]+)?$ ]] || { echo 'Invalid control image name.' >&2; exit 2; }
control_dest="$control_parent/$control_name"
[[ ! -e "$control_dest" && ! -L "$control_dest" ]] || { echo 'Control destination exists; inspect before rebuilding.' >&2; exit 2; }
control_work="$(mktemp -d "${VSC_SCRATCH_GPFS1:?Mindwell requires native GPFS scratch}/jevpfn-control-build-XXXXXX")"
export APPTAINER_TMPDIR="$control_work/apptainer-tmp" APPTAINER_CACHEDIR="$control_work/apptainer-cache"
export PIP_CACHE_DIR="$control_work/pip-cache" CONDA_PKGS_DIRS="$control_work/conda-packages"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
control_wrapper="$(dirname "$(dirname "$(readlink -f "$(command -v conda-containerize)")")")"
control_config="${CW_GLOBAL_YAML:-$control_wrapper/default_config}"
if [[ -d "$control_config" ]]; then control_config="$control_config/config.yaml"; fi
cp -R "$control_wrapper" "$control_work/wrapper"
"$control_python" - "$control_work" "$control_repo" "$control_config" <<'PY'
import shlex,sys,yaml,shutil
from pathlib import Path
from src.cluster.container_templates import explicit_conda_prefix,pinned_miniforge
from src.utils.config import load_yaml
work,repo,site=map(Path,sys.argv[1:]); cfg=load_yaml('0/control_runtime')
settings=yaml.safe_load(site.read_text())
settings.setdefault('force',{}).update(build_tmpdir_base=str(work),workdir=str(work),share_container=False,
    max_num_cpus=4,num_cpus=4,sqfs_options='-noappend -mem 2G -processors 4',singularity_executable_path=shutil.which('apptainer'))
(work/'site.yaml').write_text(yaml.safe_dump(settings))
(work/'environment.yaml').write_text(yaml.safe_dump(dict(channels=['conda-forge'],dependencies=['python='+cfg['python'],'pip'])))
(work/'requirements.txt').write_text('\n'.join(k+'=='+v for k,v in cfg['packages'].items())+'\n')
q=shlex.quote
(work/'post_install.sh').write_text('set -euo pipefail\nexport PYTHONNOUSERSITE=1\n'
    +'python -m pip install -r '+q(str(work/'requirements.txt'))+'\n'
    +'python -m pip install --no-deps -e '+q(str(repo))+'\npython -m pip check\n')
template=work/'wrapper/templates/conda.sh'
template.write_text(pinned_miniforge(explicit_conda_prefix(template.read_text())))
alias=work/'wrapper/frontends/conda-containerize'; alias.unlink(missing_ok=True); alias.symlink_to('containerize')
PY
export CW_GLOBAL_YAML="$control_work/site.yaml"
cd "$control_work"
bash "$control_work/wrapper/frontends/conda-containerize" new --prefix "$control_dest" --pre-install "$control_repo/scripts/container_conda_pre_install.sh" --post-install "$control_work/post_install.sh" "$control_work/environment.yaml"
cd "$control_repo"
"$control_python" -m src.cluster.control_runtime seal "$control_dest"
printf 'CPU image built and sealed at %s.\n' "$control_dest"
