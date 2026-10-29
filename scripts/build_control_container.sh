#!/bin/bash
# Recreate the recorded CPU dependencies; no populated environment is relocated.
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use a CPU Slurm build allocation.' >&2; exit 2; }
cd "${VSC_DATA:?}/JEVPFN"
export JEVPFN_CONTROL_LEGACY=1 JEVPFN_CONDA_ENV=JEVPFN
export JEVPFN_STAGING_ROOT="${JEVPFN_PROJECT_ROOT:-/lustre1/project/stg_00211}"
source scripts/activate_conda.sh
source scripts/check_container_tools.sh
control_repo="$PWD"
control_dest="$JEVPFN_STAGING_ROOT/JEVPFN/containers/control-v1"
[[ ! -e "$control_dest" && ! -L "$control_dest" ]] || { echo 'Control destination exists; inspect before rebuilding.' >&2; exit 2; }
control_work="$(mktemp -d "${VSC_SCRATCH_LUSTRE1:?}/jevpfn-control-build-XXXXXX")"
export APPTAINER_TMPDIR="$control_work/apptainer-tmp" APPTAINER_CACHEDIR="$control_work/apptainer-cache"
export PIP_CACHE_DIR="$control_work/pip-cache" CONDA_PKGS_DIRS="$control_work/conda-packages"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
control_wrapper="$(dirname "$(dirname "$(readlink -f "$(command -v conda-containerize)")")")"
control_config="${CW_GLOBAL_YAML:-$control_wrapper/default_config}"
if [[ -d "$control_config" ]]; then control_config="$control_config/config.yaml"; fi
cp -R "$control_wrapper" "$control_work/wrapper"
"$JEVPFN_PYTHON" - "$control_work" "$control_repo" "$control_config" <<'PY'
import shlex,sys,yaml,shutil
from pathlib import Path
from src.cluster.container_templates import explicit_conda_prefix,pinned_miniforge
from src.utils.config import load_yaml
work,repo,site=map(Path,sys.argv[1:]); cfg=load_yaml('experiment_0/control_runtime')
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
"$JEVPFN_PYTHON" -m src.cluster.control_runtime seal "$control_dest"
printf 'CPU image built and sealed; legacy environment retained until acceptance.\n'
