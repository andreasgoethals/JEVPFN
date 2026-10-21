#!/bin/bash
# Called by the bounded CPU build allocation; scratch holds all unpacked build files.
set -euo pipefail
JEVPFN_BUILD_PROVIDER="${1:?Pass cloudflare|intern|liquid|strands|perplexity}"
JEVPFN_BUILD_MODE="${2:-wrap}"
case "$JEVPFN_BUILD_PROVIDER" in
    cloudflare|intern|liquid|strands|perplexity) JEVPFN_BUILD_DECISION=1 ;;
    catboost|tabpfn_3_5|tabicl_v2|limix_2|kumo_tabular|linear|tabstar|contexttab|ag_mm)
        JEVPFN_BUILD_DECISION=0
        [[ "$JEVPFN_BUILD_MODE" == wrap ]] || { echo 'Predictors: wrap a separately reviewed environment; no guessed dependency installation.' >&2; exit 2; } ;;
    *) exit 2 ;;
esac
case "$JEVPFN_BUILD_MODE" in wrap|new) ;; *) echo 'Mode must be wrap or new.' >&2; exit 2 ;; esac
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Build in a compute allocation.' >&2; exit 2; }
JEVPFN_BUILD_REPO="${VSC_DATA:?}/JEVPFN"
export JEVPFN_STAGING_ROOT="${JEVPFN_STAGING_ROOT:-/lustre1/project/stg_00211}"
export JEVPFN_CONDA_ENV=JEVPFN
cd "$JEVPFN_BUILD_REPO"
source scripts/activate_conda.sh
JEVPFN_BUILD_PYTHON="${JEVPFN_PYTHON:?}"
source scripts/check_container_tools.sh
"$JEVPFN_BUILD_PYTHON" -c 'from src.cluster.freeze import check_if_present; assert not check_if_present()["frozen"], "Runtime builds belong before the final frozen validation"'
if [[ "$JEVPFN_BUILD_DECISION" == 1 ]]; then
    "$JEVPFN_BUILD_PYTHON" -c 'import sys; from src.jev.checkpoints import verify_checkpoint; verify_checkpoint(sys.argv[1]); print("Pinned checkpoint verified")' "$JEVPFN_BUILD_PROVIDER"
fi
JEVPFN_BUILD_DEST="$JEVPFN_STAGING_ROOT/JEVPFN/containers/$JEVPFN_BUILD_PROVIDER"
if [[ -d "$JEVPFN_BUILD_DEST" && ! -L "$JEVPFN_BUILD_DEST" ]]; then
    rmdir -- "$JEVPFN_BUILD_DEST" 2>/dev/null || true  # Only an empty failed-build directory.
fi
[[ ! -e "$JEVPFN_BUILD_DEST" && ! -L "$JEVPFN_BUILD_DEST" ]] || { echo "Nonempty or linked destination exists; inspect it before rebuilding: $JEVPFN_BUILD_DEST" >&2; exit 2; }
# The wICE shared Lustre scratch survives a failed allocation for diagnosis.
JEVPFN_BUILD_SCRATCH="${VSC_SCRATCH_LUSTRE1:-${VSC_SCRATCH_NODE:-${VSC_SCRATCH:-}}}"
[[ -n "$JEVPFN_BUILD_SCRATCH" && -d "$JEVPFN_BUILD_SCRATCH" ]] || { echo 'No existing scratch allocation is available.' >&2; exit 2; }
JEVPFN_BUILD_WORK="$(mktemp -d "$JEVPFN_BUILD_SCRATCH/jevpfn-container-$JEVPFN_BUILD_PROVIDER-XXXXXX")"
echo "Build scratch: $JEVPFN_BUILD_WORK"
export APPTAINER_TMPDIR="$JEVPFN_BUILD_WORK/apptainer-tmp"
export APPTAINER_CACHEDIR="$JEVPFN_BUILD_WORK/apptainer-cache"
export PIP_CACHE_DIR="$JEVPFN_BUILD_WORK/pip-cache"
export CONDA_PKGS_DIRS="$JEVPFN_BUILD_WORK/conda-packages"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
JEVPFN_WRAPPER_ROOT="$(dirname "$(dirname "$(readlink -f "$(command -v wrap-install)")")")"
JEVPFN_SITE_CONFIG="${CW_GLOBAL_YAML:-$JEVPFN_WRAPPER_ROOT/default_config}"
if [[ -d "$JEVPFN_SITE_CONFIG" ]]; then JEVPFN_SITE_CONFIG="$JEVPFN_SITE_CONFIG/config.yaml"; fi
[[ -f "$JEVPFN_SITE_CONFIG" ]] || { echo 'No readable wrapper site configuration was found.' >&2; exit 2; }
"$JEVPFN_BUILD_PYTHON" - "$JEVPFN_SITE_CONFIG" "$JEVPFN_BUILD_WORK" <<'PY'
import sys,yaml,shutil
from pathlib import Path
source,work=Path(sys.argv[1]),Path(sys.argv[2])
cfg=yaml.safe_load(source.read_text())
cfg.setdefault('force',{}).update(build_tmpdir_base=str(work),workdir=str(work),share_container=False,
    max_num_cpus=4,num_cpus=4,sqfs_options='-noappend -mem 2G -processors 4',
    singularity_executable_path=shutil.which('apptainer') or 'apptainer')
(work/'site.yaml').write_text(yaml.safe_dump(cfg),encoding='utf-8')
PY
export CW_GLOBAL_YAML="$JEVPFN_BUILD_WORK/site.yaml"
cd "$JEVPFN_BUILD_WORK"
if [[ "$JEVPFN_BUILD_MODE" == wrap ]]; then
    JEVPFN_BUILD_SOURCE="$JEVPFN_STAGING_ROOT/JEVPFN/environments/$JEVPFN_BUILD_PROVIDER"
    [[ -x "$JEVPFN_BUILD_SOURCE/bin/python" ]] || { echo 'No complete source environment; use new mode.' >&2; exit 2; }
    # The site's archival copy fails when propagating native Lustre ACLs into the image.
    # Patch only a private scratch copy; never edit the site installation or source ACLs.
    JEVPFN_PRIVATE_WRAPPER="$JEVPFN_BUILD_WORK/wrapper"
    mkdir "$JEVPFN_PRIVATE_WRAPPER"
    cp -R -- "$JEVPFN_WRAPPER_ROOT/." "$JEVPFN_PRIVATE_WRAPPER"
    "$JEVPFN_BUILD_PYTHON" - "$JEVPFN_PRIVATE_WRAPPER" "$JEVPFN_BUILD_REPO" <<'PY'
import sys
from pathlib import Path
root,repo=map(Path,sys.argv[1:])
template=root/'templates/wrap.sh'
expected='cp -a $CW_SOURCE_MOUNT_POINT/. $CW_INSTALLATION_PATH\ncd $CW_WORKDIR\nsource $CW_INSTALLATION_PATH/_post_install.sh'
if template.read_text().strip()!=expected:
    raise ValueError('Site wrap template changed; review it before applying the ACL workaround')
template.write_bytes((repo/'scripts/wrap_environment_template.sh').read_bytes())
# Keep dispatch inside this private copy even if the site alias was absolute.
alias=root/'frontends/wrap-install'
alias.unlink(missing_ok=True)
alias.symlink_to('containerize')
print('Private scratch wrapper: copy contents/links/times without source ACLs or owners')
PY
    export CW_DEBUG_KEEP_FILES=1
    bash "$JEVPFN_PRIVATE_WRAPPER/frontends/wrap-install" --mask -w bin "$JEVPFN_BUILD_SOURCE" --prefix "$JEVPFN_BUILD_DEST"
else
    # Fail before building/copying an environment if the allocation has no index access.
    "$JEVPFN_BUILD_PYTHON" -m src.cluster.build_network
    "$JEVPFN_BUILD_PYTHON" - "$JEVPFN_BUILD_WORK" "$JEVPFN_BUILD_REPO" "$JEVPFN_BUILD_PROVIDER" "$JEVPFN_STAGING_ROOT" <<'PY'
import sys,shlex
from pathlib import Path
work,repo,provider,root=sys.argv[1:]
q=shlex.quote
Path(work,'environment.yaml').write_text('channels: [conda-forge]\ndependencies: [python=3.12, pip]\n',encoding='utf-8')
extra={'cloudflare':'clef','intern':'intern','liquid':'liquid','strands':'strands','perplexity':'token-audit'}[provider]
prefix_check='import os,sys; from pathlib import Path; expected=Path(os.environ["env_root"]).resolve(); actual=Path(sys.prefix).resolve(); print(f"Image interpreter: {actual}; expected: {expected}"); assert actual==expected, "Conda escaped the container installation root"; assert sys.version_info[:2]==(3,12), "Expected Python 3.12"'
lines=['set -euo pipefail','export PYTHONNOUSERSITE=1',f'python -I -c {q(prefix_check)}', f'cd {q(repo)}',
       f'export JEVPFN_STAGING_ROOT={q(root)}', f'export VSC_DATA={q(str(Path(repo).parent))}']
if provider=='perplexity':
    requirements=Path(root,'JEVPFN/checkpoints/decision_models/perplexity/requirements.txt')
    lines+=['python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126',
            f'python -m pip install -r {q(str(requirements))} -e {q(".[dev,token-audit]")}']
else:
    lines+=['python -m pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu126',
            f'python -m pip install -e {q(f".[dev,token-audit,{extra}]")}']
lines+=['python -m pip check',f'python -m src.cluster.token_audit --check-only --provider {q(provider)}']
Path(work,'post_install.sh').write_text('\n'.join(lines)+'\n',encoding='utf-8')
PY
    # Name-based creation/activation can resolve an existing host env1 despite envs_dirs.
    # Force the absolute image prefix in a private copy, leaving the site install untouched.
    JEVPFN_PRIVATE_WRAPPER="$JEVPFN_BUILD_WORK/wrapper"
    mkdir "$JEVPFN_PRIVATE_WRAPPER"
    cp -R -- "$JEVPFN_WRAPPER_ROOT/." "$JEVPFN_PRIVATE_WRAPPER"
    "$JEVPFN_BUILD_PYTHON" - "$JEVPFN_PRIVATE_WRAPPER" <<'PY'
import sys
from pathlib import Path
from src.cluster.container_templates import explicit_conda_prefix
root=Path(sys.argv[1])
template=root/'templates/conda.sh'
template.write_text(explicit_conda_prefix(template.read_text()),encoding='utf-8')
alias=root/'frontends/conda-containerize'
alias.unlink(missing_ok=True)
alias.symlink_to('containerize')
print('Private Conda template: explicit image prefix for creation and activation')
PY
    bash "$JEVPFN_PRIVATE_WRAPPER/frontends/conda-containerize" new --prefix "$JEVPFN_BUILD_DEST" --pre-install "$JEVPFN_BUILD_REPO/scripts/container_conda_pre_install.sh" --post-install "$JEVPFN_BUILD_WORK/post_install.sh" "$JEVPFN_BUILD_WORK/environment.yaml"
fi
cd "$JEVPFN_BUILD_REPO"
if [[ "$JEVPFN_BUILD_MODE" == wrap ]]; then
    "$JEVPFN_BUILD_PYTHON" -m src.cluster.containers seal "$JEVPFN_BUILD_PROVIDER" --wrapped-existing
else
    "$JEVPFN_BUILD_PYTHON" -m src.cluster.containers seal "$JEVPFN_BUILD_PROVIDER"
fi
echo 'Container sealed. No source environment was deleted. Run GPU contracts next.'
