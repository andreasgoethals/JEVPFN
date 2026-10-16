#!/bin/bash
# Called by the bounded CPU build allocation; scratch holds all unpacked build files.
set -euo pipefail
JEVPFN_BUILD_PROVIDER="${1:?Pass cloudflare|intern|liquid|strands|perplexity}"
JEVPFN_BUILD_MODE="${2:-wrap}"
case "$JEVPFN_BUILD_PROVIDER" in cloudflare|intern|liquid|strands|perplexity) ;; *) exit 2 ;; esac
case "$JEVPFN_BUILD_MODE" in wrap|new) ;; *) echo 'Mode must be wrap or new.' >&2; exit 2 ;; esac
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Build in a compute allocation.' >&2; exit 2; }
JEVPFN_BUILD_REPO="${VSC_DATA:?}/JEVPFN"
export JEVPFN_STAGING_ROOT="${JEVPFN_STAGING_ROOT:-/lustre1/project/stg_00211}"
export JEVPFN_CONDA_ENV=JEVPFN
cd "$JEVPFN_BUILD_REPO"
source scripts/activate_conda.sh
JEVPFN_BUILD_PYTHON="${JEVPFN_PYTHON:?}"
"$JEVPFN_BUILD_PYTHON" -c 'from src.cluster.freeze import check_if_present; assert not check_if_present()["frozen"], "Runtime builds belong before the final frozen validation"'
"$JEVPFN_BUILD_PYTHON" -c 'import sys; from src.jev.checkpoints import verify_checkpoint; verify_checkpoint(sys.argv[1]); print("Pinned checkpoint verified")' "$JEVPFN_BUILD_PROVIDER"
JEVPFN_BUILD_DEST="$JEVPFN_STAGING_ROOT/JEVPFN/containers/$JEVPFN_BUILD_PROVIDER"
[[ ! -e "$JEVPFN_BUILD_DEST" ]] || { echo "Destination already exists; inspect it before rebuilding: $JEVPFN_BUILD_DEST" >&2; exit 2; }
JEVPFN_BUILD_SCRATCH="${VSC_SCRATCH_NODE:-${VSC_SCRATCH:-${VSC_SCRATCH_LUSTRE1:-}}}"
[[ -n "$JEVPFN_BUILD_SCRATCH" && -d "$JEVPFN_BUILD_SCRATCH" ]] || { echo 'No existing scratch allocation is available.' >&2; exit 2; }
JEVPFN_BUILD_WORK="$(mktemp -d "$JEVPFN_BUILD_SCRATCH/jevpfn-container-$JEVPFN_BUILD_PROVIDER-XXXXXX")"
echo "Build scratch: $JEVPFN_BUILD_WORK"
export APPTAINER_TMPDIR="$JEVPFN_BUILD_WORK/apptainer-tmp"
export APPTAINER_CACHEDIR="$JEVPFN_BUILD_WORK/apptainer-cache"
export PIP_CACHE_DIR="$JEVPFN_BUILD_WORK/pip-cache"
export CONDA_PKGS_DIRS="$JEVPFN_BUILD_WORK/conda-packages"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
module load hpc-container-wrapper/0.4.2
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
    wrap-install --mask -w bin "$JEVPFN_BUILD_SOURCE" --prefix "$JEVPFN_BUILD_DEST"
else
    "$JEVPFN_BUILD_PYTHON" - "$JEVPFN_BUILD_WORK" "$JEVPFN_BUILD_REPO" "$JEVPFN_BUILD_PROVIDER" "$JEVPFN_STAGING_ROOT" <<'PY'
import sys,shlex
from pathlib import Path
work,repo,provider,root=sys.argv[1:]
q=shlex.quote
Path(work,'environment.yaml').write_text('channels: [conda-forge]\ndependencies: [python=3.12, pip]\n',encoding='utf-8')
extra={'cloudflare':'clef','intern':'intern','liquid':'liquid','strands':'strands','perplexity':'token-audit'}[provider]
lines=['set -euo pipefail','export PYTHONNOUSERSITE=1',f'cd {q(repo)}',
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
    conda-containerize new --prefix "$JEVPFN_BUILD_DEST" --post-install "$JEVPFN_BUILD_WORK/post_install.sh" "$JEVPFN_BUILD_WORK/environment.yaml"
fi
cd "$JEVPFN_BUILD_REPO"
if [[ "$JEVPFN_BUILD_MODE" == wrap ]]; then
    "$JEVPFN_BUILD_PYTHON" -m src.cluster.containers seal "$JEVPFN_BUILD_PROVIDER" --wrapped-existing
else
    "$JEVPFN_BUILD_PYTHON" -m src.cluster.containers seal "$JEVPFN_BUILD_PROVIDER"
fi
echo 'Container sealed. No source environment was deleted. Run GPU contracts next.'
