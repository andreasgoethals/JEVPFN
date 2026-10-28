#!/bin/bash -l
# Historical native environments. Current B200 setup uses setup_extractors and sealed images.
set -euo pipefail
JEVPFN_SETUP_PROVIDER="${1:?Usage: bash scripts/setup_decision_env.sh cloudflare|intern|liquid|strands|perplexity|h2o|sii|kev|decision3|nimble}"
case "$JEVPFN_SETUP_PROVIDER" in
    liquid_lfm3) echo 'd1-2B requires its isolated Blackwell container. Use python -m src.cluster.setup_extractors --provider liquid_lfm3 --execute --watch.' >&2; exit 2 ;;
    cloudflare) JEVPFN_SETUP_EXTRA=clef ;;
    intern) JEVPFN_SETUP_EXTRA=intern ;;
    liquid) JEVPFN_SETUP_EXTRA=liquid ;;
    strands) JEVPFN_SETUP_EXTRA=strands ;;
    perplexity) JEVPFN_SETUP_EXTRA=perplexity ;;
    h2o|sii|kev|decision3|nimble) JEVPFN_SETUP_EXTRA=decision-open ;;
    *) echo 'Unknown selected decision provider.' >&2; exit 2 ;;
esac
cd "${VSC_DATA:?}/JEVPFN"
export JEVPFN_STAGING_ROOT="${JEVPFN_STAGING_ROOT:-/lustre1/project/stg_00211}"
export JEVPFN_CONDA_ENV=JEVPFN
source scripts/activate_conda.sh
"${JEVPFN_PYTHON:?}" -c 'from src.cluster.maintenance import require_idle_vsc; require_idle_vsc()'
export CONDA_PKGS_DIRS="$JEVPFN_STAGING_ROOT/JEVPFN/checkpoints/conda_packages"
export PIP_CACHE_DIR="$JEVPFN_STAGING_ROOT/JEVPFN/checkpoints/pip_cache"
export HF_HOME="$JEVPFN_STAGING_ROOT/JEVPFN/checkpoints/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
"$JEVPFN_PYTHON" -m src.jev.checkpoints --provider "$JEVPFN_SETUP_PROVIDER"
export JEVPFN_CONDA_ENV="$JEVPFN_STAGING_ROOT/JEVPFN/environments/$JEVPFN_SETUP_PROVIDER"
if [[ ! -x "$JEVPFN_CONDA_ENV/bin/python" ]]; then
    conda create --yes --prefix "$JEVPFN_CONDA_ENV" python=3.12 pip
fi
source scripts/activate_conda.sh
if [[ "$JEVPFN_SETUP_PROVIDER" == perplexity ]]; then
    # Publisher requirements differ from other environments. Verify before installing.
    JEVPFN_REQUIREMENTS="$JEVPFN_STAGING_ROOT/JEVPFN/checkpoints/decision_models/perplexity/requirements.txt"
    # A fresh environment has no project dependencies yet: this check uses only stdlib.
    "$JEVPFN_PYTHON" -c 'import hashlib,json,sys; from pathlib import Path; p=Path(sys.argv[1]); m=json.loads((p.parent/"_JEVPFN_CHECKPOINT.json").read_text()); actual=hashlib.sha256(p.read_bytes()).hexdigest(); sys.exit(0 if actual==m["files"]["requirements.txt"] else "Publisher requirements changed")' "$JEVPFN_REQUIREMENTS"
    "${JEVPFN_PYTHON:?}" -m pip install 'torch==2.14.0' 'torchvision==0.29.0' --index-url https://download.pytorch.org/whl/cu126
    "$JEVPFN_PYTHON" -m pip install -r "$JEVPFN_REQUIREMENTS" -e '.[dev,token-audit]'
else
    "${JEVPFN_PYTHON:?}" -m pip install 'torch==2.12.1' 'torchvision==0.27.1' --index-url https://download.pytorch.org/whl/cu126
    "$JEVPFN_PYTHON" -m pip install -e ".[dev,token-audit,$JEVPFN_SETUP_EXTRA]"
fi
"$JEVPFN_PYTHON" -m pip check
"$JEVPFN_PYTHON" -m src.cluster.token_audit --check-only --provider "$JEVPFN_SETUP_PROVIDER"
"$JEVPFN_PYTHON" -m src.cluster.runtime_setup "$JEVPFN_SETUP_PROVIDER"
if [[ "$JEVPFN_SETUP_PROVIDER" == strands ]]; then
    echo 'Strands: causal-conv1d acceleration is not compiled here. GPU contracts must verify the available kernel path and measure throughput; do not launch full extraction on an unprofiled fallback.'
fi
echo "Prepared environment: $JEVPFN_CONDA_ENV"
echo 'No inference or job submission performed. Actual GPU compatibility still requires the contract job.'
