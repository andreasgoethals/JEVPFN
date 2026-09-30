#!/bin/bash
# Follow CreditPFN's Conda activation and interpreter verification, with a separate env.
set -euo pipefail
if declare -F deactivate >/dev/null; then deactivate; fi
# An inherited local venv variable must not describe the selected Conda interpreter.
unset VIRTUAL_ENV
source "${VSC_DATA:?}/miniconda3/etc/profile.d/conda.sh"
conda activate "${JEVPFN_CONDA_ENV:-JEVPFN}"
export PATH="${CONDA_PREFIX:?}/bin:$PATH"
hash -r
python -c 'import os,sys; from pathlib import Path; assert Path(sys.prefix) == Path(os.environ["CONDA_PREFIX"]); assert sys.version_info[:2] == (3,12)'
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export XDG_CACHE_HOME="${VSC_SCRATCH_NODE:-${VSC_DATA:?}}/jevpfn-runtime-${SLURM_JOB_ID:-setup}"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
# Future TabPFN weights belong on durable project storage, not node scratch.
export TABPFN_MODEL_CACHE_DIR="$(python -c 'from src.utils.paths import checkpoints_dir; print(checkpoints_dir("tabpfn"))')"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/nv"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
mkdir -p "$XDG_CACHE_HOME"
