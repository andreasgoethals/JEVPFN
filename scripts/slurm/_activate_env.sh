#!/bin/bash
# Follow CreditPFN's Conda activation and interpreter verification, with a separate env.
set -euo pipefail
source scripts/activate_conda.sh
# Fail before any cache creation if PROJECT is missing or uses the other cluster's filesystem.
python -c 'import os; from src.cluster.preflight import storage_policy_failures; failures = storage_policy_failures(os.environ.get("SLURM_CLUSTER_NAME")); assert not failures, "\n".join(failures)'
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
# Compiler/runtime caches can also be large. Prefer disposable scratch, then PROJECT;
# never use the small personal DATA allocation as a fallback for these caches.
JEVPFN_RUNTIME_ROOT="${VSC_SCRATCH_NODE:-${VSC_SCRATCH:-}}"
if [[ -z "$JEVPFN_RUNTIME_ROOT" ]]; then
    JEVPFN_RUNTIME_ROOT="$(python -c 'from src.utils.paths import large_outputs_dir; print(large_outputs_dir() / "runtime_cache")')"
fi
export XDG_CACHE_HOME="$JEVPFN_RUNTIME_ROOT/jevpfn-runtime-${SLURM_JOB_ID:-setup}"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
# Every model's durable weights belong on project storage, including HF/Torch downloads.
export TORCH_HOME="$(python -c 'from src.utils.paths import checkpoints_dir; print(checkpoints_dir("torch"))')"
export HF_HOME="$(python -c 'from src.utils.paths import checkpoints_dir; print(checkpoints_dir("huggingface"))')"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export TABPFN_MODEL_CACHE_DIR="$(python -c 'from src.utils.paths import checkpoints_dir; print(checkpoints_dir("tabpfn"))')"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/nv"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SLURM_EXPORT_ENV=ALL  # Propagate the prepared job environment to srun workers.
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$XDG_CACHE_HOME"
