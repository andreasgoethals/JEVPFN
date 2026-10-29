#!/bin/bash
# Activate the verified runtime and enforce allocation, storage and frozen-source checks.
set -euo pipefail
case "${SLURM_JOB_PARTITION:-}" in
    gpu|gpu_a100*)
        echo 'A100 allocations are disabled for JEVPFN. Use Mindwell gpu_b200.' >&2
        exit 2
        ;;
esac
source scripts/activate_b200_storage.sh
source scripts/activate_runtime.sh
python -c 'from src.cluster.scheduling import reject_a100_allocation; reject_a100_allocation()'
python -m src.cluster.freeze
# Fail before any cache creation if PROJECT is missing or uses the other cluster's filesystem.
python -c 'import os; from src.cluster.preflight import storage_policy_failures; failures = storage_policy_failures(os.environ.get("SLURM_CLUSTER_NAME")); assert not failures, "\n".join(failures)'
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export BLIS_NUM_THREADS="$OMP_NUM_THREADS"
export OMP_MAX_ACTIVE_LEVELS=1
# Rebuildable compiler/plot caches use the allocation's native scratch, not a
# cross-mounted generic VSC_SCRATCH or persistent PROJECT fallback.
XDG_CACHE_HOME="$(/usr/bin/python3 -I src/cluster/runtime_cache.py create)"
export XDG_CACHE_HOME
jevpfn_cleanup_runtime_cache() {
    local result=$?
    trap - EXIT
    if ! /usr/bin/python3 -I "$VSC_DATA/JEVPFN/src/cluster/runtime_cache.py" cleanup "$XDG_CACHE_HOME"; then
        if [[ "$result" == 0 ]]; then result=1; fi
    fi
    exit "$result"
}
trap jevpfn_cleanup_runtime_cache EXIT
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
