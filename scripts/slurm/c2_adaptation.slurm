#!/bin/bash -l
# Group C, C2: Strands learning from labels on ONE dataset x outer fold x trained cell
# (head_only | lora_only | lora_head): three cross-fitted fits, then the fold bundle.
# `sbatch c2_adaptation.slurm pilot` times one LoRA fit on the smallest dataset instead.
# Size --time from the pilot: python -m src.evaluation.strands_adaptation --size-from PILOT.json
# Requires rq3/adaptation training_enabled: true (and, for head_only, the C1 store).
#SBATCH --job-name=jevpfn-c2-adaptation
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=mindwell
#SBATCH --partition=gpu_b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output="output_JEVPFN/exp10_adaptation/logs/%x_%j.log"
#SBATCH --error="output_JEVPFN/exp10_adaptation/logs/%x_%j.log"
set -euo pipefail
source "${VSC_DATA:?}/JEVPFN/scripts/storage_environment.sh"
export JEVPFN_PHASE=rq3
export JEVPFN_PROVIDER=strands
export JEVPFN_STAGING_ROOT="${JEVPFN_DURABLE_ROOT:?Select the allocated GPFS PROJECT root}"
export JEVPFN_CONDA_ENV="${JEVPFN_STAGING_ROOT}/JEVPFN/environments/strands"
MODE="${1:?Usage: sbatch c2_adaptation.slurm pilot | DATASET FOLD CELL}"
cd "${VSC_DATA:?}/JEVPFN"
source scripts/slurm/_provider_memory.sh
source scripts/slurm/_activate_env.sh
if [[ "$MODE" == pilot ]]; then
    ARGS=(--pilot --execute)
else
    FOLD="${2:?Outer fold 0-4}"
    CELL="${3:?Cell head_only | lora_only | lora_head}"
    case "$CELL" in head_only|lora_only|lora_head) ;; *) echo "Unknown cell: $CELL" >&2; exit 2 ;; esac
    ARGS=(--dataset "$MODE" --fold "$FOLD" --cell "$CELL" --execute)
fi
jevpfn_run "${JEVPFN_MONITOR_PYTHON:?}" -m src.cluster.monitor -- "${JEVPFN_PYTHON:?}" -m src.evaluation.strands_adaptation "${ARGS[@]}"
