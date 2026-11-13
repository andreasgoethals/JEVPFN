#!/bin/bash -l
#SBATCH --job-name=jevpfn-tar
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=mindwell
#SBATCH --partition=gpu_b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output="output_JEVPFN/6. experiment_3/logs/%x_%A_%a.log"
#SBATCH --error="output_JEVPFN/6. experiment_3/logs/%x_%A_%a.log"
set -euo pipefail
cd "${VSC_DATA:?}/JEVPFN"
source scripts/storage_environment.sh
export JEVPFN_PHASE=experiment_3
export JEVPFN_STORAGE_MODE=tar_preparation
export JEVPFN_CONDA_ENV=e5_tar
export JEVPFN_RUNTIME_PROFILE=blackwell-cu130
export JEVPFN_STAGING_ROOT="${JEVPFN_DURABLE_ROOT:?Select the allocated GPFS PROJECT root}"
source scripts/slurm/_activate_env.sh
export TMPDIR="$XDG_CACHE_HOME/trainer"
mkdir -p "$TMPDIR"
jevpfn_run "${JEVPFN_MONITOR_PYTHON:?}" -m src.cluster.monitor --host-step -- "${JEVPFN_PYTHON:?}" -m src.evaluation.tar --dataset "${1:?Dataset ID}" --fold "${SLURM_ARRAY_TASK_ID:?}" --execute
