#!/bin/bash -l
# One dataset, five outer-fold tasks; submit after its TAR training array succeeds.
#SBATCH --job-name=jevpfn-tar-pca
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=mindwell
#SBATCH --partition=batch_graniterapids
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output="output_JEVPFN/6. experiment_3/logs/%x_%A_%a.log"
#SBATCH --error="output_JEVPFN/6. experiment_3/logs/%x_%A_%a.log"
set -euo pipefail
source "${VSC_DATA:?}/JEVPFN/scripts/storage_environment.sh"
export JEVPFN_PHASE=experiment_3
export JEVPFN_CONDA_ENV=JEVPFN
export JEVPFN_STAGING_ROOT="${JEVPFN_DURABLE_ROOT:?Select allocated GPFS PROJECT}"
cd "${VSC_DATA:?}/JEVPFN"
source scripts/slurm/_activate_env.sh
"${JEVPFN_PYTHON:?}" -m src.evaluation.tar --dataset "${1:?Pass dataset ID}" --fold "${SLURM_ARRAY_TASK_ID:?}" --pca --execute
