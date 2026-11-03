#!/bin/bash -l
# Submit independently for each dataset/encoder, before their GPU benchmark jobs.
#SBATCH --job-name=jevpfn-prepare-pca
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=mindwell
#SBATCH --partition=batch_graniterapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output="output_JEVPFN/5. experiment_2/logs/%x_%j.log"
#SBATCH --error="output_JEVPFN/5. experiment_2/logs/%x_%j.log"
set -euo pipefail
source "${VSC_DATA:?}/JEVPFN/scripts/storage_environment.sh"
export JEVPFN_PHASE=experiment_2
export JEVPFN_CONDA_ENV=JEVPFN
export JEVPFN_STAGING_ROOT="${3:-${JEVPFN_DURABLE_ROOT:-${JEVPFN_PROJECT_ROOT:-${VSC_PROJECT_ROOT:-${JEVPFN_PROJECT_ROOT:?Select the allocated durable PROJECT root}}}}}"
cd "${VSC_DATA:?}/JEVPFN"
source scripts/slurm/_activate_env.sh
python -m src.evaluation.prepare_pca --dataset "${1:?Pass dataset ID}" --encoder "${2:?Pass encoder ID}"
