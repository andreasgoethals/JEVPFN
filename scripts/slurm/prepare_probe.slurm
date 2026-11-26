#!/bin/bash -l
# B2b: prepare cross-fitted label-trained classifier features for one dataset/encoder
# (both views, five outer folds) before the predictor jobs that read them.
#SBATCH --job-name=jevpfn-prepare-probe
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=mindwell
#SBATCH --partition=batch_graniterapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --time=24:00:00
#SBATCH --output="output_JEVPFN/B/B3/logs/%x_%j.log"
#SBATCH --error="output_JEVPFN/B/B3/logs/%x_%j.log"
set -euo pipefail
source "${VSC_DATA:?}/JEVPFN/scripts/storage_environment.sh"
export JEVPFN_PHASE=B3
export JEVPFN_CONDA_ENV=JEVPFN
export JEVPFN_STAGING_ROOT="${3:-${JEVPFN_DURABLE_ROOT:-${JEVPFN_PROJECT_ROOT:-${VSC_PROJECT_ROOT:-${JEVPFN_PROJECT_ROOT:?Select the allocated durable PROJECT root}}}}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
cd "${VSC_DATA:?}/JEVPFN"
source scripts/slurm/_activate_env.sh
python -m src.evaluation.label_probe --dataset "${1:?Pass dataset ID}" --encoder "${2:?Pass encoder ID}"
