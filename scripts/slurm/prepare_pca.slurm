#!/bin/bash -l
# Submit independently for each dataset/encoder, before their GPU benchmark jobs.
#SBATCH --job-name=jevpfn-prepare-pca
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=wice
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56000M
#SBATCH --time=24:00:00
#SBATCH --output="output_JEVPFN/5. experiment_2/logs/%x_%j.log"
#SBATCH --error="output_JEVPFN/5. experiment_2/logs/%x_%j.log"
set -euo pipefail
export JEVPFN_PHASE=experiment_2
export JEVPFN_CONDA_ENV=JEVPFN
export JEVPFN_STAGING_ROOT="${3:-/lustre1/project/stg_00211}"
cd "${VSC_DATA:?}/JEVPFN"
source scripts/slurm/_activate_env.sh
python -m src.evaluation.prepare_pca --dataset "${1:?Pass dataset ID}" --encoder "${2:?Pass encoder ID}"
