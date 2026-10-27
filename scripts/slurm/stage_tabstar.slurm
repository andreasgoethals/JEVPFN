#!/bin/bash -l
#SBATCH --job-name=jevpfn-stage-tabstar
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=wice
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6800M
#SBATCH --time=02:00:00
#SBATCH --output="output_JEVPFN/3. experiment_0/logs/%x_%j.log"
#SBATCH --error="output_JEVPFN/3. experiment_0/logs/%x_%j.log"
set -euo pipefail
cd "${VSC_DATA:?}/JEVPFN"
export JEVPFN_STAGING_ROOT="${1:-/lustre1/project/stg_00211}"
export JEVPFN_CONDA_ENV=JEVPFN
source scripts/activate_conda.sh
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
"${JEVPFN_PYTHON:?}" -m src.cluster.predictor_assets --execute
