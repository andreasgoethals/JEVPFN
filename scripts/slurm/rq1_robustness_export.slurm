#!/bin/bash -l
#SBATCH --job-name=jevpfn-rq1-robustness-export
#SBATCH --clusters=mindwell
#SBATCH --account=lp_verbekelab
#SBATCH --partition=batch_graniterapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --time=02:00:00
#SBATCH --output="output_JEVPFN/A/exp02_robustness/logs/%x_%j.log"
#SBATCH --error="output_JEVPFN/A/exp02_robustness/logs/%x_%j.log"
set -euo pipefail
cd "${VSC_DATA:?}/JEVPFN"
source scripts/storage_environment.sh
export JEVPFN_STAGING_ROOT="${JEVPFN_DURABLE_ROOT:?}"
export JEVPFN_CONDA_ENV=JEVPFN PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source scripts/activate_conda.sh
"${JEVPFN_PYTHON:?}" -m src.evaluation.robustness_export "$@"
