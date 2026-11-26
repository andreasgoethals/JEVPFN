#!/bin/bash -l
#SBATCH --job-name=jevpfn-A0-geometry
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=mindwell
#SBATCH --partition=batch_graniterapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --time=02:00:00
#SBATCH --output="output_JEVPFN/A/A0/logs/%x_%j.log"
#SBATCH --error="output_JEVPFN/A/A0/logs/%x_%j.log"
set -euo pipefail
cd "${VSC_DATA:?}/JEVPFN"
source scripts/storage_environment.sh
export JEVPFN_PHASE=A0 JEVPFN_CONDA_ENV=JEVPFN
export JEVPFN_STAGING_ROOT="${JEVPFN_DURABLE_ROOT:?}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
source scripts/activate_conda.sh
"$JEVPFN_PYTHON" -m src.cluster.freeze
"$JEVPFN_PYTHON" -m src.evaluation.geometry_study --pilot --cohort open_weight --dataset "${1:?Pass dataset ID}"
