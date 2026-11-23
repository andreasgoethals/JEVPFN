#!/bin/bash -l
# Group B: build ONE compact export ZIP for one experiment (B1, B2, B3, B4 or B5).
#SBATCH --job-name=jevpfn-group-b-export
#SBATCH --clusters=mindwell
#SBATCH --account=lp_verbekelab
#SBATCH --partition=batch_graniterapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=12:00:00
#SBATCH --output="output_JEVPFN/B/exp08_when_text_helps/logs/%x_%j.log"
#SBATCH --error="output_JEVPFN/B/exp08_when_text_helps/logs/%x_%j.log"
set -euo pipefail
STUDY="${1:?Pass B1, B2, B3, B4 or B5}"
case "$STUDY" in B1|B2|B3|B4|B5) ;; *) echo "Unknown Group B study: $STUDY" >&2; exit 2 ;; esac
cd "${VSC_DATA:?}/JEVPFN"
source scripts/storage_environment.sh
export JEVPFN_STAGING_ROOT="${JEVPFN_DURABLE_ROOT:?}"
export JEVPFN_CONDA_ENV=JEVPFN PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source scripts/activate_conda.sh
"${JEVPFN_PYTHON:?}" -m src.evaluation.benchmark_export --study "$STUDY"
