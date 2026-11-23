#!/bin/bash -l
# Group C: build ONE compact export ZIP for one experiment (C1 representations, C2 adaptation).
# The default log folder is C1's; submit C2 with its own folder (command-line options win):
#   sbatch --output=output_JEVPFN/C/exp11_adaptation/logs/%x_%j.log
#          --error=output_JEVPFN/C/exp11_adaptation/logs/%x_%j.log scripts/slurm/group_c_export.slurm C2
#SBATCH --job-name=jevpfn-group-c-export
#SBATCH --clusters=mindwell
#SBATCH --account=lp_verbekelab
#SBATCH --partition=batch_graniterapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --time=12:00:00
#SBATCH --output="output_JEVPFN/C/exp10_representations/logs/%x_%j.log"
#SBATCH --error="output_JEVPFN/C/exp10_representations/logs/%x_%j.log"
set -euo pipefail
STUDY="${1:?Pass C1 or C2}"
case "$STUDY" in C1|C2) ;; *) echo "Unknown Group C study: $STUDY" >&2; exit 2 ;; esac
cd "${VSC_DATA:?}/JEVPFN"
source scripts/storage_environment.sh
export JEVPFN_STAGING_ROOT="${JEVPFN_DURABLE_ROOT:?}"
export JEVPFN_CONDA_ENV=JEVPFN PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source scripts/activate_conda.sh
"${JEVPFN_PYTHON:?}" -m src.evaluation.group_c --study "$STUDY"
