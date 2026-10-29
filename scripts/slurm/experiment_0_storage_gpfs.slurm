#!/bin/bash -l
# Validate the actual native B200 filesystem on two CPU nodes, without loading CUDA.
#SBATCH --job-name=jevpfn-exp0-storage-gpfs
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=mindwell
#SBATCH --partition=batch_graniterapids
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output="output_JEVPFN/3. experiment_0/logs/%x_%j.log"
#SBATCH --error="output_JEVPFN/3. experiment_0/logs/%x_%j.log"
set -euo pipefail
cd "${VSC_DATA:?}/JEVPFN"
export JEVPFN_PHASE=experiment_0
export JEVPFN_CONDA_ENV=JEVPFN
export JEVPFN_STORAGE_MODE=scratch_mirror
export JEVPFN_RUNTIME_PROFILE=blackwell-cu130
export JEVPFN_DURABLE_ROOT="${JEVPFN_DURABLE_ROOT:-${JEVPFN_PROJECT_ROOT:-${VSC_PROJECT_ROOT:-/lustre1/project/stg_00211}}}"
export JEVPFN_STORAGE_RUNTIME="${1:?Pass a staged runtime}"
source scripts/activate_conda.sh
JEVPFN_STAGING_ROOT="$("${JEVPFN_PYTHON:?}" -m src.cluster.scratch_stage path --runtime "$JEVPFN_STORAGE_RUNTIME")"
export JEVPFN_STAGING_ROOT
export SLURM_EXPORT_ENV=ALL
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
"$JEVPFN_PYTHON" -m src.cluster.freeze
"$JEVPFN_PYTHON" -m src.cluster.scratch_stage run --runtime "$JEVPFN_STORAGE_RUNTIME" -- "$JEVPFN_PYTHON" -m src.cluster.storage_contracts --execute
