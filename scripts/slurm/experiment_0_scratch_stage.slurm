#!/bin/bash -l
# Transfers and hash checks happen before a B200 allocation starts.
#SBATCH --job-name=jevpfn-exp0-stage-gpfs
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=mindwell
#SBATCH --partition=batch_graniterapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output="output_JEVPFN/3. experiment_0/logs/%x_%j.log"
#SBATCH --error="output_JEVPFN/3. experiment_0/logs/%x_%j.log"
set -euo pipefail
source "${VSC_DATA:?}/JEVPFN/scripts/storage_environment.sh"
cd "${VSC_DATA:?}/JEVPFN"
export JEVPFN_STAGING_ROOT="${JEVPFN_DURABLE_ROOT:-${JEVPFN_PROJECT_ROOT:-${VSC_PROJECT_ROOT:-${JEVPFN_PROJECT_ROOT:?Select the allocated durable PROJECT root}}}}"
export JEVPFN_DURABLE_ROOT="$JEVPFN_STAGING_ROOT"
export JEVPFN_RUNTIME_PROFILE=blackwell-cu130
export JEVPFN_CONDA_ENV=JEVPFN
source scripts/activate_conda.sh
"${JEVPFN_PYTHON:?}" -m src.cluster.scratch_stage stage --runtime "${1:?Pass runtime}"
