#!/bin/bash -l
# Independent B200 contracts follow exact token counts and checkpoint staging.
#SBATCH --job-name=jevpfn-exp0-contracts
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=mindwell
#SBATCH --partition=gpu_b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output="output_JEVPFN/3. experiment_0/logs/%x_%j.log"
#SBATCH --error="output_JEVPFN/3. experiment_0/logs/%x_%j.log"
set -euo pipefail
source "${VSC_DATA:?}/JEVPFN/scripts/storage_environment.sh"
export JEVPFN_PHASE=experiment_0
# Arguments survive KU Leuven's clean batch environment; submitting-shell exports do not.
export JEVPFN_PROVIDER="${1:?Usage: sbatch script.slurm PROVIDER [ENV_PREFIX] [PROJECT_ROOT]}"
export JEVPFN_STAGING_ROOT="${3:-${JEVPFN_DURABLE_ROOT:-${JEVPFN_PROJECT_ROOT:-${VSC_PROJECT_ROOT:-${JEVPFN_PROJECT_ROOT:?Select the allocated durable PROJECT root}}}}}"
export JEVPFN_CONDA_ENV="${2:-$JEVPFN_STAGING_ROOT/JEVPFN/environments/$JEVPFN_PROVIDER}"
cd "${VSC_DATA:?}/JEVPFN"
source scripts/slurm/_provider_memory.sh
source scripts/slurm/_activate_env.sh
python -m src.cluster.preflight --gpu --stage before_extraction
jevpfn_run "${JEVPFN_MONITOR_PYTHON:?}" -m src.cluster.monitor -- "${JEVPFN_PYTHON:?}" -m src.cluster.gpu_contracts --provider "$JEVPFN_PROVIDER" --execute
