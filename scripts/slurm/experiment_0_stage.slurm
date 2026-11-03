#!/bin/bash -l
# CPU download/fingerprinting job; never occupy a GPU for network transfers.
#SBATCH --job-name=jevpfn-exp0-stage
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=mindwell
#SBATCH --partition=batch_graniterapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output="output_JEVPFN/3. experiment_0/logs/%x_%j.log"
#SBATCH --error="output_JEVPFN/3. experiment_0/logs/%x_%j.log"
set -euo pipefail
source "${VSC_DATA:?}/JEVPFN/scripts/storage_environment.sh"
export JEVPFN_PHASE=experiment_0
export JEVPFN_CONDA_ENV=JEVPFN
export JEVPFN_STAGING_ROOT="${2:-${JEVPFN_DURABLE_ROOT:-${JEVPFN_PROJECT_ROOT:-${VSC_PROJECT_ROOT:-${JEVPFN_PROJECT_ROOT:?Select the allocated durable PROJECT root}}}}}"
cd "${VSC_DATA:?}/JEVPFN"
source scripts/slurm/_activate_env.sh
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
python -m src.jev.checkpoints --provider "${1:?Pass provider}"
