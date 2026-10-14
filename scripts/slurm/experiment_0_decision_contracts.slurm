#!/bin/bash -l
# Submit one provider at a time after exact token counts and checkpoint staging.
#SBATCH --job-name=jevpfn-exp0-contracts
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output="output_JEVPFN/3. experiment_0/logs/%x_%j.log"
#SBATCH --error="output_JEVPFN/3. experiment_0/logs/%x_%j.log"
set -euo pipefail
export JEVPFN_PHASE=experiment_0
export JEVPFN_PROVIDER="${JEVPFN_PROVIDER:?Choose cloudflare or intern (or another prepared open provider)}"
export JEVPFN_CONDA_ENV="${JEVPFN_CONDA_ENV:?Choose its isolated PROJECT environment}"
cd "${VSC_DATA:?}/JEVPFN"
source scripts/slurm/_activate_env.sh
python -m src.cluster.preflight --gpu --stage before_extraction
python -m src.cluster.gpu_contracts --provider "$JEVPFN_PROVIDER" --execute
