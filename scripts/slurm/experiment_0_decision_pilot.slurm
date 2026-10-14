#!/bin/bash -l
# Stage one model on a connected host first. Submit only one debug GPU job at a time.
# Set JEVPFN_PROVIDER and JEVPFN_CONDA_ENV to select the prepared isolated runtime.
#SBATCH --job-name=jevpfn-exp0-decision
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
export JEVPFN_PROVIDER="${JEVPFN_PROVIDER:-liquid}"
export JEVPFN_CONDA_ENV="${JEVPFN_CONDA_ENV:?Select the provider environment on PROJECT first}"
cd "${VSC_DATA:?}/JEVPFN"
source scripts/slurm/_activate_env.sh
python -m src.cluster.preflight --gpu --stage before_extraction
python -m src.jev.self_hosted --provider "$JEVPFN_PROVIDER" --execute
