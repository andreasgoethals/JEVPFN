#!/bin/bash -l
#SBATCH --job-name=jevpfn-exp0-predictor
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output="output_JEVPFN/3. experiment_0/logs/%x_%j.log"
#SBATCH --error="output_JEVPFN/3. experiment_0/logs/%x_%j.log"
set -euo pipefail
export JEVPFN_PHASE=experiment_0
export JEVPFN_CONDA_ENV="${JEVPFN_CONDA_ENV:?Select the isolated predictor environment on PROJECT}"
cd "${VSC_DATA:?}/JEVPFN"
source scripts/slurm/_activate_env.sh
python -m src.cluster.predictor_contracts --model "${JEVPFN_MODEL:?Select one predictor}" --execute
