#!/bin/bash -l
#SBATCH --job-name=jevpfn-exp0-gpu
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=output_JEVPFN/experiment_0/logs/%x_%j.log
#SBATCH --error=output_JEVPFN/experiment_0/logs/%x_%j.log
set -euo pipefail
export JEVPFN_PHASE=experiment_0
cd "${VSC_DATA:?}/JEVPFN"
source scripts/slurm/_activate_env.sh
python -m src.cluster.preflight --gpu
echo 'GPU/import preflight finished; no model weights loaded and no modelling experiment run.'
