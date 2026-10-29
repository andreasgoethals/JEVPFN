#!/bin/bash -l
#SBATCH --job-name=jevpfn-exp0-gpu
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=mindwell
#SBATCH --partition=gpu_b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=28000M
#SBATCH --time=00:10:00
#SBATCH --output="output_JEVPFN/3. experiment_0/logs/%x_%j.log"
#SBATCH --error="output_JEVPFN/3. experiment_0/logs/%x_%j.log"
set -euo pipefail
export JEVPFN_PHASE=experiment_0
export JEVPFN_CONDA_ENV="${1:?Usage: sbatch script.slurm ENV_PREFIX [PROJECT_ROOT]}"
export JEVPFN_STAGING_ROOT="${2:-${JEVPFN_DURABLE_ROOT:-${JEVPFN_PROJECT_ROOT:-${VSC_PROJECT_ROOT:-/lustre1/project/stg_00211}}}}"
cd "${VSC_DATA:?}/JEVPFN"
source scripts/slurm/_activate_env.sh
jevpfn_run python -m src.cluster.preflight --gpu
echo 'GPU/import preflight finished; no model weights loaded and no modelling experiment run.'
