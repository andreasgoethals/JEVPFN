#!/bin/bash -l
#SBATCH --job-name=jevpfn-exp0-predictor
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=mindwell
#SBATCH --partition=gpu_b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=56000M
#SBATCH --time=04:00:00
#SBATCH --output="output_JEVPFN/3. experiment_0/logs/%x_%j.log"
#SBATCH --error="output_JEVPFN/3. experiment_0/logs/%x_%j.log"
set -euo pipefail
export JEVPFN_PHASE=experiment_0
export JEVPFN_MODEL="${1:?Usage: sbatch script.slurm MODEL ENV_PREFIX [PROJECT_ROOT]}"
export JEVPFN_CONDA_ENV="${2:?Pass the isolated predictor environment prefix}"
export JEVPFN_STAGING_ROOT="${3:-${JEVPFN_DURABLE_ROOT:-${JEVPFN_PROJECT_ROOT:-${VSC_PROJECT_ROOT:-/lustre1/project/stg_00211}}}}"
cd "${VSC_DATA:?}/JEVPFN"
source scripts/slurm/_activate_env.sh
jevpfn_run "${JEVPFN_MONITOR_PYTHON:?}" -m src.cluster.monitor -- "${JEVPFN_PYTHON:?}" -m src.cluster.predictor_contracts --model "$JEVPFN_MODEL" --execute
