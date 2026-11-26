#!/bin/bash -l
#SBATCH --job-name=jevpfn-A0-replay
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=mindwell
#SBATCH --partition=gpu_b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output="output_JEVPFN/A/A0/logs/%x_%j.log"
#SBATCH --error="output_JEVPFN/A/A0/logs/%x_%j.log"
set -euo pipefail
cd "${VSC_DATA:?}/JEVPFN"
source scripts/storage_environment.sh
export JEVPFN_RUNTIME_PROFILE=blackwell-cu130 JEVPFN_PHASE=A0 JEVPFN_STORAGE_MODE=a0_pilot
export JEVPFN_PROVIDER="${1:?Pass the qualified open-weight provider}"
export JEVPFN_CONDA_ENV="$JEVPFN_PROVIDER"
export JEVPFN_STAGING_ROOT="${JEVPFN_DURABLE_ROOT:?}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source scripts/slurm/_activate_env.sh
jevpfn_run "${JEVPFN_MONITOR_PYTHON:?}" -m src.cluster.monitor --host-step -- "${JEVPFN_PYTHON:?}" -m src.evaluation.robustness --provider "$JEVPFN_PROVIDER" --pilot-suite
