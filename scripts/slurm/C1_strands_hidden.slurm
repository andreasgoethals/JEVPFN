#!/bin/bash -l
# Group C, C1: Strands hidden states (3 positions x layers 12/24, float16) for ONE dataset.
# Exactly the frozen feature-creation requests; resumable; publishes manifest.json when complete.
# Requires representations.extraction.enabled: true in config/C/representations.yaml.
#SBATCH --job-name=jevpfn-c1-hidden
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=mindwell
#SBATCH --partition=gpu_b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output="output_JEVPFN/C/C1/logs/%x_%j.log"
#SBATCH --error="output_JEVPFN/C/C1/logs/%x_%j.log"
set -euo pipefail
source "${VSC_DATA:?}/JEVPFN/scripts/storage_environment.sh"
export JEVPFN_PHASE=representations
export JEVPFN_PROVIDER=strands
# Arguments survive KU Leuven's clean batch environment; submitting-shell exports do not.
DATASET="${1:?Usage: sbatch C1_strands_hidden.slurm DATASET [ENV_PREFIX] [PROJECT_ROOT] [LIMIT]}"
export JEVPFN_STAGING_ROOT="${3:-${JEVPFN_DURABLE_ROOT:?Select the allocated GPFS PROJECT root}}"
export JEVPFN_CONDA_ENV="${2:-$JEVPFN_STAGING_ROOT/JEVPFN/environments/strands}"
cd "${VSC_DATA:?}/JEVPFN"
source scripts/slurm/_provider_memory.sh
source scripts/slurm/_activate_env.sh
ARGS=(--dataset "$DATASET" --execute)
if [[ -n "${4:-}" ]]; then
    ARGS+=(--limit "$4")  # pilot: bounded new inputs, no manifest
fi
jevpfn_run "${JEVPFN_MONITOR_PYTHON:?}" -m src.cluster.monitor -- "${JEVPFN_PYTHON:?}" -m src.jev.strands_hidden "${ARGS[@]}"
