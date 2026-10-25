#!/bin/bash -l
# Full extraction requires explicit reviewed acceptance in self_hosted.yaml.
#SBATCH --job-name=jevpfn-features
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=mindwell
#SBATCH --partition=gpu_b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=28000M
#SBATCH --time=24:00:00
#SBATCH --output="output_JEVPFN/2. feature_creation/logs/%x_%j.log"
#SBATCH --error="output_JEVPFN/2. feature_creation/logs/%x_%j.log"
set -euo pipefail
export JEVPFN_PHASE=feature_creation
# Arguments survive KU Leuven's clean batch environment; submitting-shell exports do not.
export JEVPFN_PROVIDER="${1:?Usage: sbatch script.slurm PROVIDER [ENV_PREFIX] [PROJECT_ROOT]}"
export JEVPFN_STAGING_ROOT="${3:-/lustre1/project/stg_00211}"
export JEVPFN_CONDA_ENV="${2:-$JEVPFN_STAGING_ROOT/JEVPFN/environments/$JEVPFN_PROVIDER}"
if [[ "$JEVPFN_PROVIDER" == perplexity && "${SLURM_MEM_PER_NODE:-0}" -lt 112000 ]]; then
    echo "Perplexity 27B has a larger initial host-memory profile: sbatch --mem=112000M script.slurm perplexity ..." >&2
    exit 2
fi
cd "${VSC_DATA:?}/JEVPFN"
source scripts/slurm/_activate_env.sh
jevpfn_run "${JEVPFN_MONITOR_PYTHON:?}" -m src.cluster.monitor -- "${JEVPFN_PYTHON:?}" -m src.jev.self_hosted --provider "$JEVPFN_PROVIDER" --execute --full
