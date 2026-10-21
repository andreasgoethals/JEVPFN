#!/bin/bash -l
# Stage one model on a connected host first. Submit only one debug GPU job at a time.
# Pass PROVIDER and its optional environment prefix as script arguments.
#SBATCH --job-name=jevpfn-exp0-decision
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=28000M
#SBATCH --time=01:00:00
#SBATCH --output="output_JEVPFN/3. experiment_0/logs/%x_%j.log"
#SBATCH --error="output_JEVPFN/3. experiment_0/logs/%x_%j.log"
set -euo pipefail
export JEVPFN_PHASE=experiment_0
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
python -m src.cluster.preflight --gpu --stage before_extraction
python -m src.cluster.monitor -- python -m src.jev.self_hosted --provider "$JEVPFN_PROVIDER" --execute
