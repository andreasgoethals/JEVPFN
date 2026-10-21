#!/bin/bash -l
# Reuse the compatible Clef environment without installing or changing any package.
#SBATCH --job-name=jevpfn-exp0-e5
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=28000M
#SBATCH --time=00:15:00
#SBATCH --output="output_JEVPFN/3. experiment_0/logs/%x_%j.log"
#SBATCH --error="output_JEVPFN/3. experiment_0/logs/%x_%j.log"
set -euo pipefail
export JEVPFN_PHASE=experiment_0
export JEVPFN_STAGING_ROOT="${2:-/lustre1/project/stg_00211}"
export JEVPFN_CONDA_ENV="${1:-$JEVPFN_STAGING_ROOT/JEVPFN/environments/cloudflare}"
cd "${VSC_DATA:?}/JEVPFN"
source scripts/slurm/_activate_env.sh
python -m src.cluster.monitor -- python -m src.cluster.embedding_contracts --execute --encoder "${3:-e5_small_v2}"
