#!/bin/bash -l
# Full frozen encoding is disabled until the shared embedding config is enabled.
# Weights must already be staged; reuse the compatible Clef container.
#SBATCH --job-name=jevpfn-embeddings
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=56000M
#SBATCH --time=24:00:00
#SBATCH --output="output_JEVPFN/2. feature_creation/logs/%x_%j.log"
#SBATCH --error="output_JEVPFN/2. feature_creation/logs/%x_%j.log"
set -euo pipefail
export JEVPFN_PHASE=feature_creation
JEVPFN_ENCODER="${1:?Pass e5_small_v2|e5_large_v2|qwen3_embedding_0_6b}"
export JEVPFN_CONDA_ENV="${2:-cloudflare}"
export JEVPFN_STAGING_ROOT="${3:-/lustre1/project/stg_00211}"
cd "${VSC_DATA:?}/JEVPFN"
source scripts/slurm/_activate_env.sh
JEVPFN_EMBEDDING_ARGS=(--encoder "$JEVPFN_ENCODER" --execute)
if [[ -n "${4:-}" ]]; then JEVPFN_EMBEDDING_ARGS+=(--dataset "$4"); fi
"${JEVPFN_MONITOR_PYTHON:?}" -m src.cluster.monitor -- "${JEVPFN_PYTHON:?}" -m src.data.text_embeddings "${JEVPFN_EMBEDDING_ARGS[@]}"
