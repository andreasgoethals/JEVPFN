#!/bin/bash -l
# Full extraction requires explicit reviewed acceptance in self_hosted.yaml.
#SBATCH --job-name=jevpfn-features
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output="output_JEVPFN/2. feature_creation/logs/%x_%j.log"
#SBATCH --error="output_JEVPFN/2. feature_creation/logs/%x_%j.log"
set -euo pipefail
export JEVPFN_PHASE=feature_creation
export JEVPFN_PROVIDER="${JEVPFN_PROVIDER:?Select the accepted open-weight provider}"
export JEVPFN_CONDA_ENV="${JEVPFN_CONDA_ENV:?Select its isolated PROJECT environment}"
cd "${VSC_DATA:?}/JEVPFN"
source scripts/slurm/_activate_env.sh
python -m src.jev.self_hosted --provider "$JEVPFN_PROVIDER" --execute --full
