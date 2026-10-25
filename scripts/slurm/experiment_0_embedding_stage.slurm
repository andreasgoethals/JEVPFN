#!/bin/bash -l
# Shared encoder assets are prepared on CPU, then included in the Clef working set.
#SBATCH --job-name=jevpfn-exp0-encoder-stage
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=wice
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=13600M
#SBATCH --time=01:00:00
#SBATCH --output="output_JEVPFN/3. experiment_0/logs/%x_%j.log"
#SBATCH --error="output_JEVPFN/3. experiment_0/logs/%x_%j.log"
set -euo pipefail
cd "${VSC_DATA:?}/JEVPFN"
export JEVPFN_STAGING_ROOT=/lustre1/project/stg_00211 JEVPFN_CONDA_ENV=JEVPFN
source scripts/activate_conda.sh
for encoder in e5_small_v2 e5_large_v2 qwen3_embedding_0_6b; do
    "${JEVPFN_PYTHON:?}" -m src.data.text_embeddings --stage --encoder "$encoder"
done
