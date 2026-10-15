#!/bin/bash -l
# Submit from $VSC_DATA/JEVPFN after src.cluster.setup and tokenizer staging.
#SBATCH --job-name=jevpfn-exp0-tokens
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=wice
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=13600M
#SBATCH --time=04:00:00
#SBATCH --output="output_JEVPFN/3. experiment_0/logs/%x_%j.log"
#SBATCH --error="output_JEVPFN/3. experiment_0/logs/%x_%j.log"
set -euo pipefail
export JEVPFN_PHASE=experiment_0
export JEVPFN_CONDA_ENV=JEVPFN
export JEVPFN_STAGING_ROOT="${1:-/lustre1/project/stg_00211}"
cd "${VSC_DATA:?}/JEVPFN"
source scripts/slurm/_activate_env.sh
python -m src.cluster.token_audit
