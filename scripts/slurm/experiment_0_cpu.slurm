#!/bin/bash -l
# Submit from $VSC_DATA/JEVPFN after creating "output_JEVPFN/3. experiment_0/logs".
#SBATCH --job-name=jevpfn-exp0-cpu
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=wice
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output="output_JEVPFN/3. experiment_0/logs/%x_%j.log"
#SBATCH --error="output_JEVPFN/3. experiment_0/logs/%x_%j.log"
set -euo pipefail
export JEVPFN_PHASE=experiment_0
export JEVPFN_CONDA_ENV=JEVPFN
export JEVPFN_STAGING_ROOT="${1:-/lustre1/project/stg_00211}"
cd "${VSC_DATA:?}/JEVPFN"
source scripts/slurm/_activate_env.sh
python -m src.jev.question_audit --phase experiment_0
python -m src.cluster.preflight --verify-data --tests
echo 'CPU preflight finished; this is not acceptance of the full experiment 0.'
