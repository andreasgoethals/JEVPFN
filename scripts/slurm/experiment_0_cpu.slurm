#!/bin/bash -l
# Submit from $VSC_DATA/JEVPFN after creating output_JEVPFN/experiment_0/logs.
#SBATCH --job-name=jevpfn-exp0-cpu
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=wice
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=output_JEVPFN/experiment_0/logs/%x_%j.log
#SBATCH --error=output_JEVPFN/experiment_0/logs/%x_%j.log
set -euo pipefail
export JEVPFN_PHASE=experiment_0
cd "${VSC_DATA:?}/JEVPFN"
source scripts/slurm/_activate_env.sh
python -m src.cluster.preflight
python -m src.data.prepare --offline
python -m pytest -q
echo 'CPU preflight finished; this is not acceptance of the full experiment 0.'
