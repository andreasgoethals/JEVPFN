#!/bin/bash -l
#SBATCH --job-name=jevpfn-control-test
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=wice
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3000M
#SBATCH --time=00:20:00
#SBATCH --output="output_JEVPFN/3. experiment_0/logs/%x_%j.log"
#SBATCH --error="output_JEVPFN/3. experiment_0/logs/%x_%j.log"
set -euo pipefail
cd "${VSC_DATA:?}/JEVPFN"
export JEVPFN_CONTROL_TEST=1 JEVPFN_CONTROL_LEGACY=0 JEVPFN_CONDA_ENV=JEVPFN JEVPFN_PHASE=experiment_0
export JEVPFN_STAGING_ROOT="${JEVPFN_PROJECT_ROOT:-/lustre1/project/stg_00211}"
source scripts/slurm/_activate_env.sh
control_status=0
python -m src.cluster.control_runtime test || control_status=1
python -m src.cluster.storage_contracts --execute || control_status=1
python -m src.cluster.preflight --verify-data --tests || control_status=1
exit "$control_status"
