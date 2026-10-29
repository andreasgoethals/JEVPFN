#!/bin/bash -l
#SBATCH --job-name=jevpfn-input-install
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=wice
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6800M
#SBATCH --time=00:30:00
#SBATCH --output="output_JEVPFN/3. experiment_0/logs/%x_%j.log"
#SBATCH --error="output_JEVPFN/3. experiment_0/logs/%x_%j.log"
set -euo pipefail
cd "${VSC_DATA:?}/JEVPFN"
export JEVPFN_CONDA_ENV=JEVPFN
export JEVPFN_STAGING_ROOT="${JEVPFN_PROJECT_ROOT:-/lustre1/project/stg_00211}"
source scripts/activate_conda.sh
"$JEVPFN_PYTHON" -m src.cluster.input_transfer install "$JEVPFN_STAGING_ROOT/JEVPFN/uploads/decision_inputs_v3.zip" --writers-stopped
