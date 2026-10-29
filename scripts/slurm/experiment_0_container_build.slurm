#!/bin/bash -l
#SBATCH --job-name=jevpfn-container-build
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=wice
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=13600M
#SBATCH --time=02:00:00
#SBATCH --output="output_JEVPFN/3. experiment_0/logs/%x_%j.log"
#SBATCH --error="output_JEVPFN/3. experiment_0/logs/%x_%j.log"
set -euo pipefail
export JEVPFN_STAGING_ROOT="${3:-${JEVPFN_DURABLE_ROOT:-${JEVPFN_PROJECT_ROOT:-${VSC_PROJECT_ROOT:-/lustre1/project/stg_00211}}}}"
export JEVPFN_RUNTIME_PROFILE="${4:-legacy-cu126}"
cd "${VSC_DATA:?}/JEVPFN"
bash scripts/build_decision_container.sh "${1:?Pass provider}" "${2:-wrap}"
