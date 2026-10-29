#!/bin/bash -l
#SBATCH --job-name=jevpfn-control-build
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
cd "${VSC_DATA:?}/JEVPFN"
bash scripts/build_control_container.sh
