#!/bin/bash -l
# Exercise the replacement CPU controller and its staged model image on a real B200.
# This is migration evidence; full provider acceptance still requires frozen 0A.
#SBATCH --job-name=jevpfn-control-b200
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=mindwell
#SBATCH --partition=gpu_b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=28000M
#SBATCH --time=00:10:00
#SBATCH --output="output_JEVPFN/3. experiment_0/logs/%x_%j.log"
#SBATCH --error="output_JEVPFN/3. experiment_0/logs/%x_%j.log"
set -euo pipefail
cd "${VSC_DATA:?}/JEVPFN"
export JEVPFN_PHASE=experiment_0 JEVPFN_CONDA_ENV=cloudflare
export JEVPFN_DURABLE_ROOT="${JEVPFN_PROJECT_ROOT:-/lustre1/project/stg_00211}"
source scripts/slurm/_activate_env.sh
"${JEVPFN_MONITOR_PYTHON:?}" -I -c 'import os; from src.cluster.control_runtime import verify; print("NATIVE_CONTROL_VERIFIED", verify(os.environ["JEVPFN_CONTROL_IMAGE"])["container_id"])'
jevpfn_run "${JEVPFN_MONITOR_PYTHON:?}" -m src.cluster.monitor -- "${JEVPFN_PYTHON:?}" -m src.cluster.control_runtime gpu-test
