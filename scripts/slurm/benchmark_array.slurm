#!/bin/bash -l
# Use the validated command emitted by src.cluster.benchmark_jobs.
#SBATCH --job-name=jevpfn-benchmark
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=wice
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56000M
#SBATCH --time=24:00:00
set -euo pipefail
JEVPFN_MANIFEST="${1:?Pass manifest}"
JEVPFN_MANIFEST_HASH="${2:?Pass manifest hash}"
export JEVPFN_CONDA_ENV="${3:?Pass runtime}"
export JEVPFN_STAGING_ROOT="${4:?Pass native project allocation}"
export JEVPFN_PHASE="${5:?Pass output phase}"
cd "${VSC_DATA:?}/JEVPFN"
source scripts/slurm/_activate_env.sh
"${JEVPFN_MONITOR_PYTHON:?}" -m src.cluster.monitor -- "${JEVPFN_PYTHON:?}" -m src.cluster.benchmark_jobs --run-manifest "$JEVPFN_MANIFEST" --sha256 "$JEVPFN_MANIFEST_HASH" --index "${SLURM_ARRAY_TASK_ID:?}"
