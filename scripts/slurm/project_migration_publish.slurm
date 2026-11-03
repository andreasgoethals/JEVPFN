#!/bin/bash -l
# Publish all stopped native recovery stores to the verified GPFS authority.
#SBATCH --job-name=jevpfn-migration-publish
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=mindwell
#SBATCH --partition=batch_graniterapids
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output="output_JEVPFN/2. feature_creation/manifests/storage_migration/publish_%j.log"
#SBATCH --error="output_JEVPFN/2. feature_creation/manifests/storage_migration/publish_%j.log"
set -euo pipefail
cd "${VSC_DATA:?}/JEVPFN"
source scripts/storage_environment.sh
export JEVPFN_STAGING_ROOT="$JEVPFN_DURABLE_ROOT"
export JEVPFN_PHASE=feature_creation JEVPFN_CONDA_ENV=JEVPFN JEVPFN_RUNTIME_PROFILE=blackwell-cu130
unset JEVPFN_STORAGE_MODE JEVPFN_SCRATCH_RUNTIME
source scripts/slurm/_activate_env.sh
python -m src.cluster.storage_migration publish-workspaces
python -m src.cluster.storage_migration numeric-backup
