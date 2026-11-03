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
export JEVPFN_MIGRATION_QA_CONTEXT="$(python -c 'import json,os; from src.utils import paths; print(json.dumps(dict(cluster=os.environ["SLURM_CLUSTER_NAME"],job_id=os.environ["SLURM_JOB_ID"],gpfs_scratch=os.environ["VSC_SCRATCH_GPFS1"],node_scratch=os.environ["VSC_SCRATCH_NODE"],retained_root=str(paths.results_dir("storage_migration","publication_qa",phase="feature_creation")))))')"
python -m pytest -q tests/test_closed_publication.py tests/test_scratch_stage.py tests/test_scratch_capacity.py tests/test_storage_migration.py
unset JEVPFN_MIGRATION_QA_CONTEXT
python -m src.cluster.storage_migration publish-workspaces
python -m src.cluster.storage_migration numeric-backup
