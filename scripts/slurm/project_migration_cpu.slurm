#!/bin/bash -l
# Bounded native-storage cutover QA, no model inference or API calls.
#SBATCH --job-name=jevpfn-migration-cpu
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=mindwell
#SBATCH --partition=batch_graniterapids
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output="output_JEVPFN/2. feature_creation/manifests/storage_migration/cpu_%j.log"
#SBATCH --error="output_JEVPFN/2. feature_creation/manifests/storage_migration/cpu_%j.log"
set -euo pipefail
cd "${VSC_DATA:?}/JEVPFN"
source scripts/storage_environment.sh
export JEVPFN_STAGING_ROOT="$JEVPFN_DURABLE_ROOT"
export JEVPFN_PHASE=experiment_0 JEVPFN_CONDA_ENV=JEVPFN JEVPFN_RUNTIME_PROFILE=blackwell-cu130
source scripts/slurm/_activate_env.sh
python -m src.cluster.control_runtime test
python -m src.cluster.storage_contracts --execute
python -m src.cluster.preflight --verify-data
python -m pytest -q tests/test_project_migration.py tests/test_storage_migration.py tests/test_file_lock.py tests/test_cache_merge.py tests/test_scratch_stage.py tests/test_scratch_capacity.py tests/test_feature_admission.py tests/test_freeze.py tests/test_feature_acceptance.py
python -m src.cluster.storage_migration cpu-accept
