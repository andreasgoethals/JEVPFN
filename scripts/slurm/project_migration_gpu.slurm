#!/bin/bash -l
# Amortize queueing: sequential bounded probes in one allocation, no extraction.
#SBATCH --job-name=jevpfn-migration-b200
#SBATCH --account=lp_verbekelab
#SBATCH --clusters=mindwell
#SBATCH --partition=gpu_b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output="output_JEVPFN/2. feature_creation/manifests/storage_migration/gpu_%j.log"
#SBATCH --error="output_JEVPFN/2. feature_creation/manifests/storage_migration/gpu_%j.log"
set -euo pipefail
requested="${1:-all}"
case "$requested" in
    all|liquid|liquid_lfm3|strands|perplexity|cloudflare|intern|h2o|sii|kev|decision3|nimble) ;;
    *) echo 'Unknown migration GPU provider' >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then echo 'Pass at most one provider' >&2; exit 2; fi
cd "${VSC_DATA:?}/JEVPFN"
source scripts/storage_environment.sh
retired_root="$(/usr/bin/python3 -c 'import json; from pathlib import Path; print(json.loads(Path("output_JEVPFN/2. feature_creation/manifests/storage_migration/copy_verified.json").read_text())["source"])')"
base_bindpath="${APPTAINER_BINDPATH:-}"
jevpfn_mask_retired_project() {
    mkdir -m 700 "$XDG_CACHE_HOME/retired_project"
    export APPTAINER_BINDPATH="${APPTAINER_BINDPATH:+$APPTAINER_BINDPATH,}$XDG_CACHE_HOME/retired_project:$retired_root:ro"
}
export JEVPFN_PHASE=experiment_0 JEVPFN_RUNTIME_PROFILE=blackwell-cu130
export VSC_SCRATCH_GPFS1="${VSC_SCRATCH_GPFS1:?}/jevpfn-migration-validation"
test -d "$VSC_SCRATCH_GPFS1/jevpfn-work"
for provider in liquid liquid_lfm3 strands perplexity cloudflare intern h2o sii kev decision3 nimble; do
    if [[ "$requested" != all && "$requested" != "$provider" ]]; then continue; fi
    export JEVPFN_CONDA_ENV="$provider"
    export JEVPFN_STAGING_ROOT="$JEVPFN_DURABLE_ROOT"
    export APPTAINER_BINDPATH="$base_bindpath"
    source scripts/slurm/_activate_env.sh
    jevpfn_mask_retired_project
    jevpfn_run "${JEVPFN_MONITOR_PYTHON:?}" -m src.cluster.monitor -- "${JEVPFN_PYTHON:?}" -m src.cluster.storage_migration gpu --provider "$provider"
    # _activate_env owns a per-allocation disposable cache. Retire it before
    # switching interpreters; final trap covers the last iteration/failures.
    /usr/bin/python3 -I src/cluster/runtime_cache.py cleanup "$XDG_CACHE_HOME"
done
if [[ "$requested" != all && "$requested" != cloudflare ]]; then exit 0; fi
export JEVPFN_CONDA_ENV=cloudflare JEVPFN_STAGING_ROOT="$JEVPFN_DURABLE_ROOT"
export APPTAINER_BINDPATH="$base_bindpath"
source scripts/slurm/_activate_env.sh
jevpfn_mask_retired_project
for encoder in e5_small_v2 e5_large_v2 qwen3_embedding_0_6b; do
    jevpfn_run "${JEVPFN_MONITOR_PYTHON:?}" -m src.cluster.monitor -- "${JEVPFN_PYTHON:?}" -m src.cluster.embedding_contracts --execute --encoder "$encoder"
done
