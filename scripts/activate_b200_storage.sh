#!/bin/bash
jevpfn_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]//\\//}")" && pwd)"
source "$jevpfn_script_dir/storage_environment.sh" || return 2 2>/dev/null || exit 2
# Host bootstrap; no downloads/copies occur while a GPU waits.
if [[ "${JEVPFN_STORAGE_MODE:-}" == tar_preparation ]]; then
    [[ "${SLURM_CLUSTER_NAME:-}" == mindwell && "${JEVPFN_CONDA_ENV:-}" == e5_tar && "${JEVPFN_DURABLE_ROOT:-}" == /gpfs*/project/* ]] || { echo 'TAR requires its sealed Mindwell runtime and native GPFS PROJECT.' >&2; return 2; }
    export JEVPFN_SHARED_IMMUTABLES=1 JEVPFN_RUNTIME_PROFILE=blackwell-cu130
    export JEVPFN_STAGING_ROOT="$JEVPFN_DURABLE_ROOT"
    # Fold-specific immutable bundles publish atomically; Trainer checkpoints use native scratch.
elif [[ "${SLURM_CLUSTER_NAME:-}" == mindwell && ( "${SLURM_JOB_PARTITION:-}" == gpu* || "${JEVPFN_STORAGE_MODE:-}" == scratch_mirror ) ]]; then
    export JEVPFN_STORAGE_MODE=scratch_mirror JEVPFN_RUNTIME_PROFILE=blackwell-cu130
    export JEVPFN_DURABLE_ROOT="${JEVPFN_DURABLE_ROOT:-${JEVPFN_PROJECT_ROOT:-${VSC_PROJECT_ROOT:-${JEVPFN_PROJECT_ROOT:?Select the allocated durable PROJECT root}}}}"
    if [[ "$JEVPFN_DURABLE_ROOT" == /gpfs*/project/* ]]; then
        export JEVPFN_SHARED_IMMUTABLES=1
    fi
    JEVPFN_SAVED_RUNTIME="${JEVPFN_CONDA_ENV:?}"
    export JEVPFN_SCRATCH_RUNTIME="${JEVPFN_SAVED_RUNTIME##*/}"
    export JEVPFN_CONDA_ENV=JEVPFN
    source scripts/activate_conda.sh || return
    JEVPFN_NATIVE_ROOT="$("${JEVPFN_PYTHON:?}" -m src.cluster.scratch_stage path --runtime "$JEVPFN_SCRATCH_RUNTIME")" || return
    export JEVPFN_STAGING_ROOT="$JEVPFN_NATIVE_ROOT"
    export JEVPFN_CONDA_ENV="$JEVPFN_SCRATCH_RUNTIME"
    [[ -f "$JEVPFN_STAGING_ROOT/JEVPFN/_JEVPFN_WORKSPACE.json" ]] || { echo 'Run the CPU GPFS staging job first.' >&2; return 2; }
fi

jevpfn_run() {
    if [[ "${JEVPFN_STORAGE_MODE:-}" == scratch_mirror ]]; then
        "${JEVPFN_MONITOR_PYTHON:?}" -m src.cluster.scratch_stage run --runtime "${JEVPFN_SCRATCH_RUNTIME:?}" -- "$@"
    else
        "$@"
    fi
}
