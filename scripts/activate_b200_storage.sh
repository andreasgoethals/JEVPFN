#!/bin/bash
# Host bootstrap; no downloads/copies occur while a GPU waits.
if [[ "${SLURM_CLUSTER_NAME:-}" == mindwell ]]; then
    export JEVPFN_STORAGE_MODE=scratch_mirror JEVPFN_RUNTIME_PROFILE=blackwell-cu130
    export JEVPFN_DURABLE_ROOT="${JEVPFN_DURABLE_ROOT:-${JEVPFN_PROJECT_ROOT:-${VSC_PROJECT_ROOT:-/lustre1/project/stg_00211}}}"
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
