#!/bin/bash
# Select a sealed compact runtime when available; retain native Conda for CPU/custom envs.
JEVPFN_REQUESTED_ENV="${JEVPFN_CONDA_ENV:-JEVPFN}"
JEVPFN_RUNTIME_NAME="${JEVPFN_REQUESTED_ENV##*/}"
JEVPFN_RUNTIME_BASE="${JEVPFN_STAGING_ROOT:-/lustre1/project/stg_00211}/JEVPFN"
JEVPFN_PACKED_ENV="$JEVPFN_RUNTIME_BASE/containers/$JEVPFN_RUNTIME_NAME"
unset JEVPFN_CONTAINER_PROVIDER JEVPFN_CONTAINER_ID
case "$JEVPFN_RUNTIME_NAME" in
    cloudflare|intern|liquid|strands|perplexity|catboost|tabpfn_3_5|tabicl_v2|limix_2|kumo_tabular|linear|tabstar|contexttab)
        if [[ ( "$JEVPFN_REQUESTED_ENV" == "$JEVPFN_RUNTIME_BASE/environments/$JEVPFN_RUNTIME_NAME" || "$JEVPFN_REQUESTED_ENV" == "$JEVPFN_PACKED_ENV" ) && -f "$JEVPFN_PACKED_ENV/_JEVPFN_CONTAINER.json" ]]; then
            export JEVPFN_CONDA_ENV=JEVPFN
            source scripts/activate_conda.sh || return
            JEVPFN_RUNTIME_EXPORTS="$("${JEVPFN_PYTHON:?}" -m src.cluster.containers verify "$JEVPFN_RUNTIME_NAME" --shell)" || return
            eval "$JEVPFN_RUNTIME_EXPORTS"
            export JEVPFN_CONDA_ENV="$JEVPFN_REQUESTED_ENV"
            unset CONDA_PREFIX CONDA_DEFAULT_ENV
            export PATH="$JEVPFN_PACKED_ENV/bin:$PATH"
            export JEVPFN_PYTHON="$JEVPFN_PACKED_ENV/bin/python"
            if [[ "${SLURM_GPUS_ON_NODE:-0}" != 0 || "${SLURM_JOB_PARTITION:-}" == gpu* ]]; then
                export APPTAINER_NV=1 SINGULARITY_NV=1
            fi
            hash -r
            "$JEVPFN_PYTHON" -I -c 'import sys; assert sys.version_info[:2] == (3,12); print("JEVPFN packed Python:", sys.executable)' || return
            return
        fi
        ;;
esac
source scripts/activate_conda.sh
