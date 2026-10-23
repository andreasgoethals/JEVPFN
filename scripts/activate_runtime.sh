#!/bin/bash
# Select a sealed compact runtime when available; retain native Conda for CPU/custom envs.
JEVPFN_REQUESTED_ENV="${JEVPFN_CONDA_ENV:-JEVPFN}"
JEVPFN_RUNTIME_NAME="${JEVPFN_REQUESTED_ENV##*/}"
JEVPFN_RUNTIME_BASE="${JEVPFN_STAGING_ROOT:-/lustre1/project/stg_00211}/JEVPFN"
JEVPFN_PACKED_ENV="$JEVPFN_RUNTIME_BASE/containers/$JEVPFN_RUNTIME_NAME"
unset JEVPFN_CONTAINER_PROVIDER JEVPFN_CONTAINER_ID JEVPFN_MONITOR_PYTHON JEVPFN_NVIDIA_SMI
case "$JEVPFN_RUNTIME_NAME" in
    cloudflare|intern|liquid|strands|perplexity|h2o|sii|kev|decision3|nimble|catboost|tabpfn_3_5|tabicl_v2|limix_2|kumo_tabular|linear|tabstar|contexttab|ag_mm)
        if [[ "$JEVPFN_REQUESTED_ENV" == "$JEVPFN_RUNTIME_NAME" ]]; then
            JEVPFN_REQUESTED_ENV="$JEVPFN_RUNTIME_BASE/environments/$JEVPFN_RUNTIME_NAME"
            export JEVPFN_CONDA_ENV="$JEVPFN_REQUESTED_ENV"
        fi
        if [[ ( "$JEVPFN_REQUESTED_ENV" == "$JEVPFN_RUNTIME_BASE/environments/$JEVPFN_RUNTIME_NAME" || "$JEVPFN_REQUESTED_ENV" == "$JEVPFN_PACKED_ENV" ) && -f "$JEVPFN_PACKED_ENV/_JEVPFN_CONTAINER.json" ]]; then
            export JEVPFN_CONDA_ENV=JEVPFN
            source scripts/activate_conda.sh || return
            JEVPFN_RUNTIME_EXPORTS="$("${JEVPFN_PYTHON:?}" -m src.cluster.containers verify "$JEVPFN_RUNTIME_NAME" --shell)" || return
            eval "$JEVPFN_RUNTIME_EXPORTS"
            # Monitor on the host: NVML/global Slurm GPU IDs need not share the image's view.
            export JEVPFN_MONITOR_PYTHON="$JEVPFN_PYTHON"
            export JEVPFN_NVIDIA_SMI="$(type -P nvidia-smi || true)"
            export JEVPFN_CONDA_ENV="$JEVPFN_REQUESTED_ENV"
            # Clear the WHOLE host activation stack before the image initializes Conda.
            # Leaving CONDA_SHLVL=2 without CONDA_PREFIX makes its activator dereference None.
            for JEVPFN_HOST_CONDA_VARIABLE in ${!CONDA_@} ${!_CONDA_@}; do
                unset "$JEVPFN_HOST_CONDA_VARIABLE"
            done
            unset _CE_M _CE_CONDA
            unset -f conda __conda_activate __conda_exe __conda_reactivate __conda_hashr
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
source scripts/activate_conda.sh || return
export JEVPFN_MONITOR_PYTHON="$JEVPFN_PYTHON"
export JEVPFN_NVIDIA_SMI="$(type -P nvidia-smi || true)"
