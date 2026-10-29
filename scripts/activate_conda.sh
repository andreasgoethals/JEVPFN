#!/bin/bash
# Source from a login/setup shell or a Slurm job. This never installs packages.
# Keep interpreter selection identical before installation and during experiments.
jevpfn_activate_conda() {
    unset JEVPFN_PYTHON  # A rejected control image must not leave a previous interpreter selected.
    local control_root="${JEVPFN_DURABLE_ROOT:-${JEVPFN_PROJECT_ROOT:-${VSC_PROJECT_ROOT:-/lustre1/project/stg_00211}}}/JEVPFN/containers"
    if [[ "${JEVPFN_CONTROL_LEGACY:-0}" != 1 && ( "${JEVPFN_CONTROL_TEST:-0}" == 1 || -f "$control_root/control.active.json" ) && "${JEVPFN_CONDA_ENV:-JEVPFN}" == JEVPFN ]]; then
        local control_image="${JEVPFN_STAGING_ROOT:-${control_root%/JEVPFN/containers}}/JEVPFN/containers/control-v1"
        source scripts/activate_control.sh "$control_image"
        return $?
    fi
    if declare -F deactivate >/dev/null; then deactivate || return; fi
    unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
    export PYTHONNOUSERSITE=1
    source "${VSC_DATA:?}/miniconda3/etc/profile.d/conda.sh" || return
    conda activate "${JEVPFN_CONDA_ENV:-JEVPFN}" || return
    # Aliases/functions can shadow even a correctly ordered PATH in interactive shells.
    unalias python python3 pip pip3 2>/dev/null || true
    unset -f python python3 pip pip3 2>/dev/null || true
    export PATH="${CONDA_PREFIX:?}/bin:$PATH"
    hash -r
    # DATA can have both /data and /vsc-hard-mounts spellings; compare real locations.
    "$CONDA_PREFIX/bin/python" -I -c 'import os,sys; from pathlib import Path; assert Path(sys.prefix).resolve() == Path(os.environ["CONDA_PREFIX"]).resolve(), "Wrong Conda interpreter"; assert sys.version_info[:2] == (3,12), "JEVPFN requires Python 3.12"; print("JEVPFN Python:", sys.executable)' || return
    export JEVPFN_PYTHON="$CONDA_PREFIX/bin/python"
}

jevpfn_activate_conda
