#!/bin/bash
# Source from a login/setup shell or a Slurm job. This never installs packages.
# Keep interpreter selection identical before installation and during experiments.
jevpfn_activate_conda() {
    unset JEVPFN_PYTHON
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
