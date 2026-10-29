#!/bin/bash
# Invoked by activate_conda only after the CPU image is explicitly selected.
jevpfn_activate_control() {
    local image="$1"
    /usr/bin/python3 -I scripts/../src/cluster/control_runtime.py verify "$image" >/dev/null || return
    if declare -F deactivate >/dev/null; then deactivate || return; fi
    unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
    for control_variable in ${!CONDA_@} ${!_CONDA_@}; do unset "$control_variable"; done
    unset _CE_M _CE_CONDA
    unset -f conda __conda_activate __conda_exe __conda_reactivate __conda_hashr
    unalias python python3 pip pip3 2>/dev/null || true
    unset -f python python3 pip pip3 2>/dev/null || true
    export PYTHONNOUSERSITE=1 PATH="$image/bin:$PATH" JEVPFN_PYTHON="$image/bin/python"
    export JEVPFN_CONTROL_IMAGE="$image"
    hash -r
    "$JEVPFN_PYTHON" -I -c 'import sys; assert sys.version_info[:2]==(3,12); print("JEVPFN control container:",sys.executable)' || return
}
jevpfn_activate_control "$1"
