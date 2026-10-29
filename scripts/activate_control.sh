#!/bin/bash
# Invoked by activate_conda only after the CPU image is explicitly selected.
jevpfn_activate_control() {
    local image="$1"
    local active="${JEVPFN_DURABLE_ROOT:-${JEVPFN_PROJECT_ROOT:-${VSC_PROJECT_ROOT:-/lustre1/project/stg_00211}}}/JEVPFN/containers/control.active.json"
    local verification=()
    if [[ -f "$active" ]]; then verification=(--active "$active"); fi
    /usr/bin/python3 -I src/cluster/control_runtime.py verify "$image" "${verification[@]}" >/dev/null || return
    if declare -F deactivate >/dev/null; then deactivate || return; fi
    unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
    for control_variable in ${!CONDA_@} ${!_CONDA_@}; do unset "$control_variable"; done
    unset _CE_M _CE_CONDA
    unset -f conda __conda_activate __conda_exe __conda_reactivate __conda_hashr
    unalias python python3 pip pip3 2>/dev/null || true
    unset -f python python3 pip pip3 2>/dev/null || true
    export PYTHONNOUSERSITE=1 PATH="$image/bin:$PATH" JEVPFN_PYTHON="$image/bin/python"
    # Slurm clients and the nested image launcher belong to the host installation.
    # Bind only its tools/plugins/config; the image retains its own Python dependencies.
    local tool
    for tool in /usr/bin/sacct /usr/bin/squeue /usr/bin/scontrol /usr/bin/srun /usr/bin/sbatch /usr/bin/apptainer /usr/bin/singularity /usr/lib64/slurm /usr/libexec/apptainer /etc/slurm /etc/apptainer /usr/lib64/libmunge.so*; do
        if [[ -e "$tool" ]]; then
            case ",${APPTAINER_BINDPATH:-}," in
                *",$tool:$tool:ro,"*) ;;
                *) export APPTAINER_BINDPATH="${APPTAINER_BINDPATH:+$APPTAINER_BINDPATH,}$tool:$tool:ro" ;;
            esac
        fi
    done
    export JEVPFN_CONTROL_IMAGE="$image"
    hash -r
    "$JEVPFN_PYTHON" -I -c 'import sys; assert sys.version_info[:2]==(3,12); print("JEVPFN control container:",sys.executable)' || return
}
jevpfn_activate_control "$1"
