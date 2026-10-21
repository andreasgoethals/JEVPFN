#!/bin/bash
# Runs inside hpc-container-wrapper, before creation at the explicit image prefix.
# A host .condarc / CONDA_ENVS_PATH must not redirect packages to personal DATA.
[[ "${CW_INSTALLATION_PATH:?}" == /* && "$CW_INSTALLATION_PATH" != / ]] || {
    echo 'Expected an absolute container installation root.' >&2
    return 2
}
export CONDA_ENVS_PATH="$CW_INSTALLATION_PATH/miniforge/envs"
mkdir -p -- "$CONDA_ENVS_PATH"
printf 'Conda environments restricted to container image: %s\n' "$CONDA_ENVS_PATH"
