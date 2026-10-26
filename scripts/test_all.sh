#!/bin/bash -l
# One resumable launcher for all open-weight setup/testing branches; no extraction.
# Preview: bash scripts/test_all.sh
# Run:     bash scripts/test_all.sh --execute --watch
set -euo pipefail
cd "${VSC_DATA:?}/JEVPFN"
export JEVPFN_STAGING_ROOT=/lustre1/project/stg_00211
export JEVPFN_DURABLE_ROOT="$JEVPFN_STAGING_ROOT"
export JEVPFN_RUNTIME_PROFILE=blackwell-cu130
export JEVPFN_CONDA_ENV=JEVPFN
source scripts/activate_conda.sh
exec "${JEVPFN_PYTHON:?}" -u -m src.cluster.setup_extractors --all "$@"
