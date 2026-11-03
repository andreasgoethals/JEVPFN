#!/bin/bash -l
jevpfn_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]//\\//}")" && pwd)"
source "$jevpfn_script_dir/storage_environment.sh" || return 2 2>/dev/null || exit 2
# One resumable launcher for all open-weight setup/testing branches; no extraction.
# Preview: bash scripts/test_all.sh
# Run:     bash scripts/test_all.sh --execute --watch
set -euo pipefail
cd "${VSC_DATA:?}/JEVPFN"
export JEVPFN_STAGING_ROOT="${JEVPFN_STAGING_ROOT:-${JEVPFN_PROJECT_ROOT:-${VSC_PROJECT_ROOT:-${JEVPFN_PROJECT_ROOT:?Select the allocated durable PROJECT root}}}}"
export JEVPFN_DURABLE_ROOT="$JEVPFN_STAGING_ROOT"
export JEVPFN_RUNTIME_PROFILE=blackwell-cu130
export JEVPFN_CONDA_ENV=JEVPFN
source scripts/activate_conda.sh
exec "${JEVPFN_PYTHON:?}" -u -m src.cluster.setup_extractors --all "$@"
