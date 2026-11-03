#!/bin/bash
# Explicit private deployment selection; never infer an allocation from the cluster.
if [[ -n "${JEVPFN_DURABLE_ROOT:-${JEVPFN_PROJECT_ROOT:-${VSC_PROJECT_ROOT:-}}}" ]]; then
    export JEVPFN_DURABLE_ROOT="${JEVPFN_DURABLE_ROOT:-${JEVPFN_PROJECT_ROOT:-$VSC_PROJECT_ROOT}}"
    export JEVPFN_PROJECT_ROOT="$JEVPFN_DURABLE_ROOT"
    if [[ "$JEVPFN_DURABLE_ROOT" == /gpfs*/project/* ]]; then export JEVPFN_SHARED_IMMUTABLES=1; fi
    return 0
fi
if [[ "${JEVPFN_STORAGE_MODE:-}" != scratch_mirror && -n "${JEVPFN_STAGING_ROOT:-}" ]]; then
    export JEVPFN_DURABLE_ROOT="$JEVPFN_STAGING_ROOT" JEVPFN_PROJECT_ROOT="$JEVPFN_STAGING_ROOT"
    return 0
fi
jevpfn_storage_exports="$(/usr/bin/python3 -I "${BASH_SOURCE[0]%/*}/../src/cluster/storage_config.py")" || return
eval "$jevpfn_storage_exports"
if [[ "$JEVPFN_DURABLE_ROOT" == /gpfs*/project/* ]]; then export JEVPFN_SHARED_IMMUTABLES=1; fi
unset jevpfn_storage_exports
