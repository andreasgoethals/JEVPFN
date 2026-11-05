#!/bin/bash
# Source after entering the DATA checkout. All callers request one task on one node.
jevpfn_check_provider_memory() {
    [[ "${JEVPFN_PROVIDER:?}" == perplexity ]] || return 0
    local memory="${SLURM_MEM_PER_NODE:-}" cpus="${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-}}"
    if [[ -z "$memory" && "${SLURM_MEM_PER_CPU:-}" =~ ^[0-9]+$ && "$cpus" =~ ^[0-9]+$ ]]; then
        memory=$((10#${SLURM_MEM_PER_CPU} * 10#$cpus))
    fi
    if [[ ! "$memory" =~ ^[0-9]+$ ]] || (( 10#$memory < 112000 )); then
        printf 'Perplexity requires at least 112000 MiB allocated host memory; observed %s. On Mindwell B200 request 14 CPUs at its default 8000 MiB/CPU.\n' "${memory:-unknown}" >&2
        return 2
    fi
    printf 'Perplexity host-memory guard passed: %s MiB allocated.\n' "$memory"
}
jevpfn_check_provider_memory
