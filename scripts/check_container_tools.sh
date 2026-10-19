#!/bin/bash
# Read-only login check; also sourced by the build to retain the resolved PATH.
set -euo pipefail
module load hpc-container-wrapper/0.4.2
for JEVPFN_TOOL in apptainer wrap-install conda-containerize; do
    command -v "$JEVPFN_TOOL" >/dev/null || {
        echo "Container prerequisite missing after module load: $JEVPFN_TOOL" >&2
        exit 2
    }
done

# The wrapper invokes mksquashfs directly, whereas Apptainer can use its private
# bundled binary. Resolve that documented directory from the installed buildcfg;
# do not guess paths, evaluate configuration as shell code or install packages.
if ! command -v mksquashfs >/dev/null; then
    JEVPFN_LIBEXECDIR=''
    if JEVPFN_APPTAINER_CONFIG="$(apptainer buildcfg)"; then
        while IFS= read -r JEVPFN_CONFIG_LINE; do
            case "$JEVPFN_CONFIG_LINE" in
                LIBEXECDIR=/*) JEVPFN_LIBEXECDIR="${JEVPFN_CONFIG_LINE#LIBEXECDIR=}" ;;
            esac
        done <<< "$JEVPFN_APPTAINER_CONFIG"
    fi
    if [[ -n "$JEVPFN_LIBEXECDIR" && -x "$JEVPFN_LIBEXECDIR/apptainer/bin/mksquashfs" ]]; then
        export PATH="$JEVPFN_LIBEXECDIR/apptainer/bin:$PATH"
    else
        echo 'mksquashfs is absent from PATH and the installed Apptainer bundle.' >&2
        echo 'No build started. Ask VSC support for squashfs-tools or a working module; retain this diagnostic.' >&2
        exit 2
    fi
fi
echo "Apptainer: $(command -v apptainer)"
apptainer --version
echo "SquashFS packer: $(command -v mksquashfs)"
mksquashfs -version || {
    echo 'mksquashfs exists but cannot execute; no build started.' >&2
    exit 2
}
echo 'Container build prerequisites passed; no image was built by this check.'
