#!/bin/bash
# Sourced only by the image builder. Do not modify installed wrappers or images.
jevpfn_control_bin="${JEVPFN_BUILD_PYTHON:?Absolute controller interpreter required}"
jevpfn_control_bin="${jevpfn_control_bin%/*}"
jevpfn_clean_path=""
IFS=: read -r -a jevpfn_build_path_parts <<< "$PATH"
for jevpfn_path_part in "${jevpfn_build_path_parts[@]}"; do
    if [[ -n "$jevpfn_path_part" && "$jevpfn_path_part" != "$jevpfn_control_bin" ]]; then
        jevpfn_clean_path="${jevpfn_clean_path:+$jevpfn_clean_path:}$jevpfn_path_part"
    fi
done
export PATH="$jevpfn_clean_path"
export TERM="${TERM:-dumb}"
hash -r
