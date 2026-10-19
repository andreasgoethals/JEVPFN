#!/bin/bash
# hpc-container-wrapper 0.4.2 wrap template, applied only to a private scratch copy.
# Container files need bytes, links and executable bits, not host Lustre ACLs/owners.
# Plain recursive copying applies normal destination permissions instead of source ACLs.
cp -R --preserve=links,timestamps -- "${CW_SOURCE_MOUNT_POINT:?}/." "${CW_INSTALLATION_PATH:?}"
cd "${CW_WORKDIR:?}"
source "$CW_INSTALLATION_PATH/_post_install.sh"
