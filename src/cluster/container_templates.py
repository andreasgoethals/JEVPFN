"""Narrow, checked fixes to a private copy of the site's 0.4.2 Conda template."""

MINIFORGE_VERSION = "25.3.1-0"
MINIFORGE_SHA256 = "376b160ed8130820db0ab0f3826ac1fc85923647f75c1b8231166e3d559ab768"


def pinned_miniforge(source: str) -> str:
    """Never execute a GitHub error page after a failed latest-version lookup."""
    lines = source.splitlines()
    lookup = [
        i
        for i, line in enumerate(lines)
        if "CW_CONDA_VERSION=" in line and "curl" in line and "miniforge" in line
    ]
    download = [
        i for i, line in enumerate(lines) if "curl" in line and "--output Miniforge_inst.sh" in line
    ]
    if len(lookup) != 1 or len(download) != 1:
        raise ValueError("Site Miniforge template changed; review before applying download fix")
    lines[lookup[0]] = f"        CW_CONDA_VERSION={MINIFORGE_VERSION}"
    lines[
        download[0]
    ] = f"""        [[ "$CW_CONDA_ARCH" == Linux-x86_64 ]] || {{ echo 'Unreviewed Miniforge architecture' >&2; exit 2; }}
        jevpfn_miniforge_url="https://github.com/conda-forge/miniforge/releases/download/$CW_CONDA_VERSION/Miniforge3-$CW_CONDA_VERSION-$CW_CONDA_ARCH.sh"
        curl --fail --location --show-error --retry 2 "$jevpfn_miniforge_url" --output Miniforge_inst.sh
        jevpfn_expected_sha={MINIFORGE_SHA256}
        printf '%s  Miniforge_inst.sh\\n' "$jevpfn_expected_sha" | sha256sum --check -
        head -c 2 Miniforge_inst.sh | grep -q '^#!' || {{ echo 'Downloaded Miniforge installer is not a shell script' >&2; exit 2; }}"""
    for i, line in enumerate(lines):
        if "follow_log" in line and "_inst_miniforge.log" in line:
            lines[i] += '\nwait "$inst_pid"\n[[ -x "$CW_INSTALLATION_PATH/miniforge/bin/conda" ]]'
    return "\n".join(lines) + "\n"


def explicit_conda_prefix(source: str) -> str:
    """Avoid name lookup selecting an existing host env1 during create or activation."""
    replacements = {
        "mamba $_EC create --name $CW_ENV_NAME $_FF": 'mamba $_EC create --prefix "$env_root" $_FF',
        "conda $_EC create --name $CW_ENV_NAME $_FF": '"$CW_INSTALLATION_PATH/miniforge/bin/conda" $_EC create --prefix "$env_root" $_FF',
        "conda activate $CW_ENV_NAME": 'conda activate "$env_root"',
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError("Site Conda template changed; review before applying prefix fix")
        source = source.replace(old, new)
    return source
