"""Narrow, checked fixes to a private copy of the site's 0.4.2 Conda template."""


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
