"""Load process credentials or the owner's ignored local file; never emit their value."""

import json
import os

from src.utils import paths

LOCAL_CREDENTIAL_PATH = paths.REPO_ROOT / "secrets/typesafe.local.json"


class CredentialFormatError(ValueError):
    """A local paste/input problem, before authentication or inference."""


def credentials_available():
    return credential_present("TYPESAFE_API_KEY", LOCAL_CREDENTIAL_PATH)


def credential_present(env_var, local_path):
    """Presence only: the ignored file is checked for existence, never read."""
    return env_var in os.environ or (not paths.on_vsc() and local_path.is_file())


def api_key():
    return read_api_key("TYPESAFE_API_KEY", LOCAL_CREDENTIAL_PATH)


def read_api_key(env_var, local_path):
    """Provider-neutral loader: env var first, then the local ignored file (never on VSC)."""
    if env_var in os.environ:
        value = os.environ[env_var]
    elif not paths.on_vsc() and local_path.is_file():
        try:
            value = json.loads(local_path.read_text(encoding="utf-8-sig"))["api_key"]
        except (OSError, ValueError, KeyError, TypeError):
            raise CredentialFormatError(
                "The local credential file must be a JSON object with an api_key string."
            ) from None
    else:
        value = ""
    if not isinstance(value, str):
        raise CredentialFormatError("The configured API key must be a string.")
    value = value.strip()
    if not value or not value.isascii() or not value.isprintable() or " " in value:
        raise CredentialFormatError(
            "The API key is empty or contains whitespace/control/non-ASCII characters. "
            "Copy only the key value. In classic PowerShell, use right-click or Shift+Insert "
            "to paste into a hidden prompt, or use Read-Host -AsSecureString."
        )
    return value


def error_hint(exc):
    # Do not print arbitrary SDK exception messages: they may contain request details.
    if isinstance(exc, CredentialFormatError):
        return "API key input rejected locally; check secrets/typesafe.local.json or paste only the key using right-click or Shift+Insert. No request was sent."
    if getattr(exc, "status", None) == 401:
        return "TypeSafe rejected the configured key. Check secrets/typesafe.local.json or the TYPESAFE_API_KEY environment override; use --check-account before inference."
    return None
