"""Load process credentials or the owner's ignored local file; never emit their value."""

import json
import os

from src.utils import paths

LOCAL_CREDENTIAL_PATH = paths.REPO_ROOT / "secrets/typesafe.local.json"


class CredentialFormatError(ValueError):
    """A local paste/input problem, before authentication or inference."""


def credentials_available():
    return "TYPESAFE_API_KEY" in os.environ or (
        not paths.on_vsc() and LOCAL_CREDENTIAL_PATH.is_file()
    )


def api_key():
    if "TYPESAFE_API_KEY" in os.environ:
        value = os.environ["TYPESAFE_API_KEY"]
    elif not paths.on_vsc() and LOCAL_CREDENTIAL_PATH.is_file():
        try:
            value = json.loads(LOCAL_CREDENTIAL_PATH.read_text(encoding="utf-8-sig"))["api_key"]
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
