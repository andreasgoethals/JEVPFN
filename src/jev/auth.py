"""Validate credential input locally without exposing its value or contacting TypeSafe."""

import os


class CredentialFormatError(ValueError):
    """A local paste/input problem, before authentication or inference."""


def api_key():
    value = os.environ.get("TYPESAFE_API_KEY", "").strip()
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
        return "API key input rejected locally; copy only the key and paste using right-click or Shift+Insert. No request was sent."
    return None
