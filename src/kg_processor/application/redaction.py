# SPDX-License-Identifier: Apache-2.0
"""Credential redaction for inspectable JSON artifacts.

The processor intentionally writes rich config snapshots, progress metadata,
and extraction traces so operators can understand a run after the fact. This
module keeps those artifacts useful while making sure provider credentials stay
out of logs, local files, and Snowflake VARIANT payloads.
"""

from __future__ import annotations

import re
from typing import Any

_REDACTED = "***"

_SENSITIVE_KEY_EXACT = {
    "access_token",
    "authorization",
    "connection_string",
    "database_url",
    "oauth_token",
    "password",
    "private_key",
    "private_key_path",
    "refresh_token",
    "sas_token",
    "token",
}
_SENSITIVE_KEY_SUFFIXES = (
    "_access_key_id",
    "_api_key",
    "_password",
    "_secret",
    "_token",
)
_SAS_QUERY_PARAM_RE = re.compile(
    r"([?&](?:sig|se|sp|spr|sr|st|sv|skoid|sktid|skt|ske|sks|skv)=)[^&#\s]+",
    flags=re.IGNORECASE,
)
_SECRET_QUERY_PARAM_RE = re.compile(
    r"([?&][^=&#\s]*(?:access_token|api[_-]?key|authorization|password|secret|token)=)"
    r"[^&#\s]+",
    flags=re.IGNORECASE,
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(\b(?:api[_-]?key|authorization|bearer|password|passwd|pwd|secret|token)\b"
    r"\s*[=:]\s*)([^\s,;]+)"
)
_QUOTED_SECRET_ASSIGNMENT_RE = re.compile(
    r"""(?i)(["'](?:api[_-]?key|authorization|password|passwd|pwd|secret|token)["']"""
    r"""\s*:\s*["'])(.*?)(["'])"""
)
_URL_PASSWORD_RE = re.compile(r"(\b[a-z][a-z0-9+.-]*://[^\s:/@]*:)[^\s/@]+(@)", re.IGNORECASE)
_BEARER_RE = re.compile(r"(?i)(\bBearer\s+)[A-Za-z0-9._~+/-]+=*")
_AUTHORIZATION_SCHEME_RE = re.compile(
    r"(?i)(\bAuthorization\b\s*[=:]\s*(?:Basic|Digest|Negotiate|Bearer)\s+)\S+"
)
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")
_ABSOLUTE_URL_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)


def redact_sensitive_data(value: Any) -> Any:
    """Return a JSON-like copy with sensitive mapping keys redacted.

    Redaction is intentionally key-based. We do not inspect arbitrary strings
    because document text, quotes, model outputs, and file names can legitimately
    contain words that look secret-ish. Provider adapters should still avoid
    putting credentials into metadata, and this function is the last safety net.
    """

    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if is_sensitive_key(str(key)):
                redacted[key] = _REDACTED if item else None
            else:
                redacted[key] = redact_sensitive_data(item)
        return redacted
    if isinstance(value, list):
        return [redact_sensitive_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_sensitive_data(item) for item in value)
    if isinstance(value, str) and _ABSOLUTE_URL_RE.match(value):
        return _redact_sensitive_url(value)
    return value


def _redact_sensitive_url(value: str) -> str:
    """Redact credentials from a scalar URL without interpreting ordinary prose."""

    redacted = _SAS_QUERY_PARAM_RE.sub(r"\1***", value)
    redacted = _SECRET_QUERY_PARAM_RE.sub(r"\1***", redacted)
    return _URL_PASSWORD_RE.sub(r"\1***\2", redacted)


def redact_sensitive_text(value: str) -> str:
    """Redact credentials that commonly appear inside exception strings.

    The patterns cover cloud query signatures, DSN/key-value credentials, URL
    userinfo passwords, bearer tokens, and JWTs. They intentionally retain keys,
    usernames, hosts, and parameter names so diagnostics remain actionable.
    """

    redacted = _QUOTED_SECRET_ASSIGNMENT_RE.sub(r"\1***\3", value)
    redacted = _SAS_QUERY_PARAM_RE.sub(r"\1***", redacted)
    redacted = _SECRET_QUERY_PARAM_RE.sub(r"\1***", redacted)
    redacted = _URL_PASSWORD_RE.sub(r"\1***\2", redacted)
    redacted = _AUTHORIZATION_SCHEME_RE.sub(r"\1***", redacted)
    redacted = _BEARER_RE.sub(r"\1***", redacted)
    redacted = _JWT_RE.sub("***", redacted)
    return _SECRET_ASSIGNMENT_RE.sub(r"\1***", redacted)


def is_sensitive_key(key: str) -> bool:
    """Return whether a mapping key should be redacted in review artifacts."""

    lowered = key.lower()
    return (
        lowered == "api_key"
        or lowered in _SENSITIVE_KEY_EXACT
        or lowered.endswith(_SENSITIVE_KEY_SUFFIXES)
        or "secret" in lowered
    )


def credentialed_urls(value: Any, path: str = "") -> list[str]:
    """Name every URL in a JSON-like value that carries a credential.

    Rendering settings into a deployment specification is the one place where
    redacting would be wrong: a spec with ``***`` in an endpoint is one the
    worker cannot use. The caller therefore refuses instead, and this returns
    the dotted paths of the offending fields (``azure_blob.account_url``) so
    the operator knows which value to move into a secret. A URL counts as
    credentialed when the key-agnostic URL redaction would change it: a
    ``user:password@`` userinfo part, a SAS signature, or a token-like query
    parameter.
    """

    if isinstance(value, dict):
        return [
            found
            for key, item in value.items()
            for found in credentialed_urls(item, f"{path}.{key}" if path else str(key))
        ]
    if isinstance(value, list | tuple):
        return [
            found
            for index, item in enumerate(value)
            for found in credentialed_urls(item, f"{path}[{index}]")
        ]
    if (
        isinstance(value, str)
        and _ABSOLUTE_URL_RE.match(value)
        and _redact_sensitive_url(value) != value
    ):
        return [path or "<value>"]
    return []
