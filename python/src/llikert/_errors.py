"""Client exceptions. Service errors keep the protocol's machine-readable code."""

from __future__ import annotations

from typing import Any


class LlikertError(Exception):
    """Base class for llikert client errors."""


def _describe(status: int, code: str, message: str, details: dict[str, Any] | None) -> str:
    """The error line, followed by the field-level problems of a 422. A field the service
    rejects as unknown means it is older than this package, which is worth saying outright."""
    text = f"[{status} {code}] {message}"
    errors = (details or {}).get("errors") or []
    if not isinstance(errors, list):
        return text
    for err in errors[:5]:
        where = ".".join(str(part) for part in err.get("loc", ()))
        text += f"\n  - {where + ': ' if where else ''}{err.get('message') or err.get('type') or 'invalid'}"
    if len(errors) > 5:
        text += f"\n  - {len(errors) - 5} more problem(s) not shown"
    unknown = sorted({".".join(str(p) for p in e.get("loc", ())) for e in errors if e.get("type") == "extra_forbidden"})
    if unknown:
        text += (f"\n  The service does not know the field {', '.join(unknown)}, which this package sends. "
                 "It is probably running an older version of llikert; ask whoever runs the scoring service "
                 "to update and restart it.")
    return text


class ServiceError(LlikertError):
    """The service answered with a protocol error envelope (or an unparseable error)."""

    def __init__(self, status: int, code: str, message: str, details: dict[str, Any] | None = None, request_id: str | None = None):
        super().__init__(_describe(status, code, message, details))
        self.status = status
        self.code = code
        self.message = message
        self.details = details
        self.request_id = request_id


class AuthenticationError(ServiceError):
    """401/403: missing or invalid credentials."""


class FingerprintMismatch(ServiceError):
    """The prepared task or checkpoint belongs to a different model or execution configuration."""


class PrepareError(ServiceError):
    """The task is invalid for the loaded model (for example, a response code is not one token)."""

    @property
    def problems(self) -> list[dict[str, Any]]:
        return (self.details or {}).get("problems", [])

    @property
    def suggestions(self) -> dict[str, Any] | None:
        return (self.details or {}).get("suggestions")


class ServiceUnavailable(ServiceError):
    """The service did not become ready, or stayed busy, within the retry budget."""


class TransportError(LlikertError):
    """No usable HTTP response (connection failure or timeout) after bounded retries."""


class ProtocolError(LlikertError):
    """The service response does not match the protocol (wrong version, shape, or identity)."""


class CheckpointError(LlikertError):
    """The checkpoint cannot be used: it belongs to a different run, is locked, or is corrupt."""
