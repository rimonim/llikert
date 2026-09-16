"""Client exceptions. Service errors keep the protocol's machine-readable code."""

from __future__ import annotations

from typing import Any


class LlikertError(Exception):
    """Base class for llikert client errors."""


class ServiceError(LlikertError):
    """The service answered with a protocol error envelope (or an unparseable error)."""

    def __init__(self, status: int, code: str, message: str, details: dict[str, Any] | None = None, request_id: str | None = None):
        super().__init__(f"[{status} {code}] {message}")
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
