"""Error codes shared by the service and the protocol.

Request-level failures raise :class:`ServiceError` and become an HTTP error envelope.
Per-item failures are reported inside a successful score response with an
:class:`ItemStatus` other than ``ok``.
"""

from __future__ import annotations

import enum
from typing import Any


class ItemStatus(str, enum.Enum):
    OK = "ok"
    MISSING_INPUT = "missing_input"
    EMPTY_INPUT = "empty_input"
    INVALID_ENCODING = "invalid_encoding"
    CONTEXT_LIMIT = "context_limit"
    BOUNDARY_ERROR = "boundary_error"
    NUMERICAL_ERROR = "numerical_error"
    INFERENCE_ERROR = "inference_error"


class ServiceError(Exception):
    """A request-level failure with a stable machine-readable code and a safe message."""

    def __init__(self, status: int, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details

    def envelope(self, request_id: str | None = None) -> dict[str, Any]:
        error: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details is not None:
            error["details"] = self.details
        if request_id is not None:
            error["request_id"] = request_id
        return {"error": error}


def bad_request(code: str, message: str, details: dict[str, Any] | None = None) -> ServiceError:
    return ServiceError(400, code, message, details)


def invalid(code: str, message: str, details: dict[str, Any] | None = None) -> ServiceError:
    return ServiceError(422, code, message, details)


def conflict(code: str, message: str, details: dict[str, Any] | None = None) -> ServiceError:
    return ServiceError(409, code, message, details)


def unavailable(code: str, message: str) -> ServiceError:
    return ServiceError(503, code, message)
