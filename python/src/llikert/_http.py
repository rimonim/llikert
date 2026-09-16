"""HTTP transport with the shared bounded retry policy (decision D16)."""

from __future__ import annotations

import email.utils
import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from llikert._errors import AuthenticationError, FingerprintMismatch, PrepareError, ServiceError, ServiceUnavailable, TransportError
from llikert._version import PROTOCOL_VERSION, __version__

RETRY_STATUSES = frozenset({429, 502, 503, 504})


@dataclass(frozen=True)
class RetryPolicy:
    max_tries: int = 6
    base_seconds: float = 1.0
    cap_seconds: float = 60.0
    sleep: Callable[[float], None] = field(default=time.sleep, repr=False, compare=False)

    def backoff(self, attempt: int) -> float:
        """Full jitter: uniform in [0, min(cap, base * 2**attempt)]."""
        return random.uniform(0, min(self.cap_seconds, self.base_seconds * 2**attempt))


def retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, when.timestamp() - time.time())


def _excerpt(text: str, limit: int = 200) -> str:
    text = re.sub(r"<[^>]*>", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:limit]


def error_from_response(resp: httpx.Response) -> ServiceError:
    try:
        body = resp.json()
        err = body["error"]
        code, message = str(err["code"]), str(err["message"])
        details, request_id = err.get("details"), err.get("request_id")
    except (ValueError, KeyError, TypeError):
        code = "http_error"
        message = f"HTTP {resp.status_code} without a protocol error body"
        excerpt = _excerpt(resp.text)
        if excerpt:
            message += f": {excerpt}"
        details, request_id = None, None
    status = resp.status_code
    if status in (401, 403):
        cls = AuthenticationError
    elif status == 409:
        cls = FingerprintMismatch
    elif status == 422 and code in {"invalid_response_codes", "invalid_task", "task_exceeds_context", "render_error", "boundary_error"}:
        cls = PrepareError
    elif status in (429, 503):
        cls = ServiceUnavailable
    else:
        cls = ServiceError
    return cls(status, code, message, details, request_id)


class Transport:
    def __init__(
        self,
        url: str,
        token: str | None = None,
        timeout: float = 300.0,
        retry: RetryPolicy = RetryPolicy(),
        scale_up_timeout: int | None = None,
        client: httpx.Client | None = None,
    ):
        if not url:
            raise ValueError("a service URL is required")
        self.url = url.rstrip("/")
        headers = {"User-Agent": f"llikert-python/{__version__}", "Accept": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if scale_up_timeout is not None:
            headers["X-Scale-Up-Timeout"] = str(int(scale_up_timeout))
        self.retry = retry
        # redirects are not followed, so credentials never reach another host
        self._client = client or httpx.Client(timeout=httpx.Timeout(timeout, connect=min(30.0, timeout)), follow_redirects=False)
        self._headers = headers

    def close(self) -> None:
        self._client.close()

    def request(self, method: str, path: str, body: Any = None, *, retry: bool = True, auth: bool = True) -> Any:
        payload = None
        if body is not None:
            payload = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        headers = dict(self._headers)
        if not auth:
            headers.pop("Authorization", None)
        if payload is not None:
            headers["Content-Type"] = "application/json"
        tries = self.retry.max_tries if retry else 1
        last_error: Exception | None = None
        for attempt in range(tries):
            try:
                resp = self._client.request(method, self.url + path, content=payload, headers=headers)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = TransportError(f"{type(exc).__name__} contacting the scoring service")
                if attempt + 1 < tries:
                    self.retry.sleep(self.retry.backoff(attempt))
                    continue
                raise last_error from None
            if resp.status_code == 200:
                if resp.headers.get("LLikert-Protocol", str(PROTOCOL_VERSION)) != str(PROTOCOL_VERSION):
                    raise ServiceError(200, "protocol_version_mismatch", "the service speaks a different protocol version")
                try:
                    return resp.json()
                except ValueError:
                    raise ServiceError(200, "invalid_response", "the service returned a non-JSON success response") from None
            error = error_from_response(resp)
            if resp.status_code in RETRY_STATUSES and attempt + 1 < tries:
                wait = retry_after_seconds(resp.headers.get("Retry-After"))
                self.retry.sleep(min(self.retry.cap_seconds, wait) if wait is not None else self.retry.backoff(attempt))
                last_error = error
                continue
            raise error
        raise last_error  # pragma: no cover - loop always returns or raises
