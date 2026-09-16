"""HTTP protocol v1 (plan section 9): /health, /v1/info, /v1/prepare, /v1/score.

Request bodies are parsed here rather than by FastAPI so that JSON is strict: UTF-8
only, no NaN/Infinity, no duplicate keys, unknown fields rejected. All inference and
tokenization runs on one worker thread behind a bounded queue; /health is answered
from the event loop and stays responsive while scoring runs.
"""

from __future__ import annotations

import asyncio
import enum
import hmac
import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.responses import Response
from pydantic import BaseModel, ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from llikert._version import PROTOCOL_VERSION
from llikert.server.errors import ServiceError, bad_request, invalid
from llikert.server.schemas import PrepareRequest, ScoreRequest, openapi_components
from llikert.server.service import ScoringService, StartupError
from llikert.server.task import safe_validation_details

log = logging.getLogger("llikert.server")
PROTOCOL_HEADER = "LLikert-Protocol"


class AuthMode(str, enum.Enum):
    NONE = "none"
    TOKEN = "token"
    PLATFORM = "platform"


@dataclass(frozen=True)
class HttpSettings:
    auth: AuthMode = AuthMode.NONE
    token: str | None = None
    max_body_bytes: int = 8 * 1024 * 1024
    queue_limit: int = 4
    retry_after_seconds: int = 2


# -- strict JSON ---------------------------------------------------------------------


def _reject_constant(name: str):
    raise ValueError(f"non-standard JSON constant {name}")


def _no_duplicates(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError("duplicate object key")
        out[k] = v
    return out


def parse_json(body: bytes) -> Any:
    try:
        text = body.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise bad_request("invalid_json", "request body is not valid UTF-8") from None
    try:
        return json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_no_duplicates)
    except (ValueError, RecursionError):
        raise bad_request("invalid_json", "request body is not strict JSON") from None


def json_response(status: int, content: Any, headers: dict[str, str] | None = None) -> Response:
    body = json.dumps(content, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    h = {PROTOCOL_HEADER: str(PROTOCOL_VERSION)}
    h.update(headers or {})
    return Response(content=body, status_code=status, media_type="application/json", headers=h)


def error_response(exc: ServiceError, request_id: str | None, headers: dict[str, str] | None = None) -> Response:
    return json_response(exc.status, exc.envelope(request_id), headers)


# -- application ---------------------------------------------------------------------


class _State:
    def __init__(self):
        self.service: ScoringService | None = None
        self.startup_error: str | None = None
        self.pending = 0


def create_app(load_service: Callable[[], ScoringService], settings: HttpSettings = HttpSettings()) -> FastAPI:
    """Build the app. ``load_service`` runs on the worker thread at startup and must
    return a ready service (smoke check passed) or raise."""
    if settings.auth == AuthMode.TOKEN and not settings.token:
        raise ValueError("token auth requires a token")
    state = _State()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llikert-inference")

    def _load():
        try:
            service = load_service()
            if not service.ready:
                service.smoke_check()
            state.service = service
            log.info("service ready: engine_fingerprint=%s", service.fingerprint)
        except StartupError as exc:
            state.startup_error = str(exc)
            log.error("startup failed: %s", exc)
        except Exception as exc:  # noqa: BLE001 - never leak details; keep the process alive for /health
            state.startup_error = type(exc).__name__
            log.error("startup failed: %s", type(exc).__name__)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        executor.submit(_load)
        try:
            yield
        finally:
            def _close():
                if state.service is not None:
                    state.service.adapter.close()
            executor.submit(_close).result()
            executor.shutdown(wait=True)

    app = FastAPI(
        title="LLikert scoring service",
        version=str(PROTOCOL_VERSION),
        lifespan=lifespan,
        # interactive docs and the schema are served only without auth (loopback use);
        # `llikert export-schemas` writes the same OpenAPI document offline
        docs_url="/docs" if settings.auth == AuthMode.NONE else None,
        openapi_url="/openapi.json" if settings.auth == AuthMode.NONE else None,
        redoc_url=None,
    )
    app.state.llikert = state

    def authorized(request: Request) -> bool:
        if settings.auth != AuthMode.TOKEN:
            return True
        header = request.headers.get("authorization", "")
        scheme, _, credential = header.partition(" ")
        return scheme.lower() == "bearer" and hmac.compare_digest(credential.encode(), settings.token.encode())

    async def read_body(request: Request) -> bytes:
        declared = request.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > settings.max_body_bytes:
            raise ServiceError(413, "body_too_large", "request body exceeds max_body_bytes")
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > settings.max_body_bytes:
                raise ServiceError(413, "body_too_large", "request body exceeds max_body_bytes")
            chunks.append(chunk)
        return b"".join(chunks)

    async def run_inference(fn: Callable[[], Any]) -> Any:
        if state.pending >= settings.queue_limit:
            raise ServiceError(429, "queue_full", "the service is busy; retry later")
        state.pending += 1
        try:
            return await asyncio.get_running_loop().run_in_executor(executor, fn)
        finally:
            state.pending -= 1

    def service_or_503() -> ScoringService:
        if state.service is None or not state.service.healthy:
            code = "startup_failed" if state.startup_error else "starting"
            raise ServiceError(503, code, "the scoring engine is not ready")
        return state.service

    async def handle(request: Request, handler: Callable[[Request], Any]) -> Response:
        try:
            if not authorized(request):
                raise ServiceError(401, "unauthorized", "missing or invalid credentials")
            return await handler(request)
        except ServiceError as exc:
            headers = {"Retry-After": str(settings.retry_after_seconds)} if exc.status in (429, 503) else None
            return error_response(exc, getattr(request.state, "request_id", None), headers)
        except Exception as exc:  # noqa: BLE001
            log.error("internal error: %s", type(exc).__name__)
            return error_response(ServiceError(500, "internal_error", "internal server error"), getattr(request.state, "request_id", None))

    def validated(model: type[BaseModel], obj: Any) -> BaseModel:
        try:
            return model.model_validate(obj)
        except ValidationError as exc:
            raise invalid("invalid_request", "request failed validation", {"errors": safe_validation_details(exc)}) from None
        except UnicodeEncodeError:
            raise invalid("invalid_encoding", "request contains invalid Unicode") from None

    @app.get("/health", include_in_schema=True)
    async def health() -> Response:
        service = state.service
        if service is not None and service.healthy:
            return json_response(200, {"status": "ready"})
        return json_response(503, {"status": "unavailable"}, {"Retry-After": str(settings.retry_after_seconds)})

    @app.get("/v1/info")
    async def info(request: Request) -> Response:
        async def handler(_: Request) -> Response:
            return json_response(200, service_or_503().info())

        return await handle(request, handler)

    @app.post(
        "/v1/prepare",
        openapi_extra={"requestBody": {"required": True, "content": {"application/json": {"schema": {"$ref": "#/components/schemas/PrepareRequest"}}}}},
    )
    async def prepare(request: Request) -> Response:
        async def handler(req: Request) -> Response:
            body = validated(PrepareRequest, parse_json(await read_body(req)))
            req.state.request_id = body.request_id or str(uuid.uuid4())
            service = service_or_503()
            out = await run_inference(lambda: service.prepare(body.task, body.preview_text))
            out["request_id"] = req.state.request_id
            return json_response(200, out)

        return await handle(request, handler)

    @app.post(
        "/v1/score",
        openapi_extra={"requestBody": {"required": True, "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ScoreRequest"}}}}},
    )
    async def score(request: Request) -> Response:
        async def handler(req: Request) -> Response:
            body = validated(ScoreRequest, parse_json(await read_body(req)))
            req.state.request_id = body.request_id or str(uuid.uuid4())
            service = service_or_503()
            items = [item.model_dump() for item in body.items]
            out = await run_inference(lambda: service.score(body.prepared, items))
            out["request_id"] = req.state.request_id
            statuses: dict[str, int] = {}
            for r in out["results"]:
                statuses[r["status"]] = statuses.get(r["status"], 0) + 1
            log.info("score request_id=%s items=%d statuses=%s elapsed_ms=%s", req.state.request_id, len(items), statuses, out["meta"]["elapsed_ms"])
            return json_response(200, out)

        return await handle(request, handler)

    @app.exception_handler(StarletteHTTPException)
    async def http_exception(request: Request, exc: StarletteHTTPException) -> Response:
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return error_response(ServiceError(exc.status_code, code, code.replace("_", " ")), None)

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        from fastapi.openapi.utils import get_openapi

        schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
        schema.setdefault("components", {}).setdefault("schemas", {}).update(openapi_components())
        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi
    return app
