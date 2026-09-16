"""Protocol v1 message models: request validation, response contracts, exported schemas.

Responses are built as plain dicts by the service; tests validate them against the
response models here so the published schemas cannot drift from actual behavior.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from llikert.server.task import Category, Task


class _Strict(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


# -- requests ---------------------------------------------------------------------------


class PrepareRequest(_Strict):
    protocol_version: Literal[1]
    request_id: str | None = Field(default=None, min_length=1, max_length=128)
    task: dict[str, Any]  # validated by the service as a Task (error code invalid_task)
    preview_text: str | None = None


class Item(_Strict):
    id: str = Field(min_length=1, max_length=1024)
    text: str | None

    @field_validator("id")
    @classmethod
    def _utf8(cls, v: str) -> str:
        v.encode("utf-8")  # lone surrogates raise UnicodeEncodeError (a ValueError)
        return v


class ScoreRequest(_Strict):
    protocol_version: Literal[1]
    request_id: str | None = Field(default=None, min_length=1, max_length=128)
    prepared: dict[str, Any]  # verified by the service against its own preparation
    items: list[Item]


# -- responses --------------------------------------------------------------------------


class PreparedCategory(Category):
    token_id: int
    token_piece: str


class TemplateProfileRef(_Strict):
    name: str
    template_sha256: str


class PreparedTask(_Strict):
    prepared_schema_version: Literal[1]
    task: Task
    task_hash: str
    engine_fingerprint: str
    profile: TemplateProfileRef
    boundary_policy: str
    tail_tokens: list[int]
    categories: list[PreparedCategory]
    prepared_hash: str


class Warning_(_Strict):
    code: str
    message: str


class Preview(_Strict):
    prompt: str
    n_prompt_tokens: int


class PrepareDiagnostics(_Strict):
    n_prompt_tokens_without_text: int
    n_ctx: int
    max_text_tokens_approx: int
    warnings: list[Warning_]


class PrepareResponse(_Strict):
    protocol_version: Literal[1]
    request_id: str
    prepared: PreparedTask
    diagnostics: PrepareDiagnostics
    preview: dict[Literal["example", "text"], Preview]


ItemStatusName = Literal[
    "ok", "missing_input", "empty_input", "invalid_encoding", "context_limit",
    "boundary_error", "numerical_error", "inference_error",
]


class ItemError(_Strict):
    code: ItemStatusName
    message: str


class ItemResult(_Strict):
    id: str
    status: ItemStatusName
    error: ItemError | None
    candidate_log_probs: list[float | None]
    candidate_probs: list[float | None]
    probabilities: list[float | None]
    log_coverage: float | None
    coverage: float | None
    n_prompt_tokens: int | None
    prompt_token_sha256: str | None
    warnings: list[Warning_]
    expected_value: float | None = Field(default=None, description="present only for tasks with numeric values")


class ScoreMeta(_Strict):
    elapsed_ms: float
    execution: dict[str, Any]


class ScoreResponse(_Strict):
    protocol_version: Literal[1]
    request_id: str
    engine_fingerprint: str
    prepared_hash: str
    category_ids: list[str]
    results: list[ItemResult]
    meta: ScoreMeta


class ErrorBody(_Strict):
    code: str
    message: str
    details: dict[str, Any] | None = None
    request_id: str | None = None


class ErrorEnvelope(_Strict):
    error: ErrorBody


class InfoLimits(_Strict):
    max_items_per_request: int
    max_body_bytes: int
    queue_limit: int
    n_ctx: int
    max_prompt_tokens: int


class InfoResponse(_Strict):
    protocol_version: Literal[1]
    service: dict[str, str]
    engine_fingerprint: str
    engine: dict[str, Any]
    execution: dict[str, Any]
    limits: InfoLimits
    item_statuses: list[ItemStatusName]


EXPORTED = {
    "task.v1.json": Task,
    "prepared-task.v1.json": PreparedTask,
    "prepare-request.v1.json": PrepareRequest,
    "prepare-response.v1.json": PrepareResponse,
    "score-request.v1.json": ScoreRequest,
    "score-response.v1.json": ScoreResponse,
    "info-response.v1.json": InfoResponse,
    "error.v1.json": ErrorEnvelope,
}


def openapi_components() -> dict[str, Any]:
    components: dict[str, Any] = {}
    for model in EXPORTED.values():
        schema = model.model_json_schema(ref_template="#/components/schemas/{model}")
        components.update(schema.pop("$defs", {}))
        components[model.__name__] = schema
    components["PrepareRequest"]["properties"]["task"] = {"$ref": "#/components/schemas/Task"}
    return components


def _render_all() -> dict[str, str]:
    out = {}
    for name, model in EXPORTED.items():
        schema = model.model_json_schema()
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["$id"] = f"https://llikert.invalid/schemas/{name}"
        out[name] = json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return out


def export_schemas(out_dir: pathlib.Path, check: bool = False) -> int:
    rendered = _render_all()
    if check:
        stale = [n for n, text in rendered.items() if not (out_dir / n).is_file() or (out_dir / n).read_text() != text]
        extra = sorted(p.name for p in out_dir.glob("*.v1.json") if p.name not in rendered) if out_dir.is_dir() else []
        if stale or extra:
            print("schemas out of date: " + ", ".join(stale + extra))
            return 1
        print("schemas up to date")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in rendered.items():
        (out_dir / name).write_text(text)
    print(f"wrote {len(rendered)} schemas to {out_dir}")
    return 0
