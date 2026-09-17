"""Portable task schema, validation, canonical form, and identity hash (plan section 5.1)."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from llikert.server import prompt as prompts
from llikert.server.canonical import identity_hash
from llikert.server.errors import ServiceError, invalid

TASK_SCHEMA_VERSION = 1
# category ids become column names next to these in client result tables
RESERVED_CATEGORY_IDS = frozenset({"id", "expected_value"})


def check_utf8(value: str) -> str:
    """Reject strings that cannot be encoded as UTF-8 (lone surrogates from JSON escapes)."""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("string is not valid Unicode text (lone surrogate)") from None
    return value


class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False, frozen=True)


class Category(StrictModel):
    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    response: str = Field(min_length=1)
    value: float | None = None

    @field_validator("id", "label", "response")
    @classmethod
    def _utf8(cls, v: str) -> str:
        return check_utf8(v)

    @field_validator("id")
    @classmethod
    def _not_reserved(cls, v: str) -> str:
        if v in RESERVED_CATEGORY_IDS:
            raise ValueError(f"category id {v!r} is reserved")
        return v

    @field_validator("value")
    @classmethod
    def _finite(cls, v: float | None) -> float | None:
        if v is not None and not math.isfinite(v):
            raise ValueError("value must be finite")
        return v


class Example(StrictModel):
    item: str = Field(min_length=1)
    category_id: str = Field(min_length=1)

    @field_validator("item", "category_id")
    @classmethod
    def _utf8(cls, v: str) -> str:
        return check_utf8(v)


class PromptFormat(StrictModel):
    """Templates for turning the task and one item into chat messages (docs/prompts.md)."""

    system: str | None = prompts.DEFAULT_SYSTEM
    user: str = prompts.DEFAULT_USER
    scale: str = prompts.DEFAULT_SCALE
    code: str = Field(default=prompts.DEFAULT_CODE, min_length=1)
    code_separator: str = prompts.DEFAULT_CODE_SEPARATOR
    answer_instruction: str = prompts.DEFAULT_ANSWER_INSTRUCTION

    @field_validator("system", "user", "scale", "code", "code_separator", "answer_instruction")
    @classmethod
    def _utf8(cls, v: str | None) -> str | None:
        return v if v is None else check_utf8(v)


class Task(StrictModel):
    schema_version: Literal[1]
    name: str = Field(min_length=1)
    instructions: str
    categories: list[Category] = Field(min_length=2)
    ordered: bool = False
    examples: list[Example] = Field(default_factory=list)
    prompt: PromptFormat = Field(default_factory=PromptFormat)

    @field_validator("name", "instructions")
    @classmethod
    def _utf8(cls, v: str) -> str:
        return check_utf8(v)

    @model_validator(mode="after")
    def _consistent(self) -> Task:
        ids = [c.id for c in self.categories]
        if len(set(ids)) != len(ids):
            raise ValueError("category ids must be unique")
        responses = [c.response for c in self.categories]
        if len(set(responses)) != len(responses):
            raise ValueError("category response strings must be unique")
        n_values = sum(c.value is not None for c in self.categories)
        if n_values not in (0, len(self.categories)):
            raise ValueError("values must be supplied for every category or for none")
        known = set(ids)
        for i, ex in enumerate(self.examples):
            if ex.category_id not in known:
                raise ValueError(f"examples[{i}].category_id does not name a category")
        try:
            prompts.validate_prompt_format(self.prompt.model_dump(), has_values=n_values > 0)
        except prompts.TemplateError as exc:
            raise ValueError(str(exc)) from None
        return self

    @property
    def has_values(self) -> bool:
        return self.categories[0].value is not None

    def category(self, category_id: str) -> Category:
        return next(c for c in self.categories if c.id == category_id)

    def canonical(self) -> dict[str, Any]:
        """All fields explicit, defaults filled, order of lists preserved."""
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "instructions": self.instructions,
            "categories": [
                {"id": c.id, "label": c.label, "response": c.response, "value": c.value} for c in self.categories
            ],
            "ordered": self.ordered,
            "examples": [{"item": e.item, "category_id": e.category_id} for e in self.examples],
            "prompt": self.prompt.model_dump(),
        }

    def identity(self) -> str:
        return identity_hash(self.canonical())


def safe_validation_details(exc: ValidationError) -> list[dict[str, Any]]:
    """Validation errors without echoing submitted values (which may be research text)."""
    return [
        {"loc": [str(p) for p in err.get("loc", ())], "type": err.get("type"), "message": err.get("msg")}
        for err in exc.errors(include_url=False, include_input=False)
    ]


def parse_task(obj: Any) -> Task:
    try:
        return Task.model_validate(obj)
    except ValidationError as exc:
        raise invalid("invalid_task", "task failed validation", {"errors": safe_validation_details(exc)}) from None


__all__ = ["Category", "Example", "Task", "TASK_SCHEMA_VERSION", "parse_task", "ServiceError"]
