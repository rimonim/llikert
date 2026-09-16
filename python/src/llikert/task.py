"""Portable scoring tasks (task schema v1). The service validates authoritatively; the
checks here fail early, before anything is sent."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

TASK_SCHEMA_VERSION = 1
RESERVED_CATEGORY_IDS = frozenset({"id", "expected_value"})


def _check_string(value: Any, what: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{what} must be a string")
    if not allow_empty and value == "":
        raise ValueError(f"{what} must not be empty")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(f"{what} is not valid Unicode text") from None
    return value


@dataclass(frozen=True)
class Category:
    id: str
    label: str
    response: str
    value: float | None = None

    def __post_init__(self):
        _check_string(self.id, "category id")
        _check_string(self.label, "category label")
        _check_string(self.response, "category response")
        if self.id in RESERVED_CATEGORY_IDS:
            raise ValueError(f"category id {self.id!r} is reserved")
        if self.value is not None:
            if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
                raise TypeError("category value must be a number")
            if not math.isfinite(self.value):
                raise ValueError("category value must be finite")
            object.__setattr__(self, "value", float(self.value))


@dataclass(frozen=True)
class Example:
    text: str
    category_id: str

    def __post_init__(self):
        _check_string(self.text, "example text")
        _check_string(self.category_id, "example category_id")


@dataclass(frozen=True, init=False)
class ScoringTask:
    """A scoring task.

    Build from parallel sequences::

        ScoringTask(name=..., instructions=..., categories=["description", "question"],
                    responses=["A", "B"], values=None, ordered=False)

    ``categories`` are human-readable labels; category ids default to the labels (which
    must then be unique) or are given with ``ids``. Alternatively pass a sequence of
    :class:`Category` records and no ``responses``.
    """

    name: str
    instructions: str
    categories: tuple[Category, ...]
    ordered: bool
    examples: tuple[Example, ...] = field(default=())
    schema_version: int = TASK_SCHEMA_VERSION

    def __init__(
        self,
        name: str,
        instructions: str,
        categories: Sequence[str] | Sequence[Category],
        responses: Sequence[str] | None = None,
        values: Sequence[float] | None = None,
        ordered: bool = False,
        ids: Sequence[str] | None = None,
        examples: Iterable[Example | tuple[str, str] | Mapping[str, str]] | None = None,
    ):
        _check_string(name, "name")
        _check_string(instructions, "instructions")
        if not isinstance(ordered, bool):
            raise TypeError("ordered must be True or False")
        categories = list(categories)
        if categories and all(isinstance(c, Category) for c in categories):
            if responses is not None or values is not None or ids is not None:
                raise ValueError("pass either Category records or labels with responses/values/ids, not both")
            records = categories
        else:
            if responses is None:
                raise ValueError("responses are required when categories are labels")
            labels = categories
            responses = list(responses)
            ids = list(ids) if ids is not None else labels
            if values is not None:
                values = list(values)
            lengths = {len(labels), len(responses), len(ids)} | ({len(values)} if values is not None else set())
            if len(lengths) != 1:
                raise ValueError("categories, responses, ids and values must have the same length")
            records = [
                Category(id=i, label=label, response=r, value=None if values is None else v)
                for i, label, r, v in zip(ids, labels, responses, values if values is not None else [None] * len(labels))
            ]
        if len(records) < 2:
            raise ValueError("a task needs at least two categories")
        if len({c.id for c in records}) != len(records):
            raise ValueError("category ids must be unique (labels are used as ids unless ids are given)")
        if len({c.response for c in records}) != len(records):
            raise ValueError("response strings must be unique")
        if len({c.value is None for c in records}) != 1:
            raise ValueError("values must be supplied for every category or for none")

        parsed_examples = []
        for ex in examples or ():
            if isinstance(ex, Example):
                parsed_examples.append(ex)
            elif isinstance(ex, Mapping):
                parsed_examples.append(Example(text=ex["text"], category_id=ex["category_id"]))
            else:
                text, category_id = ex
                parsed_examples.append(Example(text=text, category_id=category_id))
        known = {c.id for c in records}
        for ex in parsed_examples:
            if ex.category_id not in known:
                raise ValueError(f"example category_id {ex.category_id!r} does not name a category")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "instructions", instructions)
        object.__setattr__(self, "categories", tuple(records))
        object.__setattr__(self, "ordered", ordered)
        object.__setattr__(self, "examples", tuple(parsed_examples))
        object.__setattr__(self, "schema_version", TASK_SCHEMA_VERSION)

    @property
    def has_values(self) -> bool:
        return self.categories[0].value is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "instructions": self.instructions,
            "categories": [{"id": c.id, "label": c.label, "response": c.response, "value": c.value} for c in self.categories],
            "ordered": self.ordered,
            "examples": [{"text": e.text, "category_id": e.category_id} for e in self.examples],
        }

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> ScoringTask:
        allowed = {"schema_version", "name", "instructions", "categories", "ordered", "examples"}
        unknown = set(obj) - allowed
        if unknown:
            raise ValueError(f"unknown task fields: {sorted(unknown)}")
        if obj.get("schema_version") != TASK_SCHEMA_VERSION:
            raise ValueError(f"unsupported task schema_version {obj.get('schema_version')!r}")
        records = []
        for c in obj.get("categories", []):
            unknown = set(c) - {"id", "label", "response", "value"}
            if unknown:
                raise ValueError(f"unknown category fields: {sorted(unknown)}")
            records.append(Category(id=c["id"], label=c["label"], response=c["response"], value=c.get("value")))
        return cls(
            name=obj.get("name"),
            instructions=obj.get("instructions"),
            categories=records,
            ordered=obj.get("ordered", False),
            examples=obj.get("examples", []),
        )

    def to_json(self, path: str | os.PathLike) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write("\n")

    @classmethod
    def from_json(cls, path: str | os.PathLike) -> ScoringTask:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
