"""Portable scoring tasks (task schema v1). The service validates authoritatively; the
checks here fail early, before anything is sent."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from llikert import _prompt

TASK_SCHEMA_VERSION = 1
DEFAULT_ANSWER_INSTRUCTION = "Answer with exactly one of the response codes listed above and nothing else."
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
    """A worked example shown to the model before each item: an item and its category.

    ``category_id`` is the category's id; :class:`ScoringTask` also accepts a response code
    there and stores the matching id.
    """

    item: str
    category_id: str

    def __post_init__(self):
        _check_string(self.item, "example item")
        _check_string(self.category_id, "example category_id")


@dataclass(frozen=True)
class PromptFormat:
    """How a task and one item become chat messages. See docs/prompts.md.

    Each item produces: a system message from ``system`` (omitted when ``None``), one
    user/assistant pair per example, and a user message from ``user``. Placeholders:

    - ``system`` and ``user``: ``{instructions}``, ``{scale}``, ``{answer_instruction}``
      (the task's ``instructions`` and ``answer_instruction``); ``{item}`` only in
      ``user``, exactly once
    - ``scale``: ``{codes}``, the category lines joined with ``code_separator``
    - ``code``, one line per category: ``{response}`` (required), ``{label}``, ``{value}``, ``{id}``

    Write ``{{`` and ``}}`` for literal braces. Placeholders are filled in one pass, so
    braces inside instructions, labels, or items are never treated as placeholders.
    """

    system: str | None = "{instructions}\n\n{scale}"
    user: str = "Text:\n<text>\n{item}\n</text>\n\n{answer_instruction}"
    scale: str = "Response codes:\n{codes}"
    code: str = "{response} = {label}"
    code_separator: str = "\n"

    def __post_init__(self):
        if self.system is not None:
            _check_string(self.system, "prompt system", allow_empty=True)
        for name in ("user", "scale", "code", "code_separator"):
            _check_string(getattr(self, name), f"prompt {name}", allow_empty=name != "code")
        _prompt.validate({**asdict(self)}, has_values=True)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    answer_instruction: str
    categories: tuple[Category, ...]
    ordered: bool
    examples: tuple[Example, ...] = field(default=())
    prompt: PromptFormat = field(default_factory=PromptFormat)
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
        answer_instruction: str = DEFAULT_ANSWER_INSTRUCTION,
        prompt: PromptFormat | None = None,
    ):
        _check_string(name, "name")
        _check_string(instructions, "instructions", allow_empty=True)
        _check_string(answer_instruction, "answer_instruction", allow_empty=True)
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
                parsed_examples.append(Example(item=ex["item"], category_id=ex["category_id"]))
            else:
                item, category_id = ex
                parsed_examples.append(Example(item=item, category_id=category_id))
        parsed_examples = [_resolve_example(ex, records) for ex in parsed_examples]

        prompt = prompt if prompt is not None else PromptFormat()
        if not isinstance(prompt, PromptFormat):
            raise TypeError("prompt must be a PromptFormat")
        if "value" in _prompt.fields(_prompt.parse_template(prompt.code, _prompt.CODE_FIELDS, "prompt.code")) and records[0].value is None:
            raise ValueError("prompt.code uses {value}, but the categories have no values")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "instructions", instructions)
        object.__setattr__(self, "answer_instruction", answer_instruction)
        object.__setattr__(self, "categories", tuple(records))
        object.__setattr__(self, "ordered", ordered)
        object.__setattr__(self, "examples", tuple(parsed_examples))
        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(self, "schema_version", TASK_SCHEMA_VERSION)

    @property
    def has_values(self) -> bool:
        return self.categories[0].value is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "instructions": self.instructions,
            "answer_instruction": self.answer_instruction,
            "categories": [{"id": c.id, "label": c.label, "response": c.response, "value": c.value} for c in self.categories],
            "ordered": self.ordered,
            "examples": [{"item": e.item, "category_id": e.category_id} for e in self.examples],
            "prompt": self.prompt.to_dict(),
        }

    def messages(self, item: str) -> list[dict[str, str]]:
        """The chat messages the model receives for ``item``, built locally (no service needed).

        The model then sees these messages in its own chat format; ``PreparedTask.preview_prompt()``
        shows that exact text after preparation.
        """
        _check_string(item, "item", allow_empty=True)
        return _prompt.build_messages(self.to_dict(), item)

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> ScoringTask:
        allowed = {"schema_version", "name", "instructions", "answer_instruction", "categories", "ordered", "examples", "prompt"}
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
            answer_instruction=obj.get("answer_instruction", DEFAULT_ANSWER_INSTRUCTION),
            prompt=_prompt_from_dict(obj.get("prompt")),
        )

    def to_json(self, path: str | os.PathLike) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write("\n")

    @classmethod
    def from_json(cls, path: str | os.PathLike) -> ScoringTask:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def _resolve_example(example: Example, records: list[Category]) -> Example:
    """An example names its category by id; a response code is accepted too, because that is
    what the example shows the model. Ids win when a value is both an id and another code."""
    ids = [c.id for c in records]
    if example.category_id in ids:
        return example
    matches = [c.id for c in records if c.response == example.category_id]
    if len(matches) == 1:
        return Example(item=example.item, category_id=matches[0])
    raise ValueError(
        f"example category {example.category_id!r} is not one of this task's categories; "
        f"use a category id ({', '.join(repr(i) for i in ids)}) "
        f"or a response code ({', '.join(repr(c.response) for c in records)})"
    )


def _prompt_from_dict(obj: Mapping[str, Any] | None) -> PromptFormat:
    if obj is None:
        return PromptFormat()
    unknown = set(obj) - {"system", "user", "scale", "code", "code_separator"}
    if unknown:
        raise ValueError(f"unknown prompt fields: {sorted(unknown)}")
    return PromptFormat(**obj)
