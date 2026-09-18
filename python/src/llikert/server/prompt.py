"""Prompt formats: how a task and one item become chat messages (docs/prompts.md).

Templates are plain strings with ``{placeholder}`` fields; ``{{`` and ``}}`` are literal
braces. Substitution is a single pass: substituted values (instructions, items, labels)
are never parsed as templates, so an item that contains ``{scale}`` stays literal.
"""

from __future__ import annotations

import re
from typing import Any

from llikert.server.canonical import _number

DEFAULT_SYSTEM = "{instructions}\n\n{scale}"
DEFAULT_USER = "Text:\n<text>\n{item}\n</text>\n\n{answer_instruction}"
DEFAULT_SCALE = "Response codes:\n{codes}"
DEFAULT_CODE = "{response} = {label}"
DEFAULT_CODE_SEPARATOR = "\n"
DEFAULT_ANSWER_INSTRUCTION = "Answer with exactly one of the response codes listed above and nothing else."

MESSAGE_FIELDS = {"instructions", "scale", "answer_instruction", "item"}
SCALE_FIELDS = {"codes"}
CODE_FIELDS = {"response", "label", "value", "id"}

_TOKEN = re.compile(r"\{\{|\}\}|\{([^{}]*)\}|\{|\}")


class TemplateError(ValueError):
    pass


def parse_template(template: str, allowed: set[str], where: str) -> list[tuple[str, str]]:
    """Split a template into ("text", s) and ("field", name) parts, validating field names."""
    parts: list[tuple[str, str]] = []
    pos = 0
    for m in _TOKEN.finditer(template):
        if m.start() > pos:
            parts.append(("text", template[pos : m.start()]))
        token = m.group(0)
        if token == "{{":
            parts.append(("text", "{"))
        elif token == "}}":
            parts.append(("text", "}"))
        elif m.group(1) is not None:
            name = m.group(1)
            if name not in allowed:
                allowed_list = ", ".join("{" + a + "}" for a in sorted(allowed))
                raise TemplateError(f"{where} uses unknown placeholder {{{name}}}; allowed: {allowed_list} (write {{{{ and }}}} for literal braces)")
            parts.append(("field", name))
        else:
            raise TemplateError(f"{where} has an unmatched brace; write {{{{ or }}}} for a literal brace")
        pos = m.end()
    if pos < len(template):
        parts.append(("text", template[pos:]))
    return parts


def fields(parts: list[tuple[str, str]]) -> list[str]:
    return [value for kind, value in parts if kind == "field"]


def fill(parts: list[tuple[str, str]], values: dict[str, str]) -> str:
    return "".join(value if kind == "text" else values[value] for kind, value in parts)


def validate_prompt_format(prompt: dict[str, Any], has_values: bool) -> None:
    """Structural rules; raises TemplateError with a message naming the problem."""
    system = prompt["system"]
    if system is not None:
        system_fields = fields(parse_template(system, MESSAGE_FIELDS, "prompt.system"))
        if "item" in system_fields:
            raise TemplateError("prompt.system must not contain {item}; the item goes in prompt.user")
    user_fields = fields(parse_template(prompt["user"], MESSAGE_FIELDS, "prompt.user"))
    if user_fields.count("item") != 1:
        raise TemplateError("prompt.user must contain {item} exactly once")
    scale_fields = fields(parse_template(prompt["scale"], SCALE_FIELDS, "prompt.scale"))
    if scale_fields.count("codes") != 1:
        raise TemplateError("prompt.scale must contain {codes} exactly once")
    code_fields = fields(parse_template(prompt["code"], CODE_FIELDS, "prompt.code"))
    if "response" not in code_fields:
        raise TemplateError("prompt.code must contain {response}")
    if "value" in code_fields and not has_values:
        raise TemplateError("prompt.code uses {value}, but the categories have no values")


def format_value(value: float | None) -> str:
    """Numbers as in JSON canonical form: 1.0 -> "1", 0.5 -> "0.5", -2.5e-7 -> "-2.5e-7"."""
    return "" if value is None else _number(value)


def scale_text(prompt: dict[str, Any], categories: list[dict[str, Any]]) -> str:
    code_parts = parse_template(prompt["code"], CODE_FIELDS, "prompt.code")
    lines = [
        fill(code_parts, {"response": c["response"], "label": c["label"], "value": format_value(c["value"]), "id": c["id"]})
        for c in categories
    ]
    return fill(parse_template(prompt["scale"], SCALE_FIELDS, "prompt.scale"), {"codes": prompt["code_separator"].join(lines)})


def build_messages(task: dict[str, Any], item: str) -> list[dict[str, str]]:
    """Chat messages for one item: optional system message, example pairs, then the item.

    ``task`` is the canonical task dict (``Task.canonical()``).
    """
    prompt = task["prompt"]
    values = {
        "instructions": task["instructions"],
        "scale": scale_text(prompt, task["categories"]),
        "answer_instruction": task["answer_instruction"],
    }
    user_parts = parse_template(prompt["user"], MESSAGE_FIELDS, "prompt.user")
    messages: list[dict[str, str]] = []
    if prompt["system"] is not None:
        messages.append({"role": "system", "content": fill(parse_template(prompt["system"], MESSAGE_FIELDS, "prompt.system"), values)})
    responses = {c["id"]: c["response"] for c in task["categories"]}
    for example in task["examples"]:
        messages.append({"role": "user", "content": fill(user_parts, {**values, "item": example["item"]})})
        messages.append({"role": "assistant", "content": responses[example["category_id"]]})
    messages.append({"role": "user", "content": fill(user_parts, {**values, "item": item})})
    return messages


def prompt_warnings(task: dict[str, Any]) -> list[dict[str, str]]:
    """Legal but probably unintended prompt formats."""
    prompt = task["prompt"]
    used: set[str] = set()
    if prompt["system"] is not None:
        used.update(fields(parse_template(prompt["system"], MESSAGE_FIELDS, "prompt.system")))
    used.update(fields(parse_template(prompt["user"], MESSAGE_FIELDS, "prompt.user")))
    warnings = []
    if task["instructions"] and "instructions" not in used:
        warnings.append({"code": "unused_instructions", "message": "the task has instructions, but no prompt template uses {instructions}"})
    if not task["instructions"] and "instructions" in used:
        warnings.append({"code": "empty_instructions", "message": "the prompt uses {instructions}, but the instructions are empty"})
    if task["answer_instruction"] and "answer_instruction" not in used:
        warnings.append({"code": "unused_answer_instruction", "message": "the task has an answer_instruction, but no prompt template uses {answer_instruction}"})
    if "scale" not in used:
        warnings.append({
            "code": "scale_not_in_prompt",
            "message": "no prompt template uses {scale}, so the response codes are shown to the model only where you wrote them yourself",
        })
    return warnings
