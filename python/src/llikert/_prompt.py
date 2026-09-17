"""Client-side copy of the prompt-format rules, for validation and offline previews.

The service builds prompts authoritatively; tests/fixtures/prompts/messages.json keeps this
implementation (and the R client's) identical to the service's.
"""

from __future__ import annotations

import math
import re
from decimal import Decimal
from typing import Any

MESSAGE_FIELDS = {"instructions", "scale", "answer_instruction", "item"}
SCALE_FIELDS = {"codes"}
CODE_FIELDS = {"response", "label", "value", "id"}
_TOKEN = re.compile(r"\{\{|\}\}|\{([^{}]*)\}|\{|\}")


def parse_template(template: str, allowed: set[str], where: str) -> list[tuple[str, str]]:
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
            if m.group(1) not in allowed:
                allowed_list = ", ".join("{" + a + "}" for a in sorted(allowed))
                raise ValueError(f"{where} uses unknown placeholder {{{m.group(1)}}}; allowed: {allowed_list} (write {{{{ and }}}} for literal braces)")
            parts.append(("field", m.group(1)))
        else:
            raise ValueError(f"{where} has an unmatched brace; write {{{{ or }}}} for a literal brace")
        pos = m.end()
    if pos < len(template):
        parts.append(("text", template[pos:]))
    return parts


def fields(parts: list[tuple[str, str]]) -> list[str]:
    return [v for k, v in parts if k == "field"]


def fill(parts: list[tuple[str, str]], values: dict[str, str]) -> str:
    return "".join(v if k == "text" else values[v] for k, v in parts)


def validate(prompt: dict[str, Any], has_values: bool) -> None:
    if prompt["system"] is not None and "item" in fields(parse_template(prompt["system"], MESSAGE_FIELDS, "prompt.system")):
        raise ValueError("prompt.system must not contain {item}; the item goes in prompt.user")
    if fields(parse_template(prompt["user"], MESSAGE_FIELDS, "prompt.user")).count("item") != 1:
        raise ValueError("prompt.user must contain {item} exactly once")
    if fields(parse_template(prompt["scale"], SCALE_FIELDS, "prompt.scale")).count("codes") != 1:
        raise ValueError("prompt.scale must contain {codes} exactly once")
    code_fields = fields(parse_template(prompt["code"], CODE_FIELDS, "prompt.code"))
    if "response" not in code_fields:
        raise ValueError("prompt.code must contain {response}")
    if "value" in code_fields and not has_values:
        raise ValueError("prompt.code uses {value}, but the categories have no values")


def format_number(x: float) -> str:
    """ECMAScript/JSON-canonical number text: 1.0 -> "1", 1e21 -> "1e+21"."""
    x = float(x)
    if not math.isfinite(x):
        raise ValueError("values must be finite")
    if x == 0:
        return "0"
    sign = "-" if x < 0 else ""
    _, digit_tuple, exponent = Decimal(repr(abs(x))).as_tuple()
    digits = "".join(map(str, digit_tuple)).rstrip("0")
    n, k = len(digit_tuple) + exponent, len(digits)
    if k <= n <= 21:
        return sign + digits + "0" * (n - k)
    if 0 < n <= 21:
        return sign + digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return sign + "0." + "0" * (-n) + digits
    e = n - 1
    return f"{sign}{digits[0]}{'.' + digits[1:] if k > 1 else ''}e{'+' if e >= 0 else '-'}{abs(e)}"


def build_messages(task: dict[str, Any], item: str) -> list[dict[str, str]]:
    prompt = task["prompt"]
    code_parts = parse_template(prompt["code"], CODE_FIELDS, "prompt.code")
    lines = [
        fill(code_parts, {"response": c["response"], "label": c["label"], "id": c["id"],
                          "value": "" if c["value"] is None else format_number(c["value"])})
        for c in task["categories"]
    ]
    values = {
        "instructions": task["instructions"],
        "scale": fill(parse_template(prompt["scale"], SCALE_FIELDS, "prompt.scale"), {"codes": prompt["code_separator"].join(lines)}),
        "answer_instruction": prompt["answer_instruction"],
    }
    user_parts = parse_template(prompt["user"], MESSAGE_FIELDS, "prompt.user")
    messages = []
    if prompt["system"] is not None:
        messages.append({"role": "system", "content": fill(parse_template(prompt["system"], MESSAGE_FIELDS, "prompt.system"), values)})
    responses = {c["id"]: c["response"] for c in task["categories"]}
    for example in task["examples"]:
        messages.append({"role": "user", "content": fill(user_parts, {**values, "item": example["item"]})})
        messages.append({"role": "assistant", "content": responses[example["category_id"]]})
    messages.append({"role": "user", "content": fill(user_parts, {**values, "item": item})})
    return messages
