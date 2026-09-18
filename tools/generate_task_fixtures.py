"""Generate the shared task and prompt fixtures from the service's own validation and renderer.

    .venv/bin/python tools/generate_task_fixtures.py

tests/fixtures/tasks/{valid,invalid}/*.json   canonical form, JCS text, and hash / expected error
tests/fixtures/prompts/messages.json          chat messages for tasks x items (client previews must match)
"""

from __future__ import annotations

import copy
import json
import pathlib
import shutil
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python" / "src"))

from llikert.server.canonical import canonicalize  # noqa: E402
from llikert.server.errors import ServiceError  # noqa: E402
from llikert.server.prompt import build_messages, prompt_warnings  # noqa: E402
from llikert.server.task import parse_task  # noqa: E402

FIXTURES = REPO / "tests" / "fixtures"


def nominal(**overrides):
    task = {
        "schema_version": 1,
        "name": "Communicative function",
        "instructions": "Classify the text's primary communicative function.",
        "categories": [
            {"id": "description", "label": "description", "response": "A", "value": None},
            {"id": "question", "label": "question", "response": "B", "value": None},
            {"id": "request", "label": "request", "response": "C", "value": None},
        ],
        "ordered": False,
        "examples": [],
    }
    task.update(overrides)
    return task


def numeric(**overrides):
    task = nominal(
        name="Sentiment",
        instructions="Rate the sentiment.",
        categories=[
            {"id": f"s{i}", "label": label, "response": str(i), "value": float(i)}
            for i, label in enumerate(["very negative", "negative", "neutral", "positive", "very positive"], start=1)
        ],
        ordered=True,
    )
    task.update(overrides)
    return task


def with_changes(task, change):
    task = copy.deepcopy(task)
    change(task)
    return task


VALID = {
    "nominal-minimal-defaults": with_changes(nominal(), lambda t: [t.pop("ordered"), t.pop("examples"), [c.pop("value") for c in t["categories"]]]),
    "nominal-with-examples": nominal(examples=[{"item": "Where is it?", "category_id": "question"}, {"item": "Close the door.", "category_id": "request"}]),
    "numeric-ordered": numeric(),
    "numeric-noninteger-negative": with_changes(numeric(), lambda t: [c.update(value=v) for c, v in zip(t["categories"], [-2.5, -1, 0, 0.1, 1e21])]),
    "unicode-whitespace-strings": with_changes(nominal(), lambda t: [t["categories"][0].update(label="Beschreibung ✎ ", response=" A"), t.update(name="Ünïcode  name", instructions="Line one\nLine two\ttab")]),
    "long-labels-letter-codes": nominal(categories=[{"id": f"c{i}", "label": "category number %d with a long descriptive name" % i, "response": chr(65 + i), "value": float(10 * (i + 1))} for i in range(12)]),
    "prompt-instructions-after-item": nominal(prompt={"system": None, "user": "{item}\n\n{instructions}\n{scale}\n{answer_instruction}"}),
    "prompt-questionnaire-minimal": numeric(
        name="Questionnaire",
        instructions="You are completing a personality questionnaire. Rate how well each statement describes you.",
        answer_instruction="Reply with the number only.",
        prompt={"system": "{instructions}\n{scale}", "user": "{item}\n\n{answer_instruction}", "scale": "{codes}",
                "code": "{response} = {label}", "code_separator": "; "},
    ),
    "prompt-values-ids-braces": with_changes(
        numeric(prompt={"code": "{response}: {label} [{value}|{id}]", "scale": "Scale {{1-5}}:\n{codes}", "user": "{{Item}} {item}"}),
        lambda t: [c.update(value=v) for c, v in zip(t["categories"], [-2.5, 0, 0.1, 1e21, 1 / 3])],
    ),
    "prompt-empty-instructions": nominal(instructions="", prompt={"system": "{scale}\n{answer_instruction}", "user": "{item}"}),
    "prompt-answer-instruction-empty": nominal(answer_instruction="", prompt={"system": "{instructions}\n\n{scale}", "user": "{item}"}),
}

INVALID = {
    "one-category": with_changes(nominal(), lambda t: t.update(categories=t["categories"][:1])),
    "duplicate-category-id": with_changes(nominal(), lambda t: t["categories"][1].update(id="description")),
    "duplicate-response": with_changes(nominal(), lambda t: t["categories"][1].update(response="A")),
    "partial-values": with_changes(nominal(), lambda t: t["categories"][0].update(value=1)),
    "empty-response": with_changes(nominal(), lambda t: t["categories"][0].update(response="")),
    "reserved-category-id": with_changes(nominal(), lambda t: t["categories"][0].update(id="expected_value")),
    "unknown-field": with_changes(nominal(), lambda t: t.update(color="blue")),
    "string-value": with_changes(nominal(), lambda t: [c.update(value=str(i)) for i, c in enumerate(t["categories"])]),
    "example-unknown-category": nominal(examples=[{"item": "x", "category_id": "nope"}]),
    "missing-instructions": with_changes(nominal(), lambda t: t.pop("instructions")),
    "prompt-user-without-item": nominal(prompt={"user": "Classify this."}),
    "prompt-item-in-system": nominal(prompt={"system": "{item}"}),
    "prompt-unknown-placeholder": nominal(prompt={"user": "{item} {text}"}),
    "prompt-unmatched-brace": nominal(prompt={"user": "{item} }"}),
    "prompt-scale-without-codes": nominal(prompt={"scale": "Codes below"}),
    "prompt-code-without-response": nominal(prompt={"code": "{label}"}),
    "prompt-value-without-values": nominal(prompt={"code": "{response} = {value}"}),
    "prompt-unknown-field": nominal(prompt={"assistant": "x"}),
    "prompt-answer-instruction-in-prompt": nominal(prompt={"answer_instruction": "Answer with one code."}),
}

ITEMS = ["Where is the station?", " leading space and trailing newline\n", "Grüße aus Köln — naïve café ✓", "Literal {scale} and {item} and {{braces}}"]


def main() -> None:
    for sub in ("valid", "invalid"):
        shutil.rmtree(FIXTURES / "tasks" / sub, ignore_errors=True)
        (FIXTURES / "tasks" / sub).mkdir(parents=True)
    cases = []
    for name, obj in VALID.items():
        task = parse_task(obj)
        canonical = task.canonical()
        case = {"input": obj, "canonical": canonical, "canonical_jcs": canonicalize(canonical), "task_hash": task.identity()}
        (FIXTURES / "tasks" / "valid" / f"{name}.json").write_text(json.dumps(case, indent=2, ensure_ascii=False) + "\n")
        cases.append({"task": name, "warnings": [w["code"] for w in prompt_warnings(canonical)],
                      "items": [{"item": item, "messages": build_messages(canonical, item)} for item in ITEMS]})
    for name, obj in INVALID.items():
        try:
            parse_task(obj)
        except ServiceError as exc:
            (FIXTURES / "tasks" / "invalid" / f"{name}.json").write_text(json.dumps({"input": obj, "error_code": exc.code}, indent=2) + "\n")
        else:
            raise SystemExit(f"{name} unexpectedly valid")
    (FIXTURES / "prompts" / "messages.json").write_text(json.dumps(
        {"description": "Chat messages built by the service for each valid task fixture and item; client previews must match exactly.",
         "cases": cases}, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(VALID)} valid tasks, {len(INVALID)} invalid, {len(cases) * len(ITEMS)} message cases")


if __name__ == "__main__":
    main()
