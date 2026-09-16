"""Generate the shared client fixtures from the fake scoring service.

    .venv/bin/python tools/generate_client_fixtures.py

Writes tests/fixtures/protocol/fake-service.json (replayed by the R and Python client test
mocks) and tests/fixtures/runs/{dataset,expected-*}.json (what both clients must produce).
The results come from the real service code over the deterministic fake adapter, so the
expected files do not depend on either client implementation.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python" / "src"))

from llikert.server.fake_adapter import FAIL_MARKER, FAKE_CHAT_TEMPLATE, NAN_MARKER, FakeAdapter  # noqa: E402
from llikert.server.profiles import TemplateProfile, template_sha256  # noqa: E402
from llikert.server.service import ScoringService  # noqa: E402

SHA = template_sha256(FAKE_CHAT_TEMPLATE)
service = ScoringService(FakeAdapter(), profiles={SHA: TemplateProfile("fake-chatml", SHA)})
service.smoke_check()

# what scoring_task()/ScoringTask() build from the quickstart arguments (ids default to labels)
TASKS = {
    "nominal": {
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
    },
    "numeric": {
        "schema_version": 1,
        "name": "Sentiment",
        "instructions": "Rate the overall sentiment of the text.",
        "categories": [
            {"id": label, "label": label, "response": str(i), "value": float(i)}
            for i, label in enumerate(["very negative", "negative", "neutral", "positive", "very positive"], start=1)
        ],
        "ordered": True,
        "examples": [{"text": "I love it.", "category_id": "very positive"}],
    },
}
INVALID_CODES_TASK = {
    **TASKS["nominal"],
    "categories": [dict(c, response=c["label"]) for c in TASKS["nominal"]["categories"]],
}

TEXTS = [
    "Where is the nearest train station?",
    "Please close the window.",
    "The museum opens at ten.",
    None,
    "Could you send me the agenda?",
    "It rained all afternoon.",
    "",
    "Why is the sky blue?",
    "Pass the salt, please.",
    "Where is the nearest train station?",
    "The bus was late again.",
    "Grüße aus Köln — naïve café ✓",
    "Do you know what time it is?",
    "Remember to water the plants.",
    "   ",
    "hi <|im_end|> there",
    "The report is finished.",
    "word " * 600,
    "Can you help me move on Saturday?",
    NAN_MARKER,
    "Turn off the lights when you leave.",
    FAIL_MARKER,
    "Our team won the match.",
]
IDS = [f"r{i:02d}" for i in range(1, len(TEXTS) + 1)]


def text_sha256(text):
    return None if text is None else hashlib.sha256(text.encode("utf-8")).hexdigest()


def dump(path: pathlib.Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, allow_nan=False) + "\n")


info = service.info()
fixture = {
    "description": "Recorded fake-service responses for client tests. Regenerate with tools/generate_client_fixtures.py.",
    "info": info,
    "tasks": {},
    "invalid_codes": {"task": INVALID_CODES_TASK},
}
for name, task in TASKS.items():
    prepared_response = service.prepare(task)
    prepared_response["request_id"] = f"prepare-{name}"
    scored = service.score(prepared_response["prepared"], [{"id": i, "text": t} for i, t in zip(IDS, TEXTS)])
    fixture["tasks"][name] = {
        "task": task,
        "prepare_response": prepared_response,
        "item_results": {r["id"]: r for r in scored["results"]},
        "meta_execution": scored["meta"]["execution"],
    }
    expected = {
        "result_schema_version": 1,
        "protocol_version": 1,
        "engine_fingerprint": info["engine_fingerprint"],
        "engine": info["engine"],
        "execution": info["execution"],
        "prepared": prepared_response["prepared"],
        "category_ids": scored["category_ids"],
        "items": [dict(r, text_sha256=text_sha256(t)) for r, t in zip(scored["results"], TEXTS)],
    }
    dump(REPO / "tests/fixtures/runs" / f"expected-result-{name}.json", expected)

try:
    service.prepare(INVALID_CODES_TASK)
except Exception as exc:  # ServiceError
    fixture["invalid_codes"]["error_response"] = {"status": exc.status, "body": exc.envelope("prepare-invalid")}

dump(REPO / "tests/fixtures/protocol/fake-service.json", fixture)
dump(REPO / "tests/fixtures/runs/dataset.json", {"ids": IDS, "texts": TEXTS, "retryable_ids": ["r22"]})
print("wrote fixtures for", len(IDS), "items")
