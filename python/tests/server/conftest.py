import pathlib

import pytest

from llikert.server.fake_adapter import FAKE_CHAT_TEMPLATE, FakeAdapter
from llikert.server.profiles import TemplateProfile, template_sha256
from llikert.server.service import Limits, ScoringService

REPO = pathlib.Path(__file__).resolve().parents[3]
FAKE_SHA = template_sha256(FAKE_CHAT_TEMPLATE)
FAKE_PROFILES = {FAKE_SHA: TemplateProfile("fake-chatml", FAKE_SHA)}


def make_service(adapter=None, limits: Limits = Limits(), ready: bool = True) -> ScoringService:
    service = ScoringService(adapter or FakeAdapter(), limits, profiles=FAKE_PROFILES)
    if ready:
        service.smoke_check()
    return service


@pytest.fixture
def service() -> ScoringService:
    return make_service()


def task_dict(**overrides):
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


def numeric_task_dict(**overrides):
    task = task_dict(
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
