import json

import pytest

from llikert import Category, Example, ScoringTask

from .mock_service import FIXTURES, load_fixture


def quickstart_nominal():
    return ScoringTask(
        name="Communicative function",
        instructions="Classify the text's primary communicative function.",
        categories=["description", "question", "request"],
        responses=["A", "B", "C"],
        ordered=False,
    )


def quickstart_numeric():
    return ScoringTask(
        name="Sentiment",
        instructions="Rate the overall sentiment of the text.",
        categories=["very negative", "negative", "neutral", "positive", "very positive"],
        responses=["1", "2", "3", "4", "5"],
        values=[1, 2, 3, 4, 5],
        ordered=True,
        examples=[("I love it.", "very positive")],
    )


def test_constructors_build_fixture_tasks():
    fixture = load_fixture("protocol/fake-service.json")["tasks"]
    assert quickstart_nominal().to_dict() == fixture["nominal"]["task"]
    assert quickstart_numeric().to_dict() == fixture["numeric"]["task"]


def test_records_and_explicit_ids():
    t = ScoringTask(name="n", instructions="i", categories=[Category("a", "Alpha", "A"), Category("b", "Beta", "B")])
    assert [c.id for c in t.categories] == ["a", "b"]
    t2 = ScoringTask(name="n", instructions="i", categories=["same", "same"], responses=["A", "B"], ids=["x", "y"])
    assert [c.id for c in t2.categories] == ["x", "y"]
    t3 = ScoringTask(name="n", instructions="i", categories=["a", "b"], responses=["A", "B"], examples=[Example("t", "a"), {"text": "u", "category_id": "b"}])
    assert len(t3.examples) == 2


@pytest.mark.parametrize(
    "kwargs,exc",
    [
        (dict(categories=["a"], responses=["A"]), ValueError),
        (dict(categories=["a", "a"], responses=["A", "B"]), ValueError),
        (dict(categories=["a", "b"], responses=["A", "A"]), ValueError),
        (dict(categories=["a", "b"], responses=["A"]), ValueError),
        (dict(categories=["a", "b"], responses=["A", "B"], values=[1, float("nan")]), ValueError),
        (dict(categories=["a", "b"], responses=["A", "B"], values=[1, "2"]), TypeError),
        (dict(categories=["a", "b"], responses=["A", ""]), ValueError),
        (dict(categories=["id", "b"], responses=["A", "B"]), ValueError),
        (dict(categories=["a", "b"], responses=["A", "B"], examples=[("x", "c")]), ValueError),
        (dict(categories=["a", "b"], responses=["A", "B"], ordered="yes"), TypeError),
        (dict(categories=["a", "b"]), ValueError),
    ],
)
def test_invalid_construction(kwargs, exc):
    with pytest.raises(exc):
        ScoringTask(name="n", instructions="i", **kwargs)


def test_numeric_looking_labels_are_not_values():
    t = ScoringTask(name="n", instructions="i", categories=["1", "2"], responses=["A", "B"])
    assert not t.has_values


def test_json_roundtrip_preserves_whitespace(tmp_path):
    t = ScoringTask(name="n ", instructions=" i\n", categories=["x ", " y"], responses=[" A", "B"], values=[-1.5, 2])
    t.to_json(tmp_path / "task.json")
    back = ScoringTask.from_json(tmp_path / "task.json")
    assert back == t and back.to_dict() == t.to_dict()


def test_shared_fixtures_agree_with_client_validation():
    for path in sorted((FIXTURES / "tasks" / "valid").glob("*.json")):
        case = json.loads(path.read_text())
        obj = dict(case["input"])
        obj.setdefault("ordered", False)
        obj.setdefault("examples", [])
        assert ScoringTask.from_dict(obj).to_dict() == case["canonical"], path.name
    for path in sorted((FIXTURES / "tasks" / "invalid").glob("*.json")):
        with pytest.raises((ValueError, TypeError, KeyError)):
            ScoringTask.from_dict(json.loads(path.read_text())["input"])
