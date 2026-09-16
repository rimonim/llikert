import json

import pytest

from llikert.server.canonical import canonicalize
from llikert.server.errors import ServiceError
from llikert.server.task import parse_task

from conftest import numeric_task_dict, task_dict


def error_of(obj):
    with pytest.raises(ServiceError) as info:
        parse_task(obj)
    assert info.value.status == 422 and info.value.code == "invalid_task"
    return info.value


def test_valid_nominal_and_numeric():
    assert not parse_task(task_dict()).has_values
    t = parse_task(numeric_task_dict())
    assert t.has_values and t.ordered


def test_defaults_are_filled_in_canonical_form():
    obj = task_dict()
    del obj["ordered"], obj["examples"]
    for c in obj["categories"]:
        del c["value"]
    canonical = parse_task(obj).canonical()
    assert canonical["ordered"] is False and canonical["examples"] == []
    assert all(c["value"] is None for c in canonical["categories"])


def test_identity_ignores_json_key_order_but_not_category_order():
    a = task_dict()
    b = json.loads(json.dumps(a, sort_keys=True))
    assert parse_task(a).identity() == parse_task(b).identity()
    c = task_dict(categories=list(reversed(a["categories"])))
    assert parse_task(a).identity() != parse_task(c).identity()


def test_whitespace_and_unicode_preserved():
    obj = task_dict()
    obj["categories"][0]["response"] = " A"
    obj["categories"][1]["label"] = "Frage ❓ "
    t = parse_task(obj)
    assert t.categories[0].response == " A" and t.categories[1].label == "Frage ❓ "


@pytest.mark.parametrize(
    "mutate",
    [
        lambda t: t.update(categories=t["categories"][:1]),
        lambda t: t["categories"][1].update(id="description"),
        lambda t: t["categories"][1].update(response="A"),
        lambda t: t["categories"][0].update(value=1.0),
        lambda t: t["categories"][0].update(response=""),
        lambda t: t["categories"][0].update(label=""),
        lambda t: t["categories"][0].update(id="expected_value"),
        lambda t: t["categories"][0].update(id="id"),
        lambda t: t.update(instructions=""),
        lambda t: t.update(schema_version=2),
        lambda t: t.update(unknown_field=True),
        lambda t: t["categories"][0].update(extra=1),
        lambda t: t.update(ordered="false"),
        lambda t: t.update(examples=[{"text": "x", "category_id": "nope"}]),
        lambda t: t.update(examples=[{"text": "", "category_id": "question"}]),
        lambda t: t["categories"][0].update(label=chr(0xD800)),
    ],
    ids=[
        "one_category", "duplicate_id", "duplicate_response", "partial_values", "empty_response", "empty_label",
        "reserved_expected_value", "reserved_id", "empty_instructions", "schema_version", "unknown_field",
        "unknown_category_field", "string_bool", "example_unknown_category", "example_empty_text", "lone_surrogate",
    ],
)
def test_invalid_tasks(mutate):
    obj = task_dict()
    mutate(obj)
    error_of(obj)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "1", True])
def test_values_must_be_finite_numbers(bad):
    obj = numeric_task_dict()
    obj["categories"][0]["value"] = bad
    error_of(obj)


def test_negative_noninteger_values_accepted():
    obj = numeric_task_dict()
    for c, v in zip(obj["categories"], [-2.5, -1, 0, 0.125, 1e3]):
        c["value"] = v
    assert [c.value for c in parse_task(obj).categories] == [-2.5, -1.0, 0.0, 0.125, 1000.0]


def test_validation_errors_do_not_echo_input():
    obj = task_dict()
    obj["categories"][0]["label"] = 12345.678
    err = error_of(obj)
    assert "12345.678" not in json.dumps(err.details)


def test_shared_task_fixtures(fixtures):
    for path in sorted((fixtures / "tasks" / "valid").glob("*.json")):
        case = json.loads(path.read_text())
        task = parse_task(case["input"])
        assert task.canonical() == case["canonical"], path.name
        assert canonicalize(task.canonical()) == case["canonical_jcs"], path.name
        assert task.identity() == case["task_hash"], path.name
    for path in sorted((fixtures / "tasks" / "invalid").glob("*.json")):
        case = json.loads(path.read_text())
        with pytest.raises(ServiceError) as info:
            parse_task(case["input"])
        assert info.value.code == case["error_code"], path.name
