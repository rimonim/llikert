import pytest

from llikert.server.canonical import identity_hash
from llikert.server.errors import ServiceError

from conftest import numeric_task_dict, task_dict


def prepare_error(service, task):
    with pytest.raises(ServiceError) as info:
        service.prepare(task)
    return info.value


def test_prepare_success_artifact(service):
    out = service.prepare(task_dict())
    art = out["prepared"]
    assert [c["token_piece"] for c in art["categories"]] == ["A", "B", "C"]
    assert len({c["token_id"] for c in art["categories"]}) == 3
    assert art["engine_fingerprint"] == service.fingerprint
    assert art["prepared_hash"] == identity_hash({k: v for k, v in art.items() if k != "prepared_hash"})
    assert art["task_hash"] == identity_hash(art["task"])
    assert out["preview"]["example"]["prompt"].endswith("<|im_start|>assistant\n")
    assert "text" not in out["preview"]


def test_prepare_is_deterministic_across_service_instances(service):
    from conftest import make_service

    assert service.prepare(task_dict())["prepared"] == make_service().prepare(task_dict())["prepared"]


def test_preview_text_only_on_request(service):
    out = service.prepare(task_dict(), preview_text="Where is the station?")
    assert "Where is the station?" in out["preview"]["text"]["prompt"]
    assert "Where is the station?" not in str(out["prepared"])


def test_multi_word_single_tokens_allowed(service):
    t = task_dict()
    for c, r in zip(t["categories"], ["Yes", "No", "very"]):
        c["response"] = r
    assert [c["token_piece"] for c in service.prepare(t)["prepared"]["categories"]] == ["Yes", "No", "very"]


def problems_of(err):
    return {p["category_id"]: p["problem"] for p in err.details["problems"]}


def test_multi_token_code_rejected_with_suggestions(service):
    t = task_dict()
    for c in t["categories"]:
        c["response"] = c["label"]  # description/question/request are multi-token in the fake vocab
    err = prepare_error(service, t)
    assert err.code == "invalid_response_codes" and err.status == 422
    assert set(problems_of(err).values()) == {"multi_token"}
    sugg = err.details["suggestions"]
    per = {e["category_id"]: e for e in sugg["prefix"]["per_category"]}
    assert per["description"]["from_response"] == "desc"
    assert per["question"]["from_response"] == "quest"
    assert per["request"]["from_response"] == "req"
    assert sugg["prefix"]["complete"] == {"source": "from_response", "responses": ["desc", "quest", "req"]}
    assert {"kind": "letters", "responses": ["A", "B", "C"]} in sugg["generic"]
    assert {"kind": "digits", "responses": ["1", "2", "3"]} in sugg["generic"]


def test_prefix_collision_flagged(service):
    t = task_dict()
    t["categories"][0].update(id="q1", label="question", response="question")
    t["categories"][1].update(id="q2", label="questionable", response="questionable")
    err = prepare_error(service, t)
    per = {e["category_id"]: e for e in err.details["suggestions"]["prefix"]["per_category"]}
    assert per["q1"]["from_response"] == per["q2"]["from_response"] == "quest"
    assert per["q1"]["collision"] and per["q2"]["collision"]
    assert not per["request"]["collision"]
    assert err.details["suggestions"]["prefix"]["complete"] is None


def test_digits_over_nine_are_multi_token(service):
    t = numeric_task_dict()
    t["categories"][4]["response"] = "10"
    assert problems_of(prepare_error(service, t)) == {"s5": "multi_token"}


def test_boundary_merge_makes_prefix_unstable(service):
    t = task_dict()
    t["categories"][2]["response"] = "Z"  # the fake vocab merges "\nZ" across the boundary
    assert problems_of(prepare_error(service, t)) == {"request": "prefix_unstable"}


def test_control_token_code_rejected(service):
    t = task_dict()
    t["categories"][2]["response"] = "<|im_end|>"
    assert problems_of(prepare_error(service, t)) == {"request": "end_of_generation_token"}


def test_non_ascii_code_needing_byte_tokens_rejected(service):
    t = task_dict()
    t["categories"][2]["response"] = "é"
    assert problems_of(prepare_error(service, t))["request"] in {"multi_token", "byte_token"}


def test_task_exceeding_context_rejected():
    from llikert.server.fake_adapter import FakeAdapter

    from conftest import make_service

    service = make_service(FakeAdapter(n_ctx=300))
    err = prepare_error(service, task_dict(instructions="word " * 400))
    assert err.code == "task_exceeds_context"


def test_whitespace_response_warning(service):
    t = task_dict()
    t["categories"][0]["response"] = " A"
    out = service.prepare(t)
    assert out["prepared"]["categories"][0]["token_piece"] == " A"
    assert [w["code"] for w in out["diagnostics"]["warnings"]] == ["response_whitespace"]
