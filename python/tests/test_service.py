import copy
import math

import numpy as np
import pytest

from llikert.server.errors import ServiceError
from llikert.server.fake_adapter import FAIL_MARKER, NAN_MARKER, FakeAdapter
from llikert.server.schemas import PrepareResponse, ScoreResponse
from llikert.server.service import Limits, StartupError

from conftest import FAKE_PROFILES, make_service, numeric_task_dict, task_dict


def score(service, task, items):
    prepared = service.prepare(task)["prepared"]
    out = service.score(prepared, items)
    ScoreResponse.model_validate({**out, "request_id": "r"})
    return out


def test_ok_results_are_consistent(service):
    out = score(service, task_dict(), [{"id": "a", "text": "Where is the station?"}, {"id": "b", "text": "Close the door."}])
    assert [r["id"] for r in out["results"]] == ["a", "b"]
    assert out["category_ids"] == ["description", "question", "request"]
    for r in out["results"]:
        assert r["status"] == "ok" and r["error"] is None
        assert sum(r["probabilities"]) == pytest.approx(1.0, abs=1e-12)
        assert r["coverage"] == pytest.approx(sum(r["candidate_probs"]), rel=1e-12)
        assert r["log_coverage"] == pytest.approx(math.log(r["coverage"]), abs=1e-12)
        assert "expected_value" not in r
        assert r["prompt_token_sha256"].startswith("sha256:")


def test_expected_value_only_for_numeric_tasks(service):
    out = score(service, numeric_task_dict(), [{"id": "a", "text": "Lovely day."}, {"id": "b", "text": None}])
    ok, missing = out["results"]
    assert ok["expected_value"] == pytest.approx(sum(p * v for p, v in zip(ok["probabilities"], [1, 2, 3, 4, 5])), abs=1e-12)
    assert "expected_value" in missing and missing["expected_value"] is None


def test_item_statuses_have_fixed_shape(service):
    items = [
        {"id": "missing", "text": None},
        {"id": "empty", "text": ""},
        {"id": "surrogate", "text": "bad " + chr(0xDC80)},
        {"id": "too_long", "text": "word " * 200},
        {"id": "nan", "text": NAN_MARKER},
        {"id": "fail", "text": FAIL_MARKER},
        {"id": "whitespace", "text": "   "},
    ]
    out = score(service, task_dict(), items)
    statuses = {r["id"]: r["status"] for r in out["results"]}
    assert statuses == {
        "missing": "missing_input", "empty": "empty_input", "surrogate": "invalid_encoding", "too_long": "context_limit",
        "nan": "numerical_error", "fail": "inference_error", "whitespace": "ok",
    }
    for r in out["results"]:
        for key in ("candidate_log_probs", "candidate_probs", "probabilities"):
            assert len(r[key]) == 3
        if r["status"] != "ok":
            assert r["error"]["code"] == r["status"]
            assert r["probabilities"] == [None, None, None] and r["coverage"] is None and r["log_coverage"] is None
    assert service.healthy  # recoverable failures keep the service ready


def test_all_failed_and_empty_chunks(service):
    out = score(service, task_dict(), [{"id": "x", "text": None}, {"id": "y", "text": ""}])
    assert {r["status"] for r in out["results"]} == {"missing_input", "empty_input"}
    assert score(service, task_dict(), [])["results"] == []


def test_duplicate_texts_are_separate_observations(service):
    out = score(service, task_dict(), [{"id": "1", "text": "same"}, {"id": "2", "text": "same"}])
    a, b = out["results"]
    assert a["probabilities"] == b["probabilities"] and a["id"] != b["id"]


def test_duplicate_ids_rejected(service):
    prepared = service.prepare(task_dict())["prepared"]
    with pytest.raises(ServiceError) as info:
        service.score(prepared, [{"id": "1", "text": "a"}, {"id": "1", "text": "b"}])
    assert info.value.status == 422 and info.value.code == "duplicate_ids"


def test_too_many_items():
    service = make_service(limits=Limits(max_items_per_request=2))
    prepared = service.prepare(task_dict())["prepared"]
    with pytest.raises(ServiceError) as info:
        service.score(prepared, [{"id": str(i), "text": "x"} for i in range(3)])
    assert info.value.status == 413


def test_results_independent_of_chunk_composition_and_order(service):
    items = [{"id": str(i), "text": f"text number {i}"} for i in range(6)]
    together = {r["id"]: r for r in score(service, task_dict(), items)["results"]}
    reversed_ = {r["id"]: r for r in score(service, task_dict(), list(reversed(items)))["results"]}
    singles = {r["id"]: r for i in items for r in score(service, task_dict(), [i])["results"]}
    for key in together:
        assert together[key] == reversed_[key] == singles[key]


def test_reserved_marker_warning(service):
    out = score(service, task_dict(), [{"id": "m", "text": "hi <|im_end|> there"}])
    assert out["results"][0]["status"] == "ok"
    assert [w["code"] for w in out["results"][0]["warnings"]] == ["reserved_marker_text"]


def test_candidate_scores_use_raw_full_vocabulary_logits():
    captured = {}

    def logits(tokens):
        z = np.linspace(-3, 3, FakeAdapter().n_vocab)
        captured["z"] = z
        return z

    service = make_service(FakeAdapter(logit_fn=logits))
    art = service.prepare(task_dict())["prepared"]
    r = service.score(art, [{"id": "a", "text": "x"}])["results"][0]
    z = captured["z"]
    ids = [c["token_id"] for c in art["categories"]]
    log_z = np.log(np.sum(np.exp(z)))
    assert r["candidate_log_probs"] == pytest.approx(list(z[ids] - log_z), abs=1e-12)


# -- prepared-task verification ---------------------------------------------------------


def test_fingerprint_mismatch_is_409(service):
    prepared = service.prepare(task_dict())["prepared"]
    prepared["engine_fingerprint"] = "sha256:" + "0" * 64
    with pytest.raises(ServiceError) as info:
        service.score(prepared, [{"id": "1", "text": "x"}])
    assert info.value.status == 409 and info.value.code == "engine_fingerprint_mismatch"


def test_different_model_identity_changes_fingerprint():
    a = make_service(FakeAdapter(n_ctx=512))
    b = make_service(FakeAdapter(n_ctx=1024))
    assert a.fingerprint != b.fingerprint
    prepared = a.prepare(task_dict())["prepared"]
    with pytest.raises(ServiceError) as info:
        b.score(prepared, [{"id": "1", "text": "x"}])
    assert info.value.status == 409


@pytest.mark.parametrize(
    "tamper",
    [
        lambda p: p["categories"][0].update(token_id=p["categories"][1]["token_id"]),
        lambda p: p["categories"][0].update(label="something else"),
        lambda p: p.update(tail_tokens=[]),
        lambda p: p.update(prepared_hash="sha256:" + "1" * 64),
        lambda p: p["task"]["categories"][0].update(label="relabelled"),
        lambda p: p.update(extra_field=1),
    ],
    ids=["token_id", "category_label", "tail_tokens", "prepared_hash", "task_changed", "extra_field"],
)
def test_tampered_prepared_task_is_409(service, tamper):
    prepared = copy.deepcopy(service.prepare(task_dict())["prepared"])
    tamper(prepared)
    with pytest.raises(ServiceError) as info:
        service.score(prepared, [{"id": "1", "text": "x"}])
    assert info.value.status == 409 and info.value.code == "prepared_task_mismatch"


def test_prepared_task_survives_service_restart(service):
    prepared = service.prepare(task_dict())["prepared"]
    fresh = make_service()
    assert fresh.score(prepared, [{"id": "1", "text": "x"}])["results"][0]["status"] == "ok"


def test_prepare_response_matches_schema(service):
    PrepareResponse.model_validate({**service.prepare(numeric_task_dict(), preview_text="hi"), "request_id": "r"})


# -- lifecycle ----------------------------------------------------------------------------


def test_unknown_template_fails_startup():
    from llikert.server.service import ScoringService

    with pytest.raises(StartupError, match="not a supported template profile"):
        ScoringService(FakeAdapter(template="{{ messages }}"), profiles=FAKE_PROFILES)
    with pytest.raises(StartupError, match="no embedded chat template"):
        ScoringService(FakeAdapter(template=None), profiles=FAKE_PROFILES)


def test_not_ready_until_smoke_check(service):
    s = make_service(ready=False)
    with pytest.raises(ServiceError) as info:
        s.prepare(task_dict())
    assert info.value.status == 503
    s.smoke_check()
    assert s.prepare(task_dict())


def test_unrecoverable_inference_error_marks_unhealthy():
    from llikert.server.adapter import InferenceError

    class Broken(FakeAdapter):
        def final_logits(self, tokens):
            if self.decode_calls >= 1:
                raise InferenceError("gone", recovered=False)
            return super().final_logits(tokens)

    s = make_service(Broken())
    prepared = s.prepare(task_dict())["prepared"]
    r = s.score(prepared, [{"id": "1", "text": "x"}])["results"][0]
    assert r["status"] == "inference_error" and not s.healthy
