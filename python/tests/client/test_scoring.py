import copy
import json
import math

import httpx
import pytest

from llikert import (
    AuthenticationError, CheckpointError, FingerprintMismatch, PreparedTask, PrepareError, ProtocolError, RetryPolicy,
    Scorer, ScoreResult, ServiceError, ServiceUnavailable, TransportError, read_checkpoint,
)
from llikert import checkpoint as checkpoint_module

from .mock_service import HEADERS, error, load_fixture
from .test_task import quickstart_nominal, quickstart_numeric

VOLATILE = {"created_at", "client"}


def normalized(result: ScoreResult) -> dict:
    return {k: v for k, v in result.to_dict().items() if k not in VOLATILE}


def expected(name: str) -> dict:
    return load_fixture(f"runs/expected-result-{name}.json")


# -- connection -------------------------------------------------------------------------------


def test_connect_reads_info(connect, mock):
    with connect() as scorer:
        assert scorer.engine_fingerprint == mock.fixture["info"]["engine_fingerprint"]
        assert scorer.limits["max_items_per_request"] == 64


def test_connect_waits_through_cold_start(connect, mock, sleeper):
    mock.inject["/v1/info"].extend([error(503, "starting"), httpx.ConnectError("refused"), error(503, "starting")])
    scorer = connect(wait=600)
    assert scorer.info["protocol_version"] == 1
    assert len(sleeper.calls) == 3


def test_connect_gives_up_after_wait(mock, sleeper):
    clock = iter(range(0, 10_000, 100))
    mock.inject["/v1/info"].extend([error(503, "starting")] * 50)
    with pytest.raises(ServiceUnavailable):
        Scorer.connect("http://mock", wait=250, retry=RetryPolicy(sleep=sleeper), _client=mock.client(), _clock=lambda: next(clock))


def test_auth_failure_is_not_retried(mock, sleeper):
    mock.token = "secret-token-value"
    with pytest.raises(AuthenticationError):
        Scorer.connect("http://mock", token="wrong", retry=RetryPolicy(sleep=sleeper), _client=mock.client())
    assert sleeper.calls == []
    assert Scorer.connect("http://mock", token="secret-token-value", _client=mock.client()).info


def test_token_from_environment_and_not_in_repr(mock, monkeypatch):
    mock.token = "env-token-0123456789"
    monkeypatch.setenv("LLIKERT_TOKEN", "env-token-0123456789")
    scorer = Scorer.connect("http://mock", _client=mock.client())
    assert "env-token" not in repr(scorer)


def test_protocol_version_mismatch(connect, mock):
    mock.fixture["info"]["protocol_version"] = 2
    with pytest.raises(ProtocolError):
        connect()


def test_redirects_are_not_followed(connect, mock):
    scorer = connect()
    mock.inject["/v1/prepare"].append(httpx.Response(307, headers={"Location": "http://elsewhere/v1/prepare"}))
    with pytest.raises(ServiceError) as info:
        scorer.prepare(quickstart_nominal())
    assert info.value.status == 307
    assert all(path != "http://elsewhere/v1/prepare" for path, _ in mock.calls)


# -- preparation ---------------------------------------------------------------------------------


def test_prepare_returns_portable_prepared_task(connect, tmp_path):
    scorer = connect()
    prepared = scorer.prepare(quickstart_nominal(), preview_item="Where?")
    assert [(c["id"], c["response"]) for c in prepared.mapping] == [("description", "A"), ("question", "B"), ("request", "C")]
    assert "Where?" in prepared.preview_prompt("item")
    assert "request" in repr(prepared)
    prepared.save(tmp_path / "prepared.json")
    loaded = PreparedTask.load(tmp_path / "prepared.json")
    assert loaded.artifact == prepared.artifact and loaded.task == quickstart_nominal()


def test_invalid_codes_raise_prepare_error_with_suggestions(connect, mock):
    from llikert import ScoringTask

    scorer = connect()
    task = ScoringTask.from_dict(mock.fixture["invalid_codes"]["task"])
    with pytest.raises(PrepareError) as info:
        scorer.prepare(task)
    assert info.value.code == "invalid_response_codes"
    assert info.value.suggestions["prefix"]["complete"]["responses"] == ["desc", "quest", "req"]
    assert {p["problem"] for p in info.value.problems} == {"multi_token"}


# -- scoring -------------------------------------------------------------------------------------


@pytest.mark.parametrize("name,task_fn", [("nominal", quickstart_nominal), ("numeric", quickstart_numeric)])
@pytest.mark.parametrize("chunk_size", [1, 5, 64])
def test_full_run_matches_shared_expected_result(connect, dataset, name, task_fn, chunk_size):
    scorer = connect()
    result = scorer.score(dataset["texts"], dataset["ids"], task=scorer.prepare(task_fn()), chunk_size=chunk_size, progress=False)
    assert normalized(result) == expected(name)


def test_result_tables_have_fixed_shape(connect, dataset):
    scorer = connect()
    result = scorer.score(dataset["texts"], dataset["ids"], task=scorer.prepare(quickstart_nominal()), progress=False)
    assert result.ids == dataset["ids"]
    assert all(len(row) == 3 for row in result.probabilities + result.candidate_probs + result.candidate_log_probs)
    assert result.expected_values is None
    diag = {d["id"]: d for d in result.diagnostics}
    assert diag["r04"]["status"] == "missing_input" and diag["r07"]["status"] == "empty_input"
    assert diag["r16"]["warnings"] == ["reserved_marker_text"]
    assert "not ok" not in repr(result) and "18 ok" in repr(result)
    records = result.to_records()
    assert list(records[0]) == ["id", "description", "question", "request"]
    numeric = scorer.score(dataset["texts"], dataset["ids"], task=scorer.prepare(quickstart_numeric()), progress=False)
    assert list(numeric.to_records()[0])[:2] == ["id", "expected_value"]
    ok = [v for v, d in zip(numeric.expected_values, numeric.diagnostics) if d["status"] == "ok"]
    assert all(1 <= v <= 5 for v in ok)


def test_missing_and_empty_texts_are_not_sent(connect, mock, dataset):
    scorer = connect()
    scorer.score(dataset["texts"], dataset["ids"], task=scorer.prepare(quickstart_nominal()), chunk_size=5, progress=False)
    sent = [i for call in mock.score_calls for i in call]
    assert "r04" not in sent and "r07" not in sent and len(sent) == 21
    assert all(len(call) <= 5 for call in mock.score_calls)


def test_zero_and_single_rows(connect, mock):
    scorer = connect()
    prepared = scorer.prepare(quickstart_nominal())
    empty = scorer.score([], task=prepared, progress=False)
    assert empty.items == [] and mock.score_calls == []
    one = scorer.score(["Where is the nearest train station?"], ["r01"], task=prepared, progress=False)
    assert one.ids == ["r01"] and one.items[0]["status"] == "ok"
    all_failed = scorer.score([None, ""], ["r04", "r07"], task=prepared, progress=False)
    assert [d["status"] for d in all_failed.diagnostics] == ["missing_input", "empty_input"]
    assert all(row == [None, None, None] for row in all_failed.probabilities)


def test_ids_and_texts_validation(connect):
    scorer = connect()
    prepared = scorer.prepare(quickstart_nominal())
    with pytest.raises(ValueError):
        scorer.score(["a", "b"], ["x", "x"], task=prepared, progress=False)
    with pytest.raises(TypeError):
        scorer.score(["a", 5], task=prepared, progress=False)
    with pytest.raises(TypeError):
        scorer.score("a single string", task=prepared, progress=False)
    with pytest.raises(TypeError):
        scorer.score(["a"], [1.5], task=prepared, progress=False)
    with pytest.raises(TypeError):
        scorer.score(["a"], [float(2**60)], task=prepared, progress=False)
    with pytest.raises(TypeError, match="prepare"):
        scorer.score(["a"], task=quickstart_nominal(), progress=False)


def test_default_ids_and_nan_missing(connect, mock):
    scorer = connect()
    prepared = scorer.prepare(quickstart_nominal())
    mock.fixture["tasks"]["nominal"]["item_results"]["1"] = dict(mock.fixture["tasks"]["nominal"]["item_results"]["r01"], id="1")
    result = scorer.score(["Where is the nearest train station?", float("nan")], task=prepared, progress=False)
    assert result.ids == ["1", "2"] and result.diagnostics[1]["status"] == "missing_input"


def test_client_side_fingerprint_check_before_any_request(connect, mock):
    scorer = connect()
    prepared = scorer.prepare(quickstart_nominal())
    stale = PreparedTask(dict(prepared.artifact, engine_fingerprint="sha256:" + "0" * 64))
    with pytest.raises(FingerprintMismatch):
        scorer.score(["x"], task=stale, progress=False)
    assert mock.score_calls == []


def test_progress_callback(connect, dataset):
    scorer = connect()
    seen = []
    scorer.score(dataset["texts"], dataset["ids"], task=scorer.prepare(quickstart_nominal()), chunk_size=10,
                 progress=lambda done, total, failed: seen.append((done, total)))
    assert seen[0][0] == 2 and seen[-1] == (23, 23)


def test_large_run_without_checkpoint_hints(connect, mock):
    scorer = connect()
    with pytest.warns(UserWarning, match="checkpoint"):
        scorer.score([None] * 201, task=scorer.prepare(quickstart_nominal()), progress=False)


# -- retries -------------------------------------------------------------------------------------


def test_transient_errors_are_retried_with_retry_after(connect, mock, sleeper, dataset):
    scorer = connect()
    prepared = scorer.prepare(quickstart_nominal())
    mock.inject["/v1/score"].extend([
        httpx.Response(502, text="<html><body><h1>502 Bad Gateway</h1></body></html>"),
        error(429, "queue_full", headers={"Retry-After": "7"}),
        httpx.ReadTimeout("slow"),
    ])
    result = scorer.score(dataset["texts"], dataset["ids"], task=prepared, progress=False)
    assert normalized(result) == expected("nominal")
    assert 7 in sleeper.calls and len(sleeper.calls) == 3


def test_non_json_error_is_summarized(connect, mock):
    scorer = connect(retry=RetryPolicy(max_tries=1))
    mock.inject["/v1/score"].append(httpx.Response(504, text="<html><title>Gateway Timeout</title><script>x</script></html>"))
    with pytest.raises(ServiceError) as info:
        scorer.score(["a"], ["r01"], task=scorer.prepare(quickstart_nominal()), progress=False)
    assert info.value.status == 504 and info.value.code == "http_error" and "<" not in info.value.message


@pytest.mark.parametrize("status,code,exc", [(400, "invalid_json", ServiceError), (409, "prepared_task_mismatch", FingerprintMismatch), (422, "invalid_request", ServiceError)])
def test_permanent_errors_are_not_retried(connect, mock, sleeper, status, code, exc):
    scorer = connect()
    mock.inject["/v1/score"].append(error(status, code))
    with pytest.raises(exc):
        scorer.score(["a"], ["r01"], task=scorer.prepare(quickstart_nominal()), progress=False)
    assert sleeper.calls == [] and len(mock.score_calls) == 1


def test_bounded_transport_retries(connect, mock, sleeper):
    scorer = connect()
    mock.inject["/v1/score"].extend([httpx.ConnectError("down")] * 6)
    with pytest.raises(TransportError):
        scorer.score(["a"], ["r01"], task=scorer.prepare(quickstart_nominal()), progress=False)
    assert len(sleeper.calls) == 5 and all(0 <= s <= 60 for s in sleeper.calls)


def test_413_halves_chunk_size(connect, mock, dataset):
    scorer = connect()
    mock.inject["/v1/score"].extend([error(413, "too_many_items"), error(413, "body_too_large")])
    result = scorer.score(dataset["texts"], dataset["ids"], task=scorer.prepare(quickstart_nominal()), chunk_size=8, progress=False)
    assert normalized(result) == expected("nominal")
    assert [len(c) for c in mock.score_calls[:3]] == [8, 4, 2]


def test_response_shape_is_verified(connect, mock):
    scorer = connect()
    prepared = scorer.prepare(quickstart_nominal())

    def reordered(request):
        body = json.loads(request.content)
        resp = mock.score(body)
        data = resp.json()
        data["results"].reverse()
        return httpx.Response(200, json=data, headers=HEADERS)

    mock.inject["/v1/score"].append(reordered)
    with pytest.raises(ProtocolError):
        scorer.score(["a", "b"], ["r01", "r02"], task=prepared, progress=False)


# -- checkpoints ---------------------------------------------------------------------------------


def interrupt_after(mock, n_calls):
    original = mock.score
    count = {"n": 0}

    def maybe_interrupt(request):
        count["n"] += 1
        if count["n"] > n_calls:
            raise KeyboardInterrupt
        return original(json.loads(request.content))

    for _ in range(n_calls + 1):
        mock.inject["/v1/score"].append(maybe_interrupt)


def test_interrupt_and_resume(connect, mock, dataset, tmp_path):
    scorer = connect()
    prepared = scorer.prepare(quickstart_nominal())
    run = tmp_path / "run"
    interrupt_after(mock, 2)
    with pytest.raises(KeyboardInterrupt):
        scorer.score(dataset["texts"], dataset["ids"], task=prepared, chunk_size=5, checkpoint=run, progress=False)
    assert not (run / ".lock").exists()
    partial = read_checkpoint(run, allow_incomplete=True)
    assert sum(d["status"] == "pending" for d in partial.diagnostics) == 21 - 10
    with pytest.raises(CheckpointError, match="incomplete"):
        read_checkpoint(run)

    mock.calls.clear()
    with pytest.raises(CheckpointError, match="already exists"):
        scorer.score(dataset["texts"], dataset["ids"], task=prepared, chunk_size=5, checkpoint=run, progress=False)
    result = scorer.score(dataset["texts"], dataset["ids"], task=prepared, chunk_size=5, checkpoint=run, resume=True, progress=False)
    resubmitted = [i for call in mock.score_calls for i in call]
    assert len(resubmitted) == 11 and "r01" not in resubmitted
    assert normalized(result) == expected("nominal")
    assert normalized(read_checkpoint(run)) == expected("nominal")


def test_resume_retries_only_retryable_failures(connect, mock, dataset, tmp_path):
    scorer = connect()
    prepared = scorer.prepare(quickstart_nominal())
    run = tmp_path / "run"
    scorer.score(dataset["texts"], dataset["ids"], task=prepared, checkpoint=run, progress=False)
    mock.calls.clear()
    scorer.score(dataset["texts"], dataset["ids"], task=prepared, checkpoint=run, resume=True, progress=False)
    assert mock.score_calls == [["r22"]]  # inference_error is retried; deterministic failures are not


def test_all_success_checkpoint_makes_no_requests(connect, mock, dataset, tmp_path):
    scorer = connect()
    prepared = scorer.prepare(quickstart_nominal())
    ok = [(i, t) for i, t in zip(dataset["ids"], dataset["texts"]) if i in {"r01", "r02", "r03", "r05"}]
    ids, texts = [i for i, _ in ok], [t for _, t in ok]
    run = tmp_path / "run"
    scorer.score(texts, ids, task=prepared, checkpoint=run, progress=False)
    mock.calls.clear()
    again = scorer.score(texts, ids, task=prepared, checkpoint=run, resume=True, progress=False)
    assert mock.score_calls == [] and again.ids == ids


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda ids, texts: (ids, texts[:-1] + ["changed text"]), "texts"),
        (lambda ids, texts: (list(reversed(ids)), list(reversed(texts))), "ids"),
    ],
)
def test_resume_rejects_changed_inputs(connect, dataset, tmp_path, change, match):
    scorer = connect()
    prepared = scorer.prepare(quickstart_nominal())
    run = tmp_path / "run"
    scorer.score(dataset["texts"], dataset["ids"], task=prepared, checkpoint=run, progress=False)
    ids, texts = change(list(dataset["ids"]), list(dataset["texts"]))
    with pytest.raises(CheckpointError, match=match):
        scorer.score(texts, ids, task=prepared, checkpoint=run, resume=True, progress=False)


def test_resume_rejects_changed_task(connect, dataset, tmp_path):
    scorer = connect()
    run = tmp_path / "run"
    scorer.score(dataset["texts"], dataset["ids"], task=scorer.prepare(quickstart_nominal()), checkpoint=run, progress=False)
    with pytest.raises(CheckpointError, match="task"):
        scorer.score(dataset["texts"], dataset["ids"], task=scorer.prepare(quickstart_numeric()), checkpoint=run, resume=True, progress=False)


def test_lock_conflict_and_force_unlock(connect, dataset, tmp_path):
    scorer = connect()
    prepared = scorer.prepare(quickstart_nominal())
    run = tmp_path / "run"
    scorer.score(dataset["texts"][:3], dataset["ids"][:3], task=prepared, checkpoint=run, progress=False)
    (run / ".lock").mkdir()
    (run / ".lock" / "owner.json").write_text(json.dumps({"host": "elsewhere", "pid": 1, "created_at": "2026-01-01T00:00:00Z"}))
    with pytest.raises(CheckpointError, match="locked"):
        scorer.score(dataset["texts"][:3], dataset["ids"][:3], task=prepared, checkpoint=run, resume=True, progress=False)
    scorer.score(dataset["texts"][:3], dataset["ids"][:3], task=prepared, checkpoint=run, resume=True, force_unlock=True, progress=False)
    assert not (run / ".lock").exists()


def test_failure_between_response_and_commit_rescores_that_chunk(connect, mock, dataset, tmp_path, monkeypatch):
    scorer = connect()
    prepared = scorer.prepare(quickstart_nominal())
    run = tmp_path / "run"
    real_write = checkpoint_module.write_json_atomic
    state = {"chunks": 0}

    def failing_write(path, obj):
        if "chunk-" in str(path):
            state["chunks"] += 1
            if state["chunks"] == 3:  # client-side failures chunk, first scored chunk, then crash
                raise OSError("disk full")
        return real_write(path, obj)

    monkeypatch.setattr(checkpoint_module, "write_json_atomic", failing_write)
    with pytest.raises(OSError):
        scorer.score(dataset["texts"], dataset["ids"], task=prepared, chunk_size=5, checkpoint=run, progress=False)
    monkeypatch.setattr(checkpoint_module, "write_json_atomic", real_write)
    assert not list((run / "chunks").glob(".tmp-*"))
    (run / "chunks" / ".tmp-leftover.json").write_text("{partial")  # an interrupted write is ignored
    mock.calls.clear()
    result = scorer.score(dataset["texts"], dataset["ids"], task=prepared, chunk_size=5, checkpoint=run, resume=True, progress=False)
    assert normalized(result) == expected("nominal")
    assert sum(len(c) for c in mock.score_calls) == 21 - 5


def test_non_checkpoint_directory_is_not_overwritten(connect, tmp_path):
    scorer = connect()
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "important.csv").write_text("x")
    with pytest.raises(CheckpointError, match="not a llikert checkpoint"):
        scorer.score(["a"], ["r01"], task=scorer.prepare(quickstart_nominal()), checkpoint=tmp_path / "data", progress=False)


def test_checkpoint_excludes_texts_and_credentials(mock, dataset, tmp_path):
    mock.token = "very-secret-token-123"
    scorer = Scorer.connect("http://mock", token="very-secret-token-123", _client=mock.client())
    run = tmp_path / "run"
    scorer.score(dataset["texts"], dataset["ids"], task=scorer.prepare(quickstart_nominal()), checkpoint=run, progress=False)
    blob = "".join(p.read_text() for p in run.rglob("*.json"))
    assert "very-secret-token" not in blob and "Where is the nearest train station?" not in blob


def test_result_save_load_roundtrip(connect, dataset, tmp_path):
    scorer = connect()
    result = scorer.score(dataset["texts"], dataset["ids"], task=scorer.prepare(quickstart_numeric()), progress=False)
    result.save(tmp_path / "result.json")
    loaded = ScoreResult.load(tmp_path / "result.json")
    assert loaded == result
    assert normalized(loaded) == expected("numeric")
