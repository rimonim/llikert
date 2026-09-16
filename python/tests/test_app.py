import json
import threading
import time

import pytest
from starlette.testclient import TestClient

from llikert.server.app import AuthMode, HttpSettings, create_app
from llikert.server.fake_adapter import FakeAdapter
from llikert.server.schemas import ErrorEnvelope, InfoResponse, PrepareResponse, ScoreResponse
from llikert.server.service import StartupError

from conftest import make_service, task_dict

TOKEN = "t" * 32


def wait_ready(client, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if client.get("/health").status_code == 200:
            return
        time.sleep(0.01)
    raise AssertionError("service did not become ready")


@pytest.fixture
def client():
    app = create_app(lambda: make_service(ready=False))
    with TestClient(app) as c:
        wait_ready(c)
        yield c


def prepare(client, task=None, **extra):
    body = {"protocol_version": 1, "task": task or task_dict(), **extra}
    return client.post("/v1/prepare", json=body)


def assert_error(resp, status, code):
    assert resp.status_code == status, resp.text
    ErrorEnvelope.model_validate(resp.json())
    assert resp.json()["error"]["code"] == code
    assert resp.headers["LLikert-Protocol"] == "1"


def test_prepare_and_score_roundtrip(client):
    r = prepare(client, request_id="req-1", preview_text="Hello")
    assert r.status_code == 200 and r.headers["LLikert-Protocol"] == "1"
    body = PrepareResponse.model_validate(r.json())
    assert body.request_id == "req-1"
    s = client.post("/v1/score", json={"protocol_version": 1, "prepared": r.json()["prepared"], "items": [{"id": "1", "text": "Hi"}, {"id": "2", "text": None}]})
    assert s.status_code == 200
    out = ScoreResponse.model_validate(s.json())
    assert [x.status for x in out.results] == ["ok", "missing_input"]
    assert out.request_id  # generated when not supplied


def test_info(client):
    info = InfoResponse.model_validate(client.get("/v1/info").json())
    assert info.protocol_version == 1 and info.limits.max_items_per_request == 64


def test_health_is_minimal(client):
    r = client.get("/health")
    assert r.json() == {"status": "ready"}


def test_health_503_while_loading_and_after_failed_startup():
    release = threading.Event()

    def slow_loader():
        release.wait(5)
        return make_service(ready=False)

    with TestClient(create_app(slow_loader)) as c:
        r = c.get("/health")
        assert r.status_code == 503 and "Retry-After" in r.headers
        assert_error(c.get("/v1/info"), 503, "starting")
        release.set()
        wait_ready(c)

    def failing_loader():
        raise StartupError("chat template sha256 abc is not a supported template profile")

    with TestClient(create_app(failing_loader)) as c:
        time.sleep(0.05)
        assert c.get("/health").status_code == 503
        assert_error(c.get("/v1/info"), 503, "startup_failed")


@pytest.mark.parametrize(
    "raw,code",
    [
        (b'{"protocol_version": 1, "task": NaN}', "invalid_json"),
        (b'{"protocol_version": 1, "protocol_version": 1, "task": {}}', "invalid_json"),
        (b"\xff\xfe not utf8", "invalid_json"),
        (b"[1, 2", "invalid_json"),
    ],
)
def test_strict_json(client, raw, code):
    assert_error(client.post("/v1/prepare", content=raw, headers={"content-type": "application/json"}), 400, code)


def test_unknown_request_field_rejected(client):
    assert_error(prepare(client, temperature=0.0), 422, "invalid_request")


def test_wrong_protocol_version(client):
    assert_error(client.post("/v1/prepare", json={"protocol_version": 2, "task": task_dict()}), 422, "invalid_request")


def test_invalid_task_code(client):
    assert_error(prepare(client, task_dict(categories=[])), 422, "invalid_task")


def test_invalid_response_codes_include_suggestions(client):
    t = task_dict()
    t["categories"][2]["response"] = "10"
    r = prepare(client, t)
    assert_error(r, 422, "invalid_response_codes")
    assert r.json()["error"]["details"]["suggestions"]["generic"]


def test_item_validation(client):
    prepared = prepare(client).json()["prepared"]
    for items in ([{"id": "1", "text": 5}], [{"id": 1, "text": "x"}], [{"id": "", "text": "x"}], [{"id": "1"}], [{"id": "1", "text": "x", "extra": 1}]):
        assert_error(client.post("/v1/score", json={"protocol_version": 1, "prepared": prepared, "items": items}), 422, "invalid_request")
    raw = '{"protocol_version": 1, "prepared": %s, "items": [{"id": "1", "text": "bad \\udc80"}]}' % json.dumps(prepared)
    r = client.post("/v1/score", content=raw.encode(), headers={"content-type": "application/json"})
    assert r.status_code == 200 and r.json()["results"][0]["status"] == "invalid_encoding"
    raw_id = '{"protocol_version": 1, "prepared": %s, "items": [{"id": "\\ud800", "text": "x"}]}' % json.dumps(prepared)
    assert_error(client.post("/v1/score", content=raw_id.encode(), headers={"content-type": "application/json"}), 422, "invalid_request")


def test_fingerprint_mismatch_409(client):
    prepared = prepare(client).json()["prepared"]
    prepared["engine_fingerprint"] = "sha256:" + "0" * 64
    assert_error(client.post("/v1/score", json={"protocol_version": 1, "prepared": prepared, "items": []}), 409, "engine_fingerprint_mismatch")


def test_body_limit():
    app = create_app(lambda: make_service(ready=False), HttpSettings(max_body_bytes=2000))
    with TestClient(app) as c:
        wait_ready(c)
        assert_error(c.post("/v1/prepare", json={"protocol_version": 1, "task": task_dict(instructions="x" * 5000)}), 413, "body_too_large")


def test_not_found_uses_envelope(client):
    assert_error(client.get("/v1/nope"), 404, "not_found")
    assert_error(client.get("/v1/score"), 405, "method_not_allowed")


def test_no_shutdown_or_model_routes(client):
    paths = set(client.get("/openapi.json").json()["paths"])
    assert paths == {"/health", "/v1/info", "/v1/prepare", "/v1/score"}


# -- auth ------------------------------------------------------------------------------------


@pytest.fixture
def token_client():
    app = create_app(lambda: make_service(ready=False), HttpSettings(auth=AuthMode.TOKEN, token=TOKEN))
    with TestClient(app) as c:
        wait_ready(c)
        yield c


def test_token_auth(token_client):
    assert token_client.get("/health").status_code == 200
    assert_error(token_client.get("/v1/info"), 401, "unauthorized")
    assert_error(token_client.get("/v1/info", headers={"Authorization": "Bearer wrong"}), 401, "unauthorized")
    assert_error(prepare(token_client), 401, "unauthorized")
    assert token_client.get("/v1/info", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200
    assert token_client.get("/openapi.json").status_code == 404
    assert token_client.get("/docs").status_code == 404


def test_token_mode_requires_token():
    with pytest.raises(ValueError):
        create_app(lambda: make_service(), HttpSettings(auth=AuthMode.TOKEN))


def test_auth_secret_never_in_responses(token_client):
    r = token_client.get("/v1/info", headers={"Authorization": f"Bearer {TOKEN}"})
    assert TOKEN not in r.text


# -- concurrency -------------------------------------------------------------------------------


def test_queue_limit_returns_429_and_health_stays_responsive():
    gate = threading.Event()
    started = threading.Event()

    class Slow(FakeAdapter):
        def final_logits(self, tokens):
            if self.decode_calls >= 1:  # after the smoke check
                started.set()
                gate.wait(5)
            return super().final_logits(tokens)

    app = create_app(lambda: make_service(Slow(), ready=False), HttpSettings(queue_limit=1))
    with TestClient(app) as c:
        wait_ready(c)
        prepared = prepare(c).json()["prepared"]
        results = {}

        def slow_request():
            results["slow"] = c.post("/v1/score", json={"protocol_version": 1, "prepared": prepared, "items": [{"id": "1", "text": "x"}]})

        t = threading.Thread(target=slow_request)
        t.start()
        assert started.wait(5)
        t0 = time.time()
        assert c.get("/health").status_code == 200
        assert time.time() - t0 < 1.0
        busy = c.post("/v1/score", json={"protocol_version": 1, "prepared": prepared, "items": []})
        assert_error(busy, 429, "queue_full")
        assert "Retry-After" in busy.headers
        gate.set()
        t.join(5)
        assert results["slow"].status_code == 200
