"""Replays tests/fixtures/protocol/fake-service.json over httpx.MockTransport."""

from __future__ import annotations

import collections
import copy
import json
import pathlib
from typing import Any

import httpx

REPO = pathlib.Path(__file__).resolve().parents[3]
FIXTURES = REPO / "tests" / "fixtures"
HEADERS = {"LLikert-Protocol": "1"}


def load_fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text())


def error(status: int, code: str, message: str = "error", headers: dict | None = None) -> httpx.Response:
    return httpx.Response(status, json={"error": {"code": code, "message": message}}, headers={**HEADERS, **(headers or {})})


class MockService:
    def __init__(self, token: str | None = None):
        self.fixture = load_fixture("protocol/fake-service.json")
        self.token = token
        self.calls: list[tuple[str, Any]] = []
        self.inject: dict[str, collections.deque] = collections.defaultdict(collections.deque)

    @property
    def score_calls(self) -> list[list[str]]:
        return [[item["id"] for item in body["items"]] for path, body in self.calls if path == "/v1/score"]

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler), base_url="http://mock")

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content) if request.content else None
        self.calls.append((path, body))
        if self.inject[path]:
            injected = self.inject[path].popleft()
            if isinstance(injected, BaseException):
                raise injected
            if callable(injected):
                return injected(request)
            return injected
        if path == "/health":
            return httpx.Response(200, json={"status": "ready"}, headers=HEADERS)
        if self.token and request.headers.get("Authorization") != f"Bearer {self.token}":
            return error(401, "unauthorized")
        if path == "/v1/info" and request.method == "GET":
            return httpx.Response(200, json=self.fixture["info"], headers=HEADERS)
        if path == "/v1/prepare" and request.method == "POST":
            return self.prepare(body)
        if path == "/v1/score" and request.method == "POST":
            return self.score(body)
        return error(404, "not_found")

    def prepare(self, body: dict) -> httpx.Response:
        for entry in self.fixture["tasks"].values():
            if body["task"] == entry["task"]:
                out = copy.deepcopy(entry["prepare_response"])
                out["request_id"] = body.get("request_id") or out["request_id"]
                if "preview_text" in body:
                    out["preview"]["text"] = {"messages": [{"role": "user", "content": body["preview_text"]}], "prompt": "…" + body["preview_text"] + "…", "n_prompt_tokens": 1}
                return httpx.Response(200, json=out, headers=HEADERS)
        invalid = self.fixture["invalid_codes"]
        if body["task"] == invalid["task"]:
            return httpx.Response(invalid["error_response"]["status"], json=invalid["error_response"]["body"], headers=HEADERS)
        return error(422, "invalid_task", "task failed validation")

    def score(self, body: dict) -> httpx.Response:
        prepared = body["prepared"]
        if prepared.get("engine_fingerprint") != self.fixture["info"]["engine_fingerprint"]:
            return error(409, "engine_fingerprint_mismatch")
        for entry in self.fixture["tasks"].values():
            if prepared == entry["prepare_response"]["prepared"]:
                results = [entry["item_results"][item["id"]] for item in body["items"]]
                out = {
                    "protocol_version": 1,
                    "request_id": body.get("request_id") or "generated",
                    "engine_fingerprint": prepared["engine_fingerprint"],
                    "prepared_hash": prepared["prepared_hash"],
                    "category_ids": [c["id"] for c in prepared["categories"]],
                    "results": results,
                    "meta": {"elapsed_ms": 1.0, "execution": entry["meta_execution"]},
                }
                return httpx.Response(200, json=out, headers=HEADERS)
        return error(409, "prepared_task_mismatch")
