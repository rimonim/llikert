"""Stress and state-isolation checks on a real model (opt in with ``pytest -m model``).

Uses LLIKERT_TEST_MODEL_SMALL for speed; the properties are engine-level.
"""

import json
import os
import pathlib
import random
import socket
import subprocess
import sys
import threading

import pytest

from llikert.server.llama_adapter import LlamaAdapter, LlamaConfig
from llikert.server.service import Limits, ScoringService

pytestmark = pytest.mark.model
REPO = pathlib.Path(__file__).resolve().parents[3]
CORPUS = json.loads((REPO / "benchmarks/corpus-v1.json").read_text())


def model_path() -> pathlib.Path:
    value = os.environ.get("LLIKERT_TEST_MODEL_SMALL")
    if not value:
        pytest.skip("LLIKERT_TEST_MODEL_SMALL not set")
    return pathlib.Path(value)


@pytest.fixture(scope="module")
def service():
    svc = ScoringService(LlamaAdapter(LlamaConfig(model_path=model_path(), n_ctx=2048)), Limits(max_items_per_request=64))
    svc.smoke_check()
    yield svc
    svc.adapter.close()


@pytest.fixture(scope="module")
def prepared(service):
    return service.prepare(CORPUS["tasks"]["topic-10"])["prepared"]


def items(n=40, seed=0):
    texts = CORPUS["texts"][:300] + CORPUS["texts"][450:460]
    chosen = random.Random(seed).sample(texts, n)
    return [{"id": t["id"], "text": t["text"]} for t in chosen]


def by_id(results):
    return {r["id"]: r for r in results}


def test_state_is_isolated_between_items(service, prepared):
    probe = [{"id": "probe", "text": "The printer at the library was a complete disaster."}]
    first = service.score(prepared, probe)["results"][0]
    long_text = [{"id": "long", "text": CORPUS["texts"][455]["text"]}]
    over = [{"id": "over", "text": "word " * 3000}]
    service.score(prepared, long_text)
    assert service.score(prepared, probe)["results"][0] == first
    assert service.score(prepared, over)["results"][0]["status"] == "context_limit"
    assert service.score(prepared, probe)["results"][0] == first
    assert service.healthy


def test_results_do_not_depend_on_chunking_or_order(service, prepared):
    batch = items()
    reference = by_id(service.score(prepared, batch)["results"])
    shuffled = list(batch)
    random.Random(1).shuffle(shuffled)
    chunked = {}
    for size in (1, 3, 7):
        for i in range(0, len(shuffled), size):
            chunked.update(by_id(service.score(prepared, shuffled[i : i + size])["results"]))
        assert chunked == reference


def test_repeated_requests_are_bitwise_identical(service, prepared):
    batch = items(16, seed=2)
    runs = [service.score(prepared, batch)["results"] for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


def test_process_restart_gives_identical_results(service, prepared, tmp_path):
    batch = items(12, seed=3)
    here = service.score(prepared, batch)["results"]
    script = tmp_path / "score.py"
    script.write_text(
        "import json, pathlib, sys\n"
        "from llikert.server.llama_adapter import LlamaAdapter, LlamaConfig\n"
        "from llikert.server.service import ScoringService\n"
        "args = json.loads(sys.stdin.read())\n"
        "svc = ScoringService(LlamaAdapter(LlamaConfig(model_path=pathlib.Path(args['model']), n_ctx=2048)))\n"
        "svc.smoke_check()\n"
        "print(json.dumps(svc.score(args['prepared'], args['items'])['results']))\n"
        "svc.adapter.close()\n"
    )
    payload = json.dumps({"model": str(model_path()), "prepared": prepared, "items": batch})
    out = subprocess.run([sys.executable, str(script)], input=payload, capture_output=True, text=True, check=True)
    there = json.loads(out.stdout.strip().splitlines()[-1])
    assert json.loads(json.dumps(here)) == there


def test_prompt_at_context_limit(service, prepared):
    n_ctx = service.adapter.n_ctx
    base = service.preparer.prompt_tokens(service_task(), "")[0]
    exact = None
    for n in range(n_ctx - len(base) - 20, n_ctx - len(base) + 20):
        text = " a" * n
        if len(service.preparer.prompt_tokens(service_task(), text)[0]) == n_ctx:
            exact = text
            break
    assert exact is not None
    result = service.score(prepared, [{"id": "exact", "text": exact}])["results"][0]
    assert result["status"] == "ok" and result["n_prompt_tokens"] == n_ctx


def service_task():
    from llikert.server.task import parse_task

    return parse_task(CORPUS["tasks"]["topic-10"])


def test_concurrent_http_clients_get_correct_results(service, prepared):
    import uvicorn

    from llikert import RetryPolicy, Scorer, ScoringTask
    from llikert.server.app import HttpSettings, create_app

    batch = items(48, seed=4)
    reference = by_id(service.score(prepared, batch)["results"])
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    app = create_app(lambda: service, HttpSettings(queue_limit=2, retry_after_seconds=1))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    errors, outputs = [], {}
    try:
        def client(k):
            try:
                with Scorer.connect(f"http://127.0.0.1:{port}", wait=60, retry=RetryPolicy(max_tries=60, base_seconds=0.05, cap_seconds=0.2)) as scorer:
                    task = scorer.prepare(ScoringTask.from_dict(CORPUS["tasks"]["topic-10"]))
                    part = batch[k::4]
                    result = scorer.score([i["text"] for i in part], [i["id"] for i in part], task=task, chunk_size=3, progress=False)
                    outputs[k] = result.items
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=client, args=(k,)) for k in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(120)
    finally:
        server.should_exit = True
        thread.join(10)
    assert not errors
    got = {r["id"]: {k: v for k, v in r.items() if k != "text_sha256"} for items_ in outputs.values() for r in items_}
    assert got == reference
