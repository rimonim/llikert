"""Benchmark one execution profile on the synthetic corpus.

    .venv/bin/python benchmarks/run.py --model models/qwen3-4b-instruct-2507-f32.gguf --label sequential \
        [--batch-size 512] [--flash-attn off] [--kv-type f32] [--limit N] [--http]

Writes benchmarks/results/<label>.json with startup, per-task throughput and latency by
text length, prompt lengths, native decode calls, peak memory, and (with --http) the
overhead of scoring the same items through the HTTP service with the Python client.
Per-item scores go to build/benchmarks/<label>-scores.json for item-by-item comparisons.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import resource
import socket
import statistics
import subprocess
import sys
import threading
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python" / "src"))

from llikert.server.llama_adapter import LlamaAdapter, LlamaConfig, sha256_file  # noqa: E402
from llikert.server.service import Limits, ScoringService  # noqa: E402


def gpu_memory_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in out.splitlines():
        pid, used = [x.strip() for x in line.split(",")]
        if int(pid) == os.getpid():
            return int(used)
    return None


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=pathlib.Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--corpus", type=pathlib.Path, default=REPO / "benchmarks/corpus-v1.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--flash-attn", default="off")
    ap.add_argument("--kv-type", default="f32")
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--chunk-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="texts per task (default: all)")
    ap.add_argument("--tasks", default="binary-food,sentiment-5,topic-10")
    ap.add_argument("--http", action="store_true", help="also measure HTTP overhead on short texts")
    args = ap.parse_args()

    corpus = json.loads(args.corpus.read_text())
    texts = corpus["texts"][: args.limit] if args.limit else corpus["texts"]

    t0 = time.perf_counter()
    sha_seconds = time.perf_counter()
    sha256_file(args.model)
    sha_seconds = time.perf_counter() - sha_seconds
    adapter = LlamaAdapter(LlamaConfig(model_path=args.model, device=args.device, n_ctx=args.n_ctx, batch_size=args.batch_size,
                                       flash_attn=args.flash_attn, kv_type=args.kv_type))
    service = ScoringService(adapter, Limits(max_items_per_request=max(64, args.chunk_size)))
    service.smoke_check()
    startup_seconds = time.perf_counter() - t0 - sha_seconds  # adapter hashes again; report load separately

    report: dict = {
        "label": args.label,
        "corpus_version": corpus["corpus_version"],
        "platform": platform.platform(),
        "settings": {k: getattr(args, k) for k in ("device", "batch_size", "flash_attn", "kv_type", "n_ctx", "chunk_size")},
        "engine_fingerprint": service.fingerprint,
        "math_libraries": service.identity["model_and_execution"].get("math_libraries"),
        "startup": {"sha256_seconds": round(sha_seconds, 2), "load_and_smoke_seconds": round(startup_seconds, 2)},
        "tasks": {},
        "scores": {},
    }
    for name in args.tasks.split(","):
        prepared = service.prepare(corpus["tasks"][name])["prepared"]
        warm = [{"id": f"warm-{i}", "text": t["text"]} for i, t in enumerate(texts[:3])]
        service.score(prepared, warm)
        before = dict(adapter.stats)
        latencies: dict[str, list[float]] = {}
        prompt_tokens: dict[str, list[int]] = {}
        scores = {}
        started = time.perf_counter()
        for i in range(0, len(texts), args.chunk_size):
            chunk = texts[i : i + args.chunk_size]
            for item in chunk:
                t_item = time.perf_counter()
                r = service.score(prepared, [{"id": item["id"], "text": item["text"]}])["results"][0]
                latencies.setdefault(item["length"], []).append(time.perf_counter() - t_item)
                prompt_tokens.setdefault(item["length"], []).append(r["n_prompt_tokens"] or 0)
                scores[item["id"]] = {"status": r["status"], "p": r["probabilities"], "log_q": r["candidate_log_probs"]}
        elapsed = time.perf_counter() - started
        after = adapter.stats
        report["tasks"][name] = {
            "items": len(texts),
            "seconds": round(elapsed, 3),
            "items_per_second": round(len(texts) / elapsed, 2),
            "decode_calls": after["decode_calls"] - before["decode_calls"],
            "tokens_evaluated": after["tokens_evaluated"] - before["tokens_evaluated"],
            "by_length": {
                length: {
                    "n": len(v),
                    "mean_prompt_tokens": round(statistics.mean(prompt_tokens[length]), 1),
                    "median_ms": round(1000 * statistics.median(v), 2),
                    "p95_ms": round(1000 * percentile(v, 0.95), 2),
                }
                for length, v in latencies.items()
            },
        }
        report["scores"][name] = scores
        print(name, json.dumps({k: v for k, v in report["tasks"][name].items() if k != "by_length"}), flush=True)

    report["memory"] = {"gpu_mib": gpu_memory_mib(), "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}

    if args.http:
        report["http"] = http_overhead(service, corpus, texts)
        print("http", json.dumps(report["http"]), flush=True)

    scores = report.pop("scores")
    out = REPO / "benchmarks/results" / f"{args.label}.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    score_dir = REPO / "build" / "benchmarks"
    score_dir.mkdir(parents=True, exist_ok=True)
    (score_dir / f"{args.label}-scores.json").write_text(json.dumps(scores) + "\n")
    print("wrote", out)
    adapter.close()
    return 0


def http_overhead(service: ScoringService, corpus: dict, texts: list[dict]) -> dict:
    import uvicorn

    from llikert import Scorer, ScoringTask
    from llikert.server.app import create_app

    sample = [t for t in texts if t["length"] == "short"][:128]
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(lambda: service), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        scorer = Scorer.connect(f"http://127.0.0.1:{port}", wait=60)
        task = ScoringTask.from_dict(corpus["tasks"]["binary-food"])
        prepared = scorer.prepare(task)
        scorer.score([t["text"] for t in sample[:16]], [t["id"] for t in sample[:16]], task=prepared, progress=False)
        t0 = time.perf_counter()
        scorer.score([t["text"] for t in sample], [t["id"] for t in sample], task=prepared, chunk_size=16, progress=False)
        via_http = time.perf_counter() - t0
        items = [{"id": t["id"], "text": t["text"]} for t in sample]
        t0 = time.perf_counter()
        for i in range(0, len(items), 16):
            service.score(prepared.artifact, items[i : i + 16])
        direct = time.perf_counter() - t0
        scorer.close()
    finally:
        server.should_exit = True
        thread.join(10)
    return {"items": len(sample), "chunk_size": 16, "http_seconds": round(via_http, 3), "direct_seconds": round(direct, 3),
            "overhead_ms_per_item": round(1000 * (via_http - direct) / len(sample), 3)}


if __name__ == "__main__":
    sys.exit(main())
