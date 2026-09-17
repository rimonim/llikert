"""Reference-fidelity gate (decision D24) against a running service over HTTP.

    LLIKERT_TOKEN=... .venv/bin/python tools/fidelity_gate_http.py --url http://127.0.0.1:8080 \
        [--fixture tests/fixtures/fidelity/qwen3-4b-instruct-2507-v2.json] [--json out.json]

Use it to verify a container or hosted endpoint on its own hardware. Prompt-token parity
with the reference is checked through each result's prompt_token_sha256.
Only the base client (httpx) is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import struct
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python" / "src"))

from llikert import Scorer, ScoringTask  # noqa: E402


def token_hash(tokens: list[int]) -> str:
    return "sha256:" + hashlib.sha256(struct.pack(f"<{len(tokens)}i", *tokens)).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--fixture", type=pathlib.Path, default=REPO / "tests/fixtures/fidelity/qwen3-4b-instruct-2507-v2.json")
    ap.add_argument("--chunk-size", type=int, default=16)
    ap.add_argument("--json", type=pathlib.Path)
    args = ap.parse_args()

    fixture = json.loads(args.fixture.read_text())
    tol = fixture["tolerances"]
    rows, token_mismatches = [], []
    with Scorer.connect(args.url, wait=600) as scorer:
        for name, task_obj in fixture["tasks"].items():
            prepared = scorer.prepare(ScoringTask.from_dict(task_obj))
            items = [i for i in fixture["items"] if i["task"] == name]
            result = scorer.score([i["text"] for i in items], [i["key"] for i in items], task=prepared,
                                  chunk_size=args.chunk_size, progress=False)
            for item, r in zip(items, result.items):
                ref = item["reference"]
                if r["status"] != "ok":
                    raise SystemExit(f"{item['key']}: status {r['status']}")
                if r["prompt_token_sha256"] != token_hash(ref["tokens"]):
                    token_mismatches.append(item["key"])
                dp = max(abs(a - b) for a, b in zip(r["probabilities"], ref["probabilities"]))
                pairs = [(a, b) for a, b in zip(r["candidate_log_probs"], ref["candidate_log_probs"]) if b > tol["log_prob_floor"]]
                dlq = max((abs(a - b) for a, b in pairs), default=0.0)
                rows.append({"key": item["key"], "max_abs_dp": dp, "max_abs_dlogq": dlq})
        info = scorer.info
    worst_p = max(r["max_abs_dp"] for r in rows)
    worst_lq = max(r["max_abs_dlogq"] for r in rows)
    passed = not token_mismatches and worst_p <= tol["max_abs_probability"] and worst_lq <= tol["max_abs_candidate_log_prob"]
    report = {
        "passed": passed,
        "url": args.url,
        "fixture": args.fixture.name,
        "items": len(rows),
        "token_mismatches": token_mismatches[:10],
        "max_abs_dp": worst_p,
        "max_abs_dlogq": worst_lq,
        "tolerances": tol,
        "engine_fingerprint": info["engine_fingerprint"],
        "engine": info["engine"]["model_and_execution"],
        "worst_items": sorted(rows, key=lambda r: -r["max_abs_dlogq"])[:3],
    }
    print(json.dumps(report, indent=1))
    if args.json:
        args.json.write_text(json.dumps(report, indent=1) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
