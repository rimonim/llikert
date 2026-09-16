"""Reference-fidelity gate (decision D24) for one model profile.

    .venv/bin/python tools/fidelity_gate.py --model models/qwen3-4b-instruct-2507-f32.gguf \
        [--device cuda] [--batch-size 512] [--flash-attn off] [--kv-type f32] [--json out.json]

Exit status 0 when both tolerances hold. Reports the worst items so failures can be inspected.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python" / "src"))

from llikert.server.llama_adapter import LlamaAdapter, LlamaConfig  # noqa: E402
from llikert.server.service import ScoringService  # noqa: E402
from llikert.server.task import parse_task  # noqa: E402


def _group(key: str) -> str:
    """'topic-10/long-0017' -> 'long'; fixture keys without a length prefix -> 'all'."""
    tail = key.split("/")[-1]
    return tail.split("-")[0] if tail.split("-")[0] in ("short", "medium", "long") else "all"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=pathlib.Path, required=True)
    ap.add_argument("--fixture", type=pathlib.Path, default=REPO / "tests/fixtures/fidelity/qwen3-4b-instruct-2507.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--flash-attn", default="off")
    ap.add_argument("--kv-type", default="f32")
    ap.add_argument("--shuffle-seed", type=int, default=None, help="score items in a shuffled order")
    ap.add_argument("--json", type=pathlib.Path)
    args = ap.parse_args()

    fixture = json.loads(args.fixture.read_text())
    tol = fixture["tolerances"]
    adapter = LlamaAdapter(LlamaConfig(model_path=args.model, device=args.device, batch_size=args.batch_size,
                                       flash_attn=args.flash_attn, kv_type=args.kv_type))
    service = ScoringService(adapter)
    service.smoke_check()
    prepared = {name: service.prepare(task)["prepared"] for name, task in fixture["tasks"].items()}
    rows = []
    t0 = time.perf_counter()
    items = list(fixture["items"])
    if args.shuffle_seed is not None:
        import random

        random.Random(args.shuffle_seed).shuffle(items)
    scores = {}
    for item in items:
        task = parse_task(fixture["tasks"][item["task"]])
        if service.preparer.prompt_tokens(task, item["text"])[0] != item["reference"]["tokens"]:
            raise SystemExit(f"prompt tokens differ from the reference for {item['key']}")
        r = service.score(prepared[item["task"]], [{"id": item["key"], "text": item["text"]}])["results"][0]
        ref = item["reference"]
        dp = max(abs(a - b) for a, b in zip(r["probabilities"], ref["probabilities"]))
        pairs = [(a, b) for a, b in zip(r["candidate_log_probs"], ref["candidate_log_probs"]) if b > tol["log_prob_floor"]]
        dlq, at = max(((abs(a - b), b) for a, b in pairs), default=(0.0, None))
        rows.append({"key": item["key"], "max_abs_dp": dp, "max_abs_dlogq": dlq, "reference_log_q_at_max": at})
        scores[item["key"]] = {"p": r["probabilities"], "log_q": r["candidate_log_probs"]}
    elapsed = time.perf_counter() - t0
    worst_p = max(r["max_abs_dp"] for r in rows)
    worst_lq = max(r["max_abs_dlogq"] for r in rows)
    passed = worst_p <= tol["max_abs_probability"] and worst_lq <= tol["max_abs_candidate_log_prob"]
    report = {
        "passed": passed,
        "settings": {"device": args.device, "batch_size": args.batch_size, "flash_attn": args.flash_attn, "kv_type": args.kv_type,
                     "shuffle_seed": args.shuffle_seed},
        "tolerances": tol,
        "max_abs_dp": worst_p,
        "max_abs_dlogq": worst_lq,
        "ms_per_item": 1000 * elapsed / len(rows),
        "engine_fingerprint": service.fingerprint,
        "math_libraries": service.identity["model_and_execution"].get("math_libraries"),
        "worst_items": sorted(rows, key=lambda r: -r["max_abs_dlogq"])[:3],
        "by_group": {
            group: {
                "items": len(members),
                "max_abs_dp": max(r["max_abs_dp"] for r in members),
                "max_abs_dlogq": max(r["max_abs_dlogq"] for r in members),
            }
            for group in sorted({_group(r["key"]) for r in rows})
            for members in [[r for r in rows if _group(r["key"]) == group]]
        },
    }
    adapter.close()
    print(json.dumps(report, indent=1))
    if args.json:
        args.json.write_text(json.dumps({**report, "scores": scores}, indent=1) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
