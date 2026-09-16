"""Reference-fidelity fixtures from transformers (decision D24).

    .venv-ref/bin/python benchmarks/reference.py --hf models/hf/Qwen3-4B-Instruct-2507 \
        --corpus build/fidelity-corpus-v2.json --out tests/fixtures/fidelity/qwen3-4b-instruct-2507-v2.json [--device cuda]

Prompts are built with llikert's renderer message structure but rendered and tokenized by
transformers (apply_chat_template + the official tokenizer), so the reference does not
share llama.cpp's tokenizer. Logits are float32 with TF32 disabled, reduced in float64.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python" / "src"))
from llikert.server.render import messages_for  # noqa: E402
from llikert.server.task import parse_task  # noqa: E402


def reduce(logits: torch.Tensor, ids: list[int]) -> tuple[list[float], list[float]]:
    z = logits.to(torch.float64)
    log_q = z[ids] - torch.logsumexp(z, dim=0)
    p = torch.softmax(z[ids], dim=0)
    return p.tolist(), log_q.tolist()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", type=pathlib.Path, required=True)
    ap.add_argument("--corpus", type=pathlib.Path)
    ap.add_argument("--from-fixture", type=pathlib.Path, help="recompute references for an existing fidelity fixture")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(str(args.hf))
    model = AutoModelForCausalLM.from_pretrained(str(args.hf), dtype=torch.float32, attn_implementation="eager").to(args.device).eval()

    if args.from_fixture:
        base = json.loads(args.from_fixture.read_text())
        tasks = base["tasks"]
        plan = [(item["key"], item["task"], item["text"]) for item in base["items"]]
    else:
        corpus = json.loads(args.corpus.read_text())
        tasks = corpus["tasks"]
        plan = [(f"{name}/{t['id']}", name, t["text"]) for name in tasks for t in corpus["texts"]]

    items = []
    t0 = time.time()
    with torch.no_grad():
        for key, task_name, text in plan:
            task = parse_task(tasks[task_name])
            rendered = tok.apply_chat_template(messages_for(task, text), tokenize=False, add_generation_prompt=True)
            ids = tok.encode(rendered, add_special_tokens=False)
            cand = []
            for c in task.categories:
                enc = tok.encode(c.response, add_special_tokens=False)
                if len(enc) != 1:
                    raise SystemExit(f"response {c.response!r} is not one token for the reference tokenizer")
                cand.append(enc[0])
            logits = model(torch.tensor([ids], device=args.device)).logits[0, -1]
            p, log_q = reduce(logits, cand)
            items.append({"key": key, "task": task_name, "text": text, "reference": {"tokens": ids, "probabilities": p, "candidate_log_probs": log_q}})
    import transformers

    out = {
        "description": "Reference-fidelity fixture (decision D24): transformers float32 (TF32 off), eager attention, official weights; "
                       "prompts rendered and tokenized by transformers.",
        "model": {"repo": "Qwen/Qwen3-4B-Instruct-2507", "revision": "cdbee75f17c01a7cc42f958dc650907174af0554"},
        "reference_software": {"transformers": transformers.__version__, "torch": torch.__version__, "dtype": "float32",
                               "attn_implementation": "eager", "device": args.device,
                               "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None},
        "tolerances": {"max_abs_probability": 0.005, "max_abs_candidate_log_prob": 0.1, "log_prob_floor": -15.0},
        "tasks": tasks,
        "items": items,
    }
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {len(items)} items to {args.out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
