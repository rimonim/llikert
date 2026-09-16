"""M0 probe: fidelity of candidate execution profiles against a transformers reference on
deliberately ambiguous texts, plus warm per-item latency.

usage:
  .venv-cpu/bin/python spikes/fidelity.py hf --hf models/hf/Qwen3-4B-Instruct-2507
  .venv/bin/python spikes/fidelity.py llama --model M --label L [--cpu] --n-ubatch N --fa off --kv 0
"""

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import m0_logits as m  # noqa: E402

FUNCTION_TEXTS = [
    "I was wondering whether the meeting is still on.",
    "It would be great if someone could fix the printer.",
    "The door is open.",
    "Do you know how late the shop is open?",
    "You might want to check the oven.",
    "Is it cold in here, or is it just me?",
    "I need the keys.",
    "Can you believe how fast this year went?",
    "Someone left the lights on again.",
    "Let's grab lunch sometime.",
    "Any chance the report will be ready today?",
    "The deadline is tomorrow, just so you know.",
    "Would it kill you to call once in a while?",
    "I'd love a cup of tea.",
    "Why don't you take a break?",
    "The printer is out of paper.",
    "Tell me about your trip.",
    "I wonder what time it is.",
    "Feel free to join us after work.",
    "Could it be that the bus is late again?",
]
SENTIMENT_TEXTS = [
    "The food was fine, though the service was slow.",
    "Not bad at all, considering the price.",
    "I expected more, but it was okay.",
    "The hotel was clean but the staff seemed tired.",
    "Honestly, it could have been worse.",
    "It rained all week, but we still had fun.",
    "The movie started strong and then dragged on.",
    "Delivery was late, but the product works well.",
]


def items():
    for i, t in enumerate(FUNCTION_TEXTS):
        yield f"function/{i:02d}", m.NOMINAL, t
    for i, t in enumerate(SENTIMENT_TEXTS):
        yield f"sentiment/{i:02d}", m.NUMERIC, t


def run_hf(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(24)
    tok = AutoTokenizer.from_pretrained(str(args.hf))
    model = AutoModelForCausalLM.from_pretrained(str(args.hf), dtype=torch.float32, attn_implementation="eager").eval()
    res = {}
    with torch.no_grad():
        for key, task, text in items():
            messages = [{"role": "system", "content": m.system_text(task)}, {"role": "user", "content": m.user_text(text)}]
            ids = tok.encode(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), add_special_tokens=False)
            cand = [tok.encode(r, add_special_tokens=False)[0] for r in task["responses"]]
            logits = model(torch.tensor([ids])).logits[0, -1].to(torch.float64).numpy()
            r = m.reduce_logits(logits, cand)
            res[key] = {"p": r["probabilities"], "log_q": r["candidate_log_probs"], "tokens": ids}
            print(key, [round(x, 3) for x in r["probabilities"]], flush=True)
    (m.OUT / "fidelity-hf-float32.json").write_text(json.dumps(res))


def run_llama(args):
    import native

    lib = native.load()
    eng = m.Engine(lib, args.model, n_gpu_layers=0 if args.cpu else -1)
    template = lib.llama_model_chat_template(eng.model, None).decode()
    tail = m.render_segments(template, m.NOMINAL, "x")[-1][1]
    ref = json.loads((m.OUT / "fidelity-hf-float32.json").read_text())
    cp = lib.llama_context_default_params()
    cp.n_ctx, cp.n_batch, cp.n_ubatch, cp.n_seq_max, cp.no_perf = 4096, 512, args.n_ubatch, 1, True
    cp.flash_attn_type = {"auto": -1, "off": 0, "on": 1}[args.fa]
    cp.type_k = cp.type_v = args.kv
    ctx = lib.llama_init_from_model(eng.model, cp)
    prepared = []
    for key, task, text in items():
        toks = m.tokens_for(eng, m.render_segments(template, task, text))
        assert toks == ref[key]["tokens"], key
        cand = [m.boundary_check(eng, tail, r)["token"]["id"] for r in task["responses"]]
        prepared.append((key, toks, cand))
    eng.decode_prompt(ctx, prepared[0][1], 512)  # warm-up
    res, dps, dls = {}, [], []
    t0 = time.time()
    for key, toks, cand in prepared:
        r = m.reduce_logits(eng.decode_prompt(ctx, toks, 512), cand)
        res[key] = {"p": r["probabilities"], "log_q": r["candidate_log_probs"]}
    elapsed = time.time() - t0
    for key, _, _ in prepared:
        a, b = res[key], ref[key]
        dps.append(max(abs(x - y) for x, y in zip(a["p"], b["p"])))
        dls.append(max((abs(x - y) for x, y in zip(a["log_q"], b["log_q"]) if y > -15), default=0.0))
    dps_sorted = sorted(dps)
    summary = {
        "label": args.label,
        "model": args.model.name,
        "device": "cpu" if args.cpu else "cuda",
        "n_ubatch": args.n_ubatch,
        "fa": args.fa,
        "kv": {0: "f32", 1: "f16"}[args.kv],
        "items": len(prepared),
        "seconds_per_item": round(elapsed / len(prepared), 4),
        "max_abs_dp": round(max(dps), 4),
        "median_abs_dp": round(dps_sorted[len(dps) // 2], 4),
        "p90_abs_dp": round(dps_sorted[int(0.9 * (len(dps) - 1))], 4),
        "max_abs_dlogq": round(max(dls), 4),
        "items_dp_gt_0.05": sum(d > 0.05 for d in dps),
    }
    print(json.dumps(summary))
    with (m.OUT / "fidelity-summary.jsonl").open("a") as f:
        f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    h = sub.add_parser("hf")
    h.add_argument("--hf", type=pathlib.Path, required=True)
    l = sub.add_parser("llama")
    l.add_argument("--model", type=pathlib.Path, required=True)
    l.add_argument("--label", required=True)
    l.add_argument("--cpu", action="store_true")
    l.add_argument("--n-ubatch", type=int, default=512)
    l.add_argument("--fa", default="auto")
    l.add_argument("--kv", type=int, default=1)
    args = ap.parse_args()
    run_hf(args) if args.mode == "hf" else run_llama(args)
