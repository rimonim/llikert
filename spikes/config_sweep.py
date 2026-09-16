"""M0 probe: candidate probabilities for all spike prompts under several llama.cpp
execution configurations, for comparison with the transformers reference.

usage: python spikes/config_sweep.py --model M [--cpu]
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import m0_logits as m  # noqa: E402
import native  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", type=pathlib.Path, required=True)
ap.add_argument("--cpu", action="store_true")
args = ap.parse_args()

lib = native.load()
eng = m.Engine(lib, args.model, n_gpu_layers=0 if args.cpu else -1)
template = lib.llama_model_chat_template(eng.model, None).decode()
tail = m.render_segments(template, m.NOMINAL, "x")[-1][1]

prompts = {}
for task in (m.NOMINAL, m.NUMERIC):
    cand = [m.boundary_check(eng, tail, r)["token"]["id"] for r in task["responses"]]
    for tid, text in m.TEXTS.items():
        prompts[f"{task['name']}/{tid}"] = (m.tokens_for(eng, m.render_segments(template, task, text)), cand)

FA = {"auto": -1, "off": 0, "on": 1}
CONFIGS = {
    "u512_fa_auto_kv16": dict(n_ubatch=512, fa="auto", kv=1),
    "u512_fa_off_kv16": dict(n_ubatch=512, fa="off", kv=1),
    "u512_fa_off_kv32": dict(n_ubatch=512, fa="off", kv=0),
    "u64_fa_auto_kv16": dict(n_ubatch=64, fa="auto", kv=1),
    "u64_fa_off_kv32": dict(n_ubatch=64, fa="off", kv=0),
    "u1_fa_off_kv32": dict(n_ubatch=1, fa="off", kv=0),
}

out = {"model_file": args.model.name, "device": "cpu" if args.cpu else "cuda", "results": {}}
for name, cfg in CONFIGS.items():
    cp = lib.llama_context_default_params()
    cp.n_ctx, cp.n_batch, cp.n_ubatch, cp.n_seq_max, cp.no_perf = 4096, 512, cfg["n_ubatch"], 1, True
    cp.flash_attn_type = FA[cfg["fa"]]
    cp.type_k = cp.type_v = cfg["kv"]
    ctx = lib.llama_init_from_model(eng.model, cp)
    res = {}
    for key, (toks, cand) in prompts.items():
        r = m.reduce_logits(eng.decode_prompt(ctx, toks, 512), cand)
        res[key] = {"p": r["probabilities"], "log_q": r["candidate_log_probs"], "log_coverage": r["log_coverage"]}
    lib.llama_free(ctx)
    out["results"][name] = res
    print("done", name, flush=True)

path = m.OUT / f"config-sweep-{args.model.stem}-{out['device']}.json"
path.write_text(json.dumps(out, indent=2))
print("wrote", path)
