"""M0 probe: which execution settings change final-position logits for one fixed prompt?

usage: python spikes/batch_probe.py --model M --hf H [--cpu] [--label L]
Env vars such as GGML_CUDA_FORCE_MMQ / GGML_CUDA_FORCE_CUBLAS can be set by the caller.
"""

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import m0_logits as m  # noqa: E402
import native  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", type=pathlib.Path, required=True)
ap.add_argument("--cpu", action="store_true")
ap.add_argument("--label", default="")
args = ap.parse_args()

lib = native.load()
eng = m.Engine(lib, args.model, n_gpu_layers=0 if args.cpu else -1)
template = lib.llama_model_chat_template(eng.model, None).decode()
toks = m.tokens_for(eng, m.render_segments(template, m.NOMINAL, m.TEXTS["t2"]))
ids = [32, 33, 34]
FA = {"auto": -1, "off": 0, "on": 1}


def run(n_batch, n_ubatch, fa="auto", chunk=None, kv=None):
    cp = lib.llama_context_default_params()
    cp.n_ctx, cp.n_batch, cp.n_ubatch, cp.n_seq_max, cp.no_perf = 4096, n_batch, n_ubatch, 1, True
    cp.flash_attn_type = FA[fa]
    if kv is not None:  # ggml_type: 0 = F32, 1 = F16
        cp.type_k = cp.type_v = kv
    ctx = lib.llama_init_from_model(eng.model, cp)
    try:
        return eng.decode_prompt(ctx, toks, chunk or n_batch)
    finally:
        lib.llama_free(ctx)


configs = [
    ("b512_u512", dict(n_batch=512, n_ubatch=512)),
    ("b512_u512_rerun", dict(n_batch=512, n_ubatch=512)),
    ("b512_u64", dict(n_batch=512, n_ubatch=64)),
    ("b64_u64", dict(n_batch=64, n_ubatch=64)),
    ("b512_u1", dict(n_batch=512, n_ubatch=1)),
    ("b1_u1", dict(n_batch=1, n_ubatch=1)),
    ("b512_u512_chunk64", dict(n_batch=512, n_ubatch=512, chunk=64)),
    ("b512_u512_fa_off", dict(n_batch=512, n_ubatch=512, fa="off")),
    ("b1_u1_fa_off", dict(n_batch=1, n_ubatch=1, fa="off")),
    ("b64_u64_fa_off", dict(n_batch=64, n_ubatch=64, fa="off")),
    ("b512_u512_kv32", dict(n_batch=512, n_ubatch=512, kv=0)),
    ("b512_u64_kv32", dict(n_batch=512, n_ubatch=64, kv=0)),
    ("b512_u1_kv32", dict(n_batch=512, n_ubatch=1, kv=0)),
    ("b512_u512_kv32_fa_off", dict(n_batch=512, n_ubatch=512, kv=0, fa="off")),
    ("b512_u1_kv32_fa_off", dict(n_batch=512, n_ubatch=1, kv=0, fa="off")),
]
ref = None
out = {"label": args.label, "device": "cpu" if args.cpu else "cuda", "n_prompt_tokens": len(toks), "configs": {}}
for name, kw in configs:
    lg = run(**kw)
    r = m.reduce_logits(lg, ids)
    if ref is None:
        ref = lg
    out["configs"][name] = {
        "p": [round(x, 5) for x in r["probabilities"]],
        "log_q": [round(x, 4) for x in r["candidate_log_probs"]],
        "max_abs_logit_vs_first": float(np.max(np.abs(lg - ref))),
    }
    print(f"{name:22s} p={out['configs'][name]['p']} dlogit={out['configs'][name]['max_abs_logit_vs_first']:.4f}", flush=True)

path = m.OUT / f"batch-probe-{args.model.stem}-{out['device']}{('-' + args.label) if args.label else ''}.json"
path.write_text(json.dumps(out, indent=2))
print("wrote", path)
