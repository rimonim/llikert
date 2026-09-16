"""M0 probe: ground-truth final-position candidate probabilities from the official
unquantized weights with transformers (eager attention, float64 and float32, CPU).

Uses the same rendered prompts and token ids as the llama.cpp spike (segment
tokenization equals HF tokenization for these texts, verified in m0_logits.py).

usage: .venv-cpu/bin/python spikes/hf_reference.py --hf models/hf/Qwen3-4B-Instruct-2507
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import m0_logits as m  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--hf", type=pathlib.Path, required=True)
ap.add_argument("--dtypes", default="float64,float32")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(str(args.hf))
template = tok.chat_template
torch.set_num_threads(24)

prompts = {}
for task in (m.NOMINAL, m.NUMERIC):
    for tid, text in m.TEXTS.items():
        messages = [{"role": "system", "content": m.system_text(task)}, {"role": "user", "content": m.user_text(text)}]
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok.encode(rendered, add_special_tokens=False)
        cand = [tok.encode(r, add_special_tokens=False)[0] for r in task["responses"]]
        prompts[f"{task['name']}/{tid}"] = (ids, cand)

out = {"model_dir": str(args.hf), "torch": torch.__version__, "results": {}}
for dtype_name in args.dtypes.split(","):
    dtype = getattr(torch, dtype_name)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(str(args.hf), dtype=dtype, attn_implementation="eager")
    model.eval()
    res = {}
    with torch.no_grad():
        for key, (ids, cand) in prompts.items():
            logits = model(torch.tensor([ids])).logits[0, -1].to(torch.float64).numpy()
            r = m.reduce_logits(logits, cand)
            res[key] = {"p": r["probabilities"], "log_q": r["candidate_log_probs"], "log_coverage": r["log_coverage"], "n_tokens": len(ids)}
            print(dtype_name, key, [round(x, 4) for x in r["probabilities"]], flush=True)
    out["results"][dtype_name] = res
    out[f"{dtype_name}_seconds"] = round(time.time() - t0, 1)
    del model

path = m.OUT / "hf-reference-qwen3-4b-instruct-2507.json"
path.write_text(json.dumps(out, indent=2))
print("wrote", path)
