"""M0 probe: all config sweeps vs the transformers float64 reference."""
import json, pathlib
OUT = pathlib.Path(__file__).parent / "out"
ref = json.loads((OUT / "hf-reference-qwen3-4b-instruct-2507.json").read_text())["results"]["float64"]

def dev(res, floor=-15.0):
    dp = dl = 0.0
    for key, r in ref.items():
        x = res[key]
        dp = max(dp, max(abs(a - b) for a, b in zip(x["p"], r["p"])))
        dl = max(dl, max((abs(a - b) for a, b in zip(x["log_q"], r["log_q"]) if b > floor), default=0.0))
    return dp, dl

rows = []
for weights in ("f32", "own-f16", "own-q8_0", "q8_0"):
    for d in ("cpu", "cuda"):
        sweep = json.loads((OUT / f"config-sweep-qwen3-4b-instruct-2507-{weights}-{d}.json").read_text())["results"]
        for name, res in sweep.items():
            dp, dl = dev(res)
            print(f"{weights:9s} {d:4s} {name:20s} max|dp|={dp:.5f} max|dlogq|={dl:.4f}  t2 p(B)={res['Communicative function/t2']['p'][1]:.4f}  sent-t3 p(4)={res['Sentiment/t3']['p'][3]:.4f}")
