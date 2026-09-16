"""M0 probe: deviation of llama.cpp configurations from the transformers float64 reference."""
import json, math, pathlib, sys
OUT = pathlib.Path(__file__).parent / "out"
ref = json.loads((OUT / "hf-reference-qwen3-4b-instruct-2507.json").read_text())["results"]
ref64, ref32 = ref["float64"], ref["float32"]

def dev(res, base, floor=-15.0):
    dp = dl = 0.0
    for key, r in base.items():
        x = res[key]
        dp = max(dp, max(abs(a - b) for a, b in zip(x["p"], r["p"])))
        # log_q compared only where the reference token is not negligible
        dl = max(dl, max((abs(a - b) for a, b in zip(x["log_q"], r["log_q"]) if b > floor), default=0.0))
    return dp, dl

def fmt(name, res):
    dp, dl = dev(res, ref64)
    t2 = res["Communicative function/t2"]["p"][1]; s3 = res["Sentiment/t3"]["p"][3]
    print(f"{name:34s} max|dp|={dp:.4f} max|dlogq|={dl:.3f}  t2 p(B)={t2:.3f}  sent-t3 p(4)={s3:.3f}")

print(f"reference float64: t2 p(B)={ref64['Communicative function/t2']['p'][1]:.4f} sent-t3 p(4)={ref64['Sentiment/t3']['p'][3]:.4f}")
fmt("hf float32", ref32)
for dev_name in ("cuda", "cpu"):
    sweep = json.loads((OUT / f"config-sweep-qwen3-4b-instruct-2507-q8_0-{dev_name}.json").read_text())["results"]
    for name, res in sweep.items():
        fmt(f"q8 {dev_name} {name}", res)
