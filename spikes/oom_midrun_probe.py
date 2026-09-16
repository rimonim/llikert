"""M3 probe: does a full-context prompt abort mid-run when the KV cache fits but compute memory does not?"""
import pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python" / "src"))
from llikert.server.llama_adapter import LlamaAdapter, LlamaConfig
n_ctx = int(sys.argv[2])
a = LlamaAdapter(LlamaConfig(model_path=pathlib.Path(sys.argv[1]), n_ctx=n_ctx))
print("context created", flush=True)
tokens = a.tokenize("word " * (n_ctx + 100), parse_special=False)[: n_ctx - 8]
t0 = time.time()
logits = a.final_logits(tokens)
print(f"full-context prompt of {len(tokens)} tokens scored in {time.time() - t0:.1f}s", flush=True)
a.close()
