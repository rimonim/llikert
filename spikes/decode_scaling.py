"""M3 probe: wall time of one llama_decode vs tokens in the batch, and multi-sequence packing."""

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python" / "src"))
from llikert.server.llama_adapter import LlamaAdapter, LlamaConfig  # noqa: E402

model = pathlib.Path(sys.argv[1])
a = LlamaAdapter(LlamaConfig(model_path=model))
tokens = (a.tokenize("The quick brown fox jumps over the lazy dog. " * 200, parse_special=False))
for n in (8, 16, 32, 64, 128, 256, 512):
    a.final_logits(tokens[:n])  # warm this shape
    t0 = time.perf_counter()
    reps = 10
    for _ in range(reps):
        a.final_logits(tokens[:n])
    ms = 1000 * (time.perf_counter() - t0) / reps
    print(f"tokens={n:4d} ms/decode={ms:7.2f} ms/token={ms/n:6.3f}")
a.close()
