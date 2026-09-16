"""M3 probe: pack several prompts into one llama_decode (distinct seq_ids, unified KV).

Reports throughput by pack size, deviation from one-prompt-per-decode, and composition
sensitivity: the same prompt scored in two different packs.
"""

import json
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python" / "src"))
from llikert.server import native  # noqa: E402
from llikert.server.llama_adapter import LlamaAdapter, LlamaConfig  # noqa: E402
from llikert.server.numerics import score_candidates  # noqa: E402
from llikert.server.service import ScoringService  # noqa: E402
from llikert.server.task import parse_task  # noqa: E402

model = pathlib.Path(sys.argv[1])
corpus = json.loads((REPO / "benchmarks/corpus-v1.json").read_text())
adapter = LlamaAdapter(LlamaConfig(model_path=model))
service = ScoringService(adapter)
service.smoke_check()
task_obj = corpus["tasks"]["topic-10"]
prepared = service.prepare(task_obj)["prepared"]
task = parse_task(task_obj)
ids = [c["token_id"] for c in prepared["categories"]]
texts = [t for t in corpus["texts"] if t["length"] in ("short", "medium")][:96]
prompts = [service.preparer.prompt_tokens(task, t["text"])[0] for t in texts]

L = adapter.lib.llama
MAX_SEQ = 32


def make_ctx(n_seq):
    cp = L.llama_context_default_params()
    cp.n_ctx, cp.n_batch, cp.n_ubatch, cp.n_seq_max = 4096, 2048, 2048, n_seq
    cp.n_threads = cp.n_threads_batch = 24
    cp.flash_attn_type = native.FLASH_ATTN_TYPES["off"]
    cp.type_k = cp.type_v = native.GGML_TYPES["f32"]
    cp.kv_unified = True
    cp.no_perf = True
    return L.llama_init_from_model(adapter._model, cp)


def packed_logits(ctx, group):
    total = sum(len(p) for p in group)
    batch = L.llama_batch_init(total, 0, 1)
    out_index = []
    try:
        L.llama_memory_clear(L.llama_get_memory(ctx), True)
        k = 0
        for s, p in enumerate(group):
            for i, tok in enumerate(p):
                batch.token[k], batch.pos[k], batch.n_seq_id[k] = tok, i, 1
                batch.seq_id[k][0] = s
                batch.logits[k] = 1 if i == len(p) - 1 else 0
                k += 1
            out_index.append(k - 1)
        batch.n_tokens = total
        rc = L.llama_decode(ctx, batch)
        assert rc == 0, rc
        return [np.ctypeslib.as_array(L.llama_get_logits_ith(ctx, j), shape=(adapter.n_vocab,)).astype(np.float64) for j in out_index]
    finally:
        L.llama_batch_free(batch)


ctx = make_ctx(MAX_SEQ)
reference = {}
t0 = time.perf_counter()
for i, p in enumerate(prompts):
    reference[i] = score_candidates(adapter.final_logits(p), ids).probabilities
seq_rate = len(prompts) / (time.perf_counter() - t0)
print(f"sequential (service path): {seq_rate:.1f} items/s")

results = {}
for pack in (1, 4, 8, 16):
    for offset in (0, pack // 2):
        packed_logits(ctx, prompts[:pack])  # warm
        t0 = time.perf_counter()
        scores = {}
        order = list(range(offset, len(prompts))) + list(range(offset))
        for g in range(0, len(order), pack):
            group = order[g : g + pack]
            for idx, logits in zip(group, packed_logits(ctx, [prompts[i] for i in group])):
                scores[idx] = score_candidates(logits, ids).probabilities
        rate = len(prompts) / (time.perf_counter() - t0)
        results[(pack, offset)] = scores
        dev = max(max(abs(a - b) for a, b in zip(scores[i], reference[i])) for i in scores)
        print(f"pack={pack:2d} offset={offset:2d}: {rate:6.1f} items/s  max|dp| vs sequential={dev:.5f}")
    if pack > 1:
        a, b = results[(pack, 0)], results[(pack, pack // 2)]
        comp = max(max(abs(x - y) for x, y in zip(a[i], b[i])) for i in a)
        print(f"pack={pack:2d}: composition sensitivity (same item, different pack) max|dp|={comp:.5f}")
L.llama_free(ctx)
adapter.close()
