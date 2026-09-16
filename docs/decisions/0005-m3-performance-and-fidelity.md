# 0005: M3 — TF32, the supported execution path, rejected optimizations

**Status:** accepted 2026-09-16
**Context:** M3 had three jobs: check the thin fidelity margin reported in 0004, benchmark the service, and add batching or prefix optimizations only where justified. The measurements below use Qwen3-4B-Instruct-2507 F32 on an RTX 4090 with CUDA 12.9 and cuBLAS 12.9.

## A. Larger fidelity corpus and a GPU reference

- **Corpus.** `benchmarks/corpus.py` generates a deterministic, nonpolitical corpus of short, medium and long texts (up to about 700 prompt tokens). It covers three tasks: binary (K = 2), sentiment (K = 5, numeric, three few-shot examples) and topic (K = 10).
- **Reference.** `benchmarks/reference.py` computes the reference with transformers float32 (eager attention, TF32 off) on a GPU. Prompts are rendered and tokenized by transformers, not llama.cpp.
- **Reference validation.** On the original 28-item fixture, the GPU reference matches the CPU reference to max |Δp| 1.6e-5 and max |Δlog q| 2.4e-4, two orders of magnitude below the tolerances.
- **New fixture.** `tests/fixtures/fidelity/qwen3-4b-instruct-2507-v2.json` has 450 items: 150 texts × 3 tasks. The integration gate now runs both fixtures. The 1,500-item benchmark corpus is checked with the same tool (its reference is regenerated, not committed).

## B. Finding: ggml enables TF32 in cuBLAS

The profile accepted in 0004 (flash attention off, KV f32, batch 512) **failed** the 450-item gate: max |Δp| 0.027 and max |Δlog q| 0.22, with deviations even on short texts. No batch size from 64 to 2048 passed.

- **Cause.** ggml's CUDA backend calls `cublasSetMathMode(..., CUBLAS_TF32_TENSOR_OP_MATH)` on its cuBLAS handles (`ggml-cuda/common.cuh`). On Ampere and newer GPUs, the matrix products of batched prompt evaluation therefore run in TF32 despite F32 weights. Single-token evaluation uses custom kernels, which is why batch size 1 matched the reference in M0.
- **Fix.** `NVIDIA_TF32_OVERRIDE=0` disables TF32 process-wide. `llikert.server.native` sets it before the native libraries load, so an operator cannot forget it, and the engine identity records `cuda_tf32: disabled`.

Gate results on the 450-item fixture, all at batch 512:

| TF32 | Flash attention | KV cache | max \|Δp\| | max \|Δlog q\| | ms/item | Gate |
|---|---|---|---|---|---|---|
| on (ggml default) | off | f32 | 0.0267 | 0.217 | 42 | fail |
| **off** | **off** | **f32** | **0.00009** | **0.0011** | **66** | **pass** |
| off | auto | f16 | 0.0305 | 0.172 | 62 | fail |
| off | off | f16 | 0.0167 | 0.127 | 65 | fail |
| off | auto | f32 | 0.0305 | 0.172 | 63 | fail (flash attention appears to force f16 attention) |

On the full 1,500-item corpus the supported configuration passes with max |Δp| 0.0013 and max |Δlog q| 0.016; the worst cases are long texts, and short and medium texts stay around 1e-4. On the original 28-item fixture it now reaches max |Δp| 1.2e-5.

**Decision (refines 0004 A).** The supported execution path is sequential evaluation with TF32 disabled, `--flash-attn off --kv-type f32 --batch-size 512`. Disabling TF32 costs about 1.5× in speed, a price worth paying for a correct measurement.

## C. Rejected optimizations

Per-decode cost is roughly fixed for short prompts: about 21 ms at 8 tokens and 27 ms at 128 tokens. Two optimizations were measured on the corpus.

| Optimization | Throughput vs sequential | Fidelity or reproducibility | Decision |
|---|---|---|---|
| **Shared-prefix reuse** (system prompt and examples evaluated once, KV state reused, suffix evaluated per item) | 14–19 → 23 items/s (1.2–1.6×) | Order-independent (bitwise identical across orders), but **fails the gate on the 1,500-item corpus**: long texts reach max \|Δp\| 0.0063 | Rejected; the code was removed |
| **Multi-sequence packing** (several prompts per decode, unified KV) | 25.6 → 39 items/s at 8 prompts per decode (1.5×) | Within tolerance of sequential (max \|Δp\| 5e-4), but an item's result depends on its batchmates (up to 2e-4 between packings), so results are no longer bitwise reproducible across chunk sizes, orderings or resumes | Rejected for v0.1; not implemented in the service |

The sequential path guarantees, and tests on a real model, that an item's result is bitwise identical regardless of chunk size, order, repetition, neighboring items and process restarts. The optimizations above would trade that guarantee, or fidelity, for at most 1.6×. That doesn't meet "only where justified". Packing could be reconsidered later as an explicit, fingerprinted trade-off.

Earlier measurements with TF32 on showed larger effects for both optimizations (prefix vs sequential up to 0.031 in `p`; packing composition sensitivity up to 0.022). TF32 was the dominant source of batch-shape sensitivity.

## D. Memory and failure behavior

- **Unfittable context.** llama.cpp reserves the KV cache and worst-case compute buffers when the context is created. With the F32 model on 24 GB, `--n-ctx 131072` (KV cache) and `--n-ctx 24576` (compute buffer) both fail cleanly at startup with `AdapterStartupError`, and `selfcheck` reports them.
- **Largest context that fits.** `--n-ctx 20480` loaded and scored a 20,472-token prompt in 10.5 s.
- **Startup warm-up.** The adapter now evaluates one full-context prompt at startup, so a memory problem surfaces before `/health` reports ready. It adds under a second at the default `--n-ctx 4096`.
- **Crash risk.** A CUDA out-of-memory error during evaluation aborts the process instead of returning an error; this was seen with a second, oversized context in a probe. The single-context service never creates additional contexts.

## E. Stress evidence (`tests/integration/test_stress.py`, small model)

- An item's result is bitwise identical after long items, after `context_limit` failures, across chunk sizes 1, 3 and 7 with shuffled order, across three repeated requests, and in a freshly started process.
- A prompt of exactly `n_ctx` tokens scores.
- Four concurrent HTTP clients against a two-slot queue all complete through 429 retries, with results bitwise identical to direct scoring.

## F. Defaults

- **Request chunk size:** stays 16, capped at `max_items_per_request` 64. A 16-item chunk of long texts takes about 3.4 s and a 64-item chunk about 13 s, well below typical proxy timeouts.
- **HTTP overhead:** 0.2 ms per item, so larger chunks bring no benefit.
