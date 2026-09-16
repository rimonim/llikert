# Benchmark report (v0.1 development, 2026-09-16)

Written for: operators planning dataset runs and maintainers comparing execution profiles. These are measurements on one machine, not guarantees.

## Setup

| Item | Value |
|---|---|
| Hardware | NVIDIA RTX 4090 (24 GB), AMD Ryzen 9 7900X, 187 GB RAM, Linux |
| Stack | llama.cpp v0.4.1, llama-cpp-python 0.3.35 (source build), CUDA 12.9 / cuBLAS 12.9, driver 575.51.03 |
| Model | Qwen3-4B-Instruct-2507, F32 GGUF (`a5733d5a…e357`) |
| Profile | Supported defaults: `--batch-size 512 --flash-attn off --kv-type f32 --n-ctx 4096`, TF32 disabled |
| Corpus | `benchmarks/corpus-v1.json`: 300 short, 150 medium and 50 long synthetic texts, per task |
| Tasks | binary (K = 2), sentiment (K = 5, three few-shot examples), topic (K = 10) |
| Command | `.venv/bin/python benchmarks/run.py --model models/qwen3-4b-instruct-2507-f32.gguf --label supported-f32-cuda --http` |

Raw results: `benchmarks/results/supported-f32-cuda.json`.

## Throughput and latency (warm, sequential, direct service calls)

| Task | Items/s | Short: prompt tokens / median ms / p95 ms | Medium | Long |
|---|---|---|---|---|
| binary-food | 18.7 | 81 / 36 / 37 | 144 / 53 / 55 | 619 / 165 / 190 |
| sentiment-5 | 14.2 | 160 / 55 / 56 | 223 / 64 / 65 | 698 / 183 / 212 |
| topic-10 | 17.2 | 101 / 41 / 42 | 164 / 55 / 57 | 639 / 167 / 204 |

- **Native evaluation:** one decode call per item, plus one more per long item, which spans two 512-token micro-batches.
- **Time per decode:** roughly fixed for short prompts, 21 ms at 8 tokens and 27 ms at 128, rising above that.
- **Planning figure:** a run of 10,000 short texts takes about 10 minutes.

## Startup and memory

| Measure | Value |
|---|---|
| SHA-256 of the 16 GB model file | 8.7 s |
| Model load, context creation, full-context warm-up, smoke check | 13.1 s |
| GPU memory (process) | 17.3 GB |
| Peak host RSS | 15.7 GB, mostly the memory-mapped model file |

## HTTP overhead

The same 128 short texts, in chunks of 16, through the HTTP service and the Python client took 4.667 s, against 4.642 s scored directly: **0.19 ms per item**.

## Fidelity of this profile

This is the D24 gate against a transformers float32 reference.

| Fixture | Items | max \|Δp\| | max \|Δlog q\| |
|---|---|---|---|
| `tests/fixtures/fidelity/qwen3-4b-instruct-2507.json` | 28 | 0.000012 | 0.00019 |
| `tests/fixtures/fidelity/qwen3-4b-instruct-2507-v2.json` | 450 | 0.000093 | 0.0011 |
| Benchmark corpus (`benchmarks/results/fidelity-corpus-v1-supported.json`) | 1,500 | 0.0013 | 0.016 |

The tolerances are 0.005 and 0.1.

## Profiles and optimizations that were measured and not adopted

The details are in `docs/decisions/0005-m3-performance-and-fidelity.md`.

| Variant | Items/s (binary / sentiment / topic) | Why not adopted |
|---|---|---|
| TF32 enabled (ggml default) | 27.6 / 22.8 / 26.6 | Fails the fidelity gate (max \|Δp\| 0.027) |
| Shared-prefix reuse (TF32 off) | 23.5 / 23.1 / 23.2 | Fails the gate on long texts (max \|Δp\| 0.0063) |
| Multi-sequence packing, 8 prompts per decode (TF32 off; short and medium topic texts) | about 1.5× sequential | An item's result depends on its batchmates (up to 2e-4), so it is not bitwise reproducible |

## Reproducing

```bash
python3 benchmarks/corpus.py --out benchmarks/corpus-v1.json            # deterministic (seed 20260916)
.venv/bin/python benchmarks/run.py --model <f32.gguf> --label <label> --http
.venv-ref/bin/python benchmarks/reference.py --hf <official weights> --corpus benchmarks/corpus-v1.json --out build/reference-corpus-v1.json
.venv/bin/python tools/fidelity_gate.py --model <f32.gguf> --fixture build/reference-corpus-v1.json
```
