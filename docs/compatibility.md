# Compatibility matrix

"Supported" means the evidence below was produced on the pinned stack. Every environment or recipe without that evidence is listed as **not verified**. Last updated 2026-09-16.

**Pinned stack:**
- llama.cpp `v0.4.1` (commit `b29c606`)
- llama-cpp-python `0.3.35`, built from source with `tools/build-llama-cpp-python.sh` and linked against **CUDA 12.9 / cuBLAS 12.9** (decision 0004 A)
- CPython 3.10 or newer, not conda (finding F2)

## Model profiles

A profile is weights, device, math libraries and execution settings together. Decision D24 requires the reference-fidelity gate: a comparison with transformers float32 (TF32 off) on the official weights, with max |Δp| ≤ 0.005 and max |Δlog q| ≤ 0.1. The fixtures are `tests/fixtures/fidelity/` (28 and 450 items); the benchmark corpus adds 1,500 items. Run it with `tools/fidelity_gate.py`.

| Profile | Weights (sha256) | Device and settings | Gate | Status |
|---|---|---|---|---|
| **Qwen3-4B-Instruct-2507, F32** | `qwen3-4b-instruct-2507-f32.gguf` (`a5733d5a…e357`) | CUDA 12.9, cuBLAS 12.9, RTX 4090; **TF32 disabled** (enforced by the service); `--batch-size 512 --flash-attn off --kv-type f32` (defaults) | **Passed**: 28 items 0.000012 / 0.00019; 450 items 0.000093 / 0.0011; 1,500 items 0.0013 / 0.016 (max \|Δp\| / max \|Δlog q\|) | Supported |
| Qwen3-4B-Instruct-2507, F32 | same | CUDA with TF32 enabled (the ggml default before 0005), any settings | Failed (450 items: max \|Δp\| 0.027) | Not supported |
| Qwen3-4B-Instruct-2507, F32 | same | CUDA, TF32 disabled, flash attention auto/on, or KV f16 | Failed (450 items: max \|Δp\| 0.017–0.031) | Not supported |
| Qwen3-4B-Instruct-2507, F32 | same | CPU, FA off, KV f32 | M0 spike only (28 items, max \|Δp\| < 1e-4), not through the service | Not verified |
| Qwen3-4B-Instruct-2507, Q8_0 | `ae916ede…d5f1` | any | Failed in M0 | Not supported (D23) |
| Qwen2.5-0.5B-Instruct, Q8_0 | `ca59ca7f…844e` | CUDA | No fidelity claim | Smoke, CI and stress-test model only |

Earlier CUDA measurements are superseded: M0 and M1 used a wheel that linked cuBLAS 11.7 (0004 A), and M2 ran with TF32 enabled (0005 B).

Performance of the supported profile is in `docs/benchmark-report.md`: 14–19 items/s on the synthetic corpus, and 17.3 GB of GPU memory.

## Template profiles

| Profile | Template sha256 | Rendering verified against transformers |
|---|---|---|
| `qwen3-instruct-2507-chatml` | `64f85b19…c326` | Yes: 10 texts, byte-identical |
| `qwen2.5-instruct-chatml` | `d5495a1e…a9a4` | Yes: 10 texts, byte-identical |

## Service environments

| Environment | Status | Evidence |
|---|---|---|
| Linux x86_64, NVIDIA driver 575.51, CUDA 12.9, RTX 4090 (local) | Verified | Unit and integration suites; end-to-end runs with both clients |
| Service image `llikert-service:0.1.0.dev0-cuda12.9` (`--gpus all`, RTX 4090) | Verified | `deploy/test-container.sh`: 16/16 checks, including the HTTP fidelity gate; same engine fingerprint as the host service (decision 0006) |
| Linux x86_64, CPU-only build | Not verified through the service | M0 spikes only |
| Hugging Face Inference Endpoint (`deploy/hf-endpoint.md`) | Not verified (experimental recipe) | Deferred by owner; no image, model or endpoint published |
| NVIDIA L4, A10G, or any GPU other than RTX 4090 | Not verified | Image compiled for compute capability 8.6 and 8.9; run `tools/fidelity_gate_http.py` before research use |
| Windows, macOS, Apple Metal inference | Not supported in v0.1 | — |

## Client environments

| Client | Environment | Status | Evidence |
|---|---|---|---|
| Python (`pip install llikert`, base) | Linux, CPython 3.10, only `httpx` installed | Verified | 63 client tests in a clean venv |
| R (`llikert`) | Linux, R 4.6.1, no Python (`rocker/r-ver:4.6.1`) | Verified | `R CMD check --no-manual`: Status OK |
| R and Python against a real service | Linux, local F32 service | Verified | Identical saved results from both clients |
| R and Python clients | Windows, macOS | Not verified | — |
