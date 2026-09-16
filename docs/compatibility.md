# Compatibility matrix

"Supported" means the evidence below was produced on the pinned stack. Every environment or recipe without that evidence is listed as **not verified**. Last updated 2026-09-16.

**Pinned stack:**
- llama.cpp `v0.4.1` (commit `b29c606`)
- llama-cpp-python `0.3.35`, built from source with `tools/build-llama-cpp-python.sh` and linked against **CUDA 12.9 / cuBLAS 12.9** (decision 0004 A)
- CPython 3.10 or newer, not conda (finding F2)

## Model profiles

A profile is weights, device, math libraries and execution settings together. Decision D24 requires the reference-fidelity gate: a comparison with transformers float32 on the official weights over the 28 items in `tests/fixtures/fidelity/`, with max |Δp| ≤ 0.005 and max |Δlog q| ≤ 0.1. Run it with `tools/fidelity_gate.py`.

| Profile | Weights (sha256) | Device and settings | Gate | Status |
|---|---|---|---|---|
| **Qwen3-4B-Instruct-2507, F32** | `qwen3-4b-instruct-2507-f32.gguf` (`a5733d5a…e357`) | CUDA 12.9, cuBLAS 12.9, RTX 4090; `--batch-size 512 --flash-attn off --kv-type f32` (defaults) | **Passed**: max \|Δp\| 0.0044, max \|Δlog q\| 0.045 | Supported |
| Qwen3-4B-Instruct-2507, F32 | same | CUDA 12.9 with flash attention auto or on, or KV f16 | Failed (max \|Δlog q\| 0.10–0.12) | Not supported |
| Qwen3-4B-Instruct-2507, F32 | same | CUDA 12.9, batch 1, FA off, KV f32 | Passed (max \|Δp\| 0.00001), about 1.2 s per item | Valid but slow; not a default |
| Qwen3-4B-Instruct-2507, F32 | same | CPU, FA off, KV f32 | M0 spike only (max \|Δp\| < 1e-4), not through the service | Not verified |
| Qwen3-4B-Instruct-2507, Q8_0 | `ae916ede…d5f1` | any | Failed in M0 | Not supported (D23) |
| Qwen2.5-0.5B-Instruct, Q8_0 | `ca59ca7f…844e` | CUDA | No fidelity claim | Smoke and CI model only |

M0 and M1 CUDA measurements were made with a wheel that accidentally linked cuBLAS 11.7. Those results are superseded by the rows above (decision 0004 A).

## Template profiles

| Profile | Template sha256 | Rendering verified against transformers |
|---|---|---|
| `qwen3-instruct-2507-chatml` | `64f85b19…c326` | Yes: 10 texts, byte-identical |
| `qwen2.5-instruct-chatml` | `d5495a1e…a9a4` | Yes: 10 texts, byte-identical |

## Service environments

| Environment | Status | Evidence |
|---|---|---|
| Linux x86_64, NVIDIA driver 575.51, CUDA 12.9, RTX 4090 (local) | Verified | Unit and integration suites; end-to-end runs with both clients |
| Container `nvidia/cuda:12.9.0-runtime-ubuntu22.04` with the NVIDIA runtime | Probe verified | Model loaded from a `/repository` mount, CUDA offload, `/health` 200; the full service image is not built yet (M4) |
| Linux x86_64, CPU-only build | Not verified through the service | M0 spikes only |
| Hugging Face Inference Endpoint | Not verified | Deferred by owner |
| Windows, macOS, Apple Metal inference | Not supported in v0.1 | — |

## Client environments

| Client | Environment | Status | Evidence |
|---|---|---|---|
| Python (`pip install llikert`, base) | Linux, CPython 3.10, only `httpx` installed | Verified | 63 client tests in a clean venv |
| R (`llikert`) | Linux, R 4.6.1, no Python (`rocker/r-ver:4.6.1`) | Verified | `R CMD check --no-manual`: Status OK |
| R and Python against a real service | Linux, local F32 service | Verified | Identical saved results from both clients |
| R and Python clients | Windows, macOS | Not verified | — |
