# Compatibility matrix

"Supported" means the evidence below was produced on the pinned stack. Every environment or recipe without that evidence is listed as **not verified**. Last updated 2026-09-16.

**Pinned stack:** llama.cpp `v0.4.1` (commit `b29c606`); llama-cpp-python `0.3.35`, built from source with `tools/build-llama-cpp-python.sh`; CPython 3.10 (not conda, see finding F2).

## Model profiles

A profile is weights, device and execution settings together. Decision D24 requires the reference-fidelity gate: a comparison with transformers float32 on the official weights over the 28 items in `tests/fixtures/fidelity/`, with tolerances max |Δp| ≤ 0.005 and max |Δlog q| ≤ 0.1.

| Profile | Weights (sha256) | Device and settings | Fidelity gate | Status |
|---|---|---|---|---|
| **Qwen3-4B-Instruct-2507, F32** | `qwen3-4b-instruct-2507-f32.gguf` (`a5733d5a…e357`) | CUDA (RTX 4090), `--batch-size 512 --flash-attn auto --kv-type f16` | **Passed** (max \|Δp\| 0.00083; max \|Δlog q\| 0.063) | Supported |
| Qwen3-4B-Instruct-2507, F32 | same | CPU, FA off, KV f32 | Measured in M0 spike only (max \|Δp\| < 1e-4), not through the service | Not verified |
| Qwen3-4B-Instruct-2507, Q8_0 | `ae916ede…d5f1` | any | Failed in M0 (max \|Δp\| up to 0.13 on the 28 items) | Not supported (D23) |
| Qwen2.5-0.5B-Instruct, Q8_0 | `ca59ca7f…844e` | CUDA | No fidelity claim | Smoke and CI model only |

## Template profiles

| Profile | Template sha256 | Rendering verified against transformers |
|---|---|---|
| `qwen3-instruct-2507-chatml` | `64f85b19…c326` | Yes: 10 texts, byte-identical |
| `qwen2.5-instruct-chatml` | `d5495a1e…a9a4` | Yes: 10 texts, byte-identical |

## Environments

| Environment | Status | Evidence |
|---|---|---|
| Linux x86_64, NVIDIA CUDA 12.9, RTX 4090 (local service) | Verified | Unit and integration suites; end-to-end CLI run |
| Linux x86_64, CPU-only build (`GGML_NATIVE=on`) | Partially verified | M0 spikes only; the service has not run on it |
| CPU-only portable build (`tools/build-llama-cpp-python.sh cpu`) | Not verified | Build script written, never executed |
| Container image (CUDA runtime) | Builds; GPU run not verified | The host lacks `nvidia-container-toolkit` |
| Hugging Face Inference Endpoint | Not verified | Deferred by owner |
| Python client (M2), R client (M2) | Not yet implemented | — |
| Windows, macOS, Apple Metal inference | Not supported in v0.1 | — |
