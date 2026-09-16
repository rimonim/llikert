# M0 spikes (deleted in M5)

Feasibility probes behind `docs/decisions/0002-m0-findings.md`. Not product code.

| Script | Purpose |
|---|---|
| `abi/abi_offsets.c`, `abi_check.py` | C `sizeof`/`offsetof` vs ctypes layouts (llikert's and the binding's) |
| `native.py` | llikert's own ctypes declarations for libllama v0.4.1 |
| `m0_logits.py` | rendering vs transformers, segment tokenization, boundary tokens, section 4 math, state/limit/pointer probes |
| `batch_probe.py`, `config_sweep.py` | execution-setting sensitivity of final-position logits |
| `hf_reference.py`, `fidelity.py` | transformers float64/float32 reference and profile fidelity + latency |
| `compare_reference.py`, `compare_sweeps.py`, `summarize.py` | report helpers |
| `container/` | CUDA runtime Dockerfile + `/health` stub (image build not executed) |

Environments: `.venv` (system Python 3.10, CUDA wheel from `build/wheels/cuda`),
`.venv-cpu` (CPU wheel, CPU torch, transformers, gguf-py for conversions).
