# Benchmarks

- `corpus.py`: deterministic synthetic corpus (`corpus-v1.json`) with tasks at K = 2, 5 and 10
- `run.py`: throughput, latency by text length, native decode calls, startup, memory, HTTP overhead
- `reference.py`: transformers float32 reference for fidelity fixtures and corpus checks (needs a torch environment)
- `results/`: committed summaries (per-item scores are written under `build/benchmarks/`)

The report is `docs/benchmark-report.md`; the decisions are `docs/decisions/0005-m3-performance-and-fidelity.md`.
