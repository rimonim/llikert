# LLikert

Research-grade candidate-probability scoring: for each text in a dataset, the next-token probabilities a causal language model assigns to explicitly specified response codes — both their original probability mass and the distribution conditional on choosing one of them.

**Status:** pre-release (v0.1 development). The scoring service (M1), the R and Python clients (M2), and the benchmarked, fidelity-gated execution path (M3) work locally; packaging and hosting (M4) are next.

- `llikert_development_plan.md` — product specification
- `docs/implementation-plan.md` — detailed plan and decisions
- `docs/decisions/` — decision records (M0 feasibility, M1 service, M2 clients and CUDA toolchain, M3 performance and fidelity)
- `docs/protocol.md` — HTTP protocol v1
- `docs/local-service.md` — running the service
- `r/llikert/README.md`, `python/README.md` — client quickstarts
- `docs/checkpoint-format.md` — result and checkpoint files
- `docs/compatibility.md` — what is verified, and on what evidence
- `docs/benchmark-report.md` — throughput, latency, memory, fidelity of the supported profile

License: MIT.
