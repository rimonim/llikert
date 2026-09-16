# LLikert

Research-grade candidate-probability scoring: for each text in a dataset, the next-token probabilities a causal language model assigns to explicitly specified response codes — both their original probability mass and the distribution conditional on choosing one of them.

**Status:** pre-release (v0.1 development). The scoring service (M1) works locally; the R and Python clients (M2) are in progress.

- `llikert_development_plan.md` — product specification
- `docs/implementation-plan.md` — detailed plan and decisions
- `docs/decisions/` — decision records (M0 feasibility findings, M1 service)
- `docs/protocol.md` — HTTP protocol v1
- `docs/local-service.md` — running the service
- `docs/compatibility.md` — what is verified, and on what evidence

License: MIT.
