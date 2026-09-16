# 0003: M1 scoring-service implementation decisions

**Status:** accepted 2026-09-16. The CUDA evidence and default execution settings are superseded by 0004 A (CUDA toolchain correction).
**Context:** These are routine implementation choices made while building M1, recorded so the clients (M2) and later milestones can rely on them. None changes the probability semantics. Where a choice refines a D-numbered decision, that decision is named.

## Protocol and artifacts

1. **A score request carries only `prepared`, with no separate "expected fingerprint" field (refines D4).**
   - The prepared artifact already contains `engine_fingerprint`, so a second copy could only disagree.
   - The service checks that fingerprint first and returns `409 engine_fingerprint_mismatch`.
   - It then re-prepares the embedded task (LRU-cached), compares the whole artifact after JCS canonicalization, and returns `409 prepared_task_mismatch` on any difference.
2. **Prepared artifact v1 contents:**
   - `prepared_schema_version`
   - canonical `task` and `task_hash`
   - `engine_fingerprint`
   - `profile` (template name and SHA-256)
   - `boundary_policy` (`tail-suffix-v1`) and `tail_tokens`
   - per-category `id`, `label`, `response`, `value`, `token_id` and `token_piece`
   - `prepared_hash`, which is the JCS hash of all the fields above

   Diagnostics and previews are returned next to the artifact, not inside it, so they never affect identity.
3. **Error envelope.** Every non-200 response is `{"error": {code, message, details?, request_id?}}` and carries the `LLikert-Protocol: 1` header. Validation details list the location and the error type only; submitted values are never echoed.
4. **Request-level error codes:**
   - 400: `invalid_json`
   - 401: `unauthorized`
   - 404: `not_found`
   - 405: `method_not_allowed`
   - 409: `engine_fingerprint_mismatch`, `prepared_task_mismatch`
   - 413: `body_too_large`, `too_many_items`
   - 422: `invalid_request`, `invalid_task`, `invalid_response_codes`, `task_exceeds_context`, `render_error`, `boundary_error`, `duplicate_ids`, `invalid_encoding`
   - 429: `queue_full`
   - 503: `starting`, `startup_failed`, `service_unavailable`
   - 500: `internal_error`

   429 and 503 responses include `Retry-After`.
5. **Item statuses** are exactly the eight in the plan. Failed items carry K-length null arrays, and `expected_value` is present only for numeric tasks (null when the item failed).
6. **Strict JSON.**
   - The service parses request bodies itself: UTF-8 only, `NaN`/`Infinity` rejected, duplicate keys rejected, unknown fields rejected (pydantic strict mode).
   - A lone surrogate is handled by field: in an item `text` it becomes the item status `invalid_encoding`; in ids and task strings it is a 422.
7. **Only `/health` is unauthenticated,** and `/openapi.json` and `/docs` are served only with `--auth none` (loopback only). `llikert export-schemas` writes the JSON Schemas offline, and a test fails if the committed `schemas/` drift from the models.

## Rendering and preparation

8. **Sentinels** are private-use code points (U+E000 … U+E001). Besides the sentinel render, the template is rendered with the real content and the two must agree. That rejects templates whose output depends on message content, which would otherwise make sentinel segmentation unfaithful.
9. **Prompt v1 (renderer version 1).**
   - The system message holds the instructions, a blank line, `Response codes:`, one `{response} = {label}` line per category, a blank line, and `Answer with exactly one of the response codes listed above and nothing else.`
   - Each example becomes a user turn plus an assistant turn containing the response string.
   - The item is a user turn of the form `Text:\n<text>\n{text}\n</text>`.
   - Qwen3 prompts produced this way are token-identical to the M0 reference fixture.
10. **Response-code suggestions (D3).**
    - The error details hold `problems` (with the tokens actually produced) and `suggestions.prefix.per_category` (`from_response`, `from_label`, `collision`).
    - `suggestions.prefix.complete` holds a ready-to-use `responses` list when every category has a unique valid prefix.
    - `suggestions.generic` offers the letter and digit sets that pass the boundary check.
11. **Reserved-marker text** in an item gets the warning `reserved_marker_text` and is still scored, as ordinary text.

## Engine and execution

12. **One batch-size setting.** `--batch-size` sets both `n_batch` and `n_ubatch`. M0 showed that only the physical micro-batch affects the numbers, so the fingerprint records the effective `n_ubatch`. This refines D6.
13. **`--device cuda|cpu` replaces a raw layer count.**
    - `cuda` offloads all layers and fails at startup if no GPU backend is present.
    - `cpu` offloads nothing and also disables KQV and op offload.
    - Partial offload is not offered, because its fidelity is unmeasured.
14. **Fingerprint v1 contents** (`GET /v1/info` shows them in full):
    - the llama.cpp version and ggml version and commit, plus the llama-cpp-python version;
    - the GGUF SHA-256, architecture, file type, parameter count, vocabulary size and BOS policy;
    - device type and device descriptions;
    - `n_gpu_layers`, effective `n_ctx`, `n_ubatch`, flash-attention mode and KV type;
    - template profile, renderer, boundary-policy, postprocessing and prepared-schema versions;
    - the tokenization policy.

    Thread count, model file name, `general.name` and system info are reported under `execution`, not fingerprinted.
15. **Pinned native library.**
    - The server extra does not depend on `llama-cpp-python`, because a PyPI wheel would vendor an older llama.cpp.
    - At startup the service requires `libllama.so.0.4.1`, loads it with llikert's own ctypes declarations, and never imports the binding's Python package.
    - `tools/build-llama-cpp-python.sh` produces the pinned wheel.
16. **Numerics.** `p` is computed as a softmax over the candidate logits. That is mathematically identical to `exp(log_q − log_coverage)`, but it avoids subtracting two large log values: in the underflow test the sum of `p` improved from 1 + 1.6e-14 to exact. `log_q` and `log_coverage` still use the full-vocabulary normalizer.
17. **Native logging.** llama.cpp messages are routed to the `llikert.server.llama` logger at debug level. The callback is removed on close and at interpreter exit, which fixed an intermittent crash at test-process shutdown.
18. **Auth token.** `--auth token` reads `LLIKERT_API_TOKEN`, requires at least 16 characters, and compares in constant time.

## Evidence (2026-09-16, RTX 4090, CUDA 12.9)

- **Unit suite, no model:** `pytest` — 139 passed. It covers JCS against RFC 8785 plus 2,998 doubles formatted by Node.js; hand-derived numerics; task validation and fixtures; rendering; preparation and suggestions; service statuses and 409 checks; HTTP strictness, auth, limits, the 429 queue and readiness; the ABI layout against the pinned header; the lightweight base import; and schema drift.
- **Integration:** `pytest -m model` — 5 passed.
  - Qwen2.5-0.5B matches the M0 spike to within 1e-6.
  - Real-tokenizer boundary rules and marker injection behave as specified.
  - With `n_ctx` = 256, a 256-token prompt scores, a 257-token prompt gets `context_limit`, and the next item is OK.
  - **The D24 gate passes for Qwen3-4B-Instruct-2507 F32 on CUDA with defaults:** max |Δp| = 0.00083 and max |Δlog q| = 0.063 over 28 items.
- **End-to-end CLI:** `llikert serve` with the F32 profile, `--model-sha256` and `--auth token`.
  - `/health` went 503 → 200 after 13.9 s, including hashing the 16 GB file.
  - `/v1/info` without a token returned 401.
  - A six-item chunk with ok, missing, empty and over-context items scored in 83 ms, and a malformed task returned 422.
