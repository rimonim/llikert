# LLikert v0.1: Detailed implementation plan

## Context

`llikert_development_plan.md` defines the product: a llama.cpp scoring service, plus thin R and Python HTTP clients, that returns full-vocabulary next-token probabilities for response codes a researcher specifies. The plan settles scope and mathematics but leaves open many interface, protocol and engineering choices. This document turns milestones M0–M2 into concrete tasks, sketches M3–M5, records the decisions already made with the owner, and lists the remaining choices with a recommendation for each.

**Current state:** `/home/louis/Documents/llikert` contains only the plan document. It is not a git repository, and the referenced `likert_client.py` and `likert_server.py` files are not present. That is fine, because the plan says they are background only.

**Local environment (useful for M0):**
- Python 3.12.8 (miniconda)
- R 4.6.1, with httr2, jsonlite, openssl, cli, testthat and S7 installed
- Docker
- RTX 4090 (24 GB), 187 GB RAM
- `~/llama.cpp` at commit fef693dc (May 2025), built with CUDA; `models/` holds `ggml-vocab-*.gguf` vocab-only files, which are ideal for tokenizer unit tests without weights, plus `Qwen3-4B-Q8_0.gguf`
- llama-cpp-python, httpx, fastapi and pydantic are not installed

## Decisions confirmed with owner

| # | Decision |
|---|---|
| C1 | **R API uses the plan's names:** `scoring_task()`, `scorer_connect()`, `prepare_task(task, engine)`, `score_texts(texts, ids, task, engine, ...)`. Helpers follow the same unprefixed style. |
| C2 | **The R result is an S3 `llikert_result` that subclasses a tibble.** It is wide, in input order: `id`, then `expected_value` (numeric tasks only), then one column of conditional probabilities `p` per category id. Diagnostics, candidate probabilities and log-probabilities, the category map and the manifest are stored as attributes and read through accessors. |
| C3 | **First model profile is Qwen3-4B-Instruct-2507** (non-thinking, Apache-2.0). Qwen2.5-0.5B-Instruct Q8_0 is the small CI and smoke model. **Weight format revised by 0002 D23: F32 GGUF on CUDA, because Q8_0 failed fidelity against the unquantized model.** |
| C4 | **Checkpointing is off by default.** When N > 200 and no `checkpoint` is given, print a one-time hint. |

## Remaining decisions (recommendation in bold; you can override when approving)

**Service and protocol**
- **D1 Template rendering:** **Render the chat template embedded in the GGUF with a sandboxed `jinja2` environment, and accept it only if its SHA-256 is on an allowlist of model profiles.** Anything not on the allowlist gets a structured `unsupported_template` error. Golden prompts are captured once from the HF tokenizer's `apply_chat_template` and committed as fixtures. The native `llama_chat_apply_template` is not used, because its template set is limited and it runs heuristics.
- **D2 Special-token safety:** **Render with sentinel placeholders, split at the sentinels, and tokenize in segments.** Template segments use `parse_special=True`; user content, instructions and examples use `parse_special=False`, so markers inside dataset text become ordinary tokens. The final prompt is defined as the concatenation of these segments. If content contains text matching a reserved marker, the item gets the warning `reserved_marker_text`, not an error.
- **D3 Answer-boundary check:**
  - At prepare time, the tail segment (for example `<|im_start|>assistant\n`) is tokenized with and without each response string appended. The difference must be exactly one ordinary token, distinct across categories, and not a control or end-of-generation token.
  - Content always ends before a special end-of-turn token. So the per-item check reduces to confirming that the item's token sequence ends with the prepared tail tokens; if not, the item gets `boundary_error`.
  - **When a code fails, the `prepare` error includes suggestions but never remaps anything.** Every suggestion passes the same D3 boundary check before it is offered, and the task stays unchanged until the researcher resubmits it with the new responses. Suggestions come in two groups:
    - **Prefix suggestions.** Tokenize each category's supplied `response` at the answer boundary, and separately its `label`. The response string that the first token decodes to is a candidate code, for example `"description"` → `"desc"` or `" Desc"`. Where the response and the label give different first tokens, list both. Each prefix is valid only if it is an ordinary token and differs from every other category's candidate. Categories whose prefixes collide are flagged. If every category has a unique valid prefix, the whole set is offered as a ready-to-use `responses` vector. Otherwise the non-colliding prefixes are offered, and the colliding categories are marked as needing a manual choice.
    - **Generic sets.** Single-token letter (`A`–`Z`) or digit (`1`–`9`) sequences of length K.
  - The error message and the docs point out that a prefix code changes the instrument: the instructions will show the prefix as the answer code. In R the suggestions come as a data frame on the condition object, `cnd$suggestions`; in Python they are `PrepareError.suggestions`.
- **D4 What a score request carries:** **The client sends back the whole prepared artifact it received** (which embeds the canonical task), the expected engine fingerprint, a request id, and `items`. The server prepares the task again (LRU-cached by task hash) and compares the full prepared hash. On mismatch it returns `409` before any scoring.
- **D5 Canonicalization and hashing:** **RFC 8785 (JCS) applied to the canonical task, with defaults filled and keys ordered. The identifier is SHA-256 hex prefixed with `sha256:`.** Only the server hashes. Fixtures under `tests/fixtures/tasks/` pair each input task with its canonical form and hash. R sends values with `jsonlite` `digits = NA` so doubles round-trip exactly.
- **D6 Engine fingerprint:**
  - **Included:** a hash (JCS) of the GGUF SHA-256, template SHA-256, renderer version, postprocessing version, tokenizer BOS/special policy, llama-cpp-python version, llama.cpp commit, device class and GPU name, `n_gpu_layers`, `n_ctx`, `n_batch`, `n_ubatch`, KV-cache types and `flash_attn`.
  - **Excluded:** host, port, timestamps, PID and request ids.
  - The GGUF hash is computed at startup. You can pass `--model-sha256` so startup verifies the file instead of only reporting its hash.
  - **Effect:** a CPU replica and a GPU replica get different fingerprints on purpose, so clients receive `409` rather than mixing results.
- **D7 Auth modes:**
  - **`--auth none`:** refuses to start on a non-loopback address.
  - **`--auth token`:** reads `LLIKERT_API_TOKEN`; bearer token compared in constant time.
  - **`--auth platform`:** for non-loopback binds behind a platform that enforces auth, such as a protected HF endpoint. It must be chosen explicitly and logs a startup warning.
  - Only `/health` is open in every mode.
- **D8 Preview without a fifth endpoint:** **`/v1/prepare` always returns a preview built from a harmless built-in example. An optional `preview_text` request field renders a real text on request.** The preview is not part of the prepared artifact or its hash and is never logged.
- **D9 Concurrency:** **One Uvicorn worker. Inference runs on a single-thread executor behind a bounded pending queue (default 4); when the queue is full the server returns `429` with `Retry-After`.** `/health` answers from the event loop while inference runs.
- **D10 Server limits** (advertised in `/v1/info`): `max_items_per_request` 64, `max_body_bytes` 8 MiB, `n_ctx`, `max_prompt_tokens`. All are operator-configurable.

**Client semantics (shared R/Python; tested with fixtures)**
- **D11 Missing and empty input:** R `NA` or Python `None` gives `missing_input`; `""` gives `empty_input`. The client assigns both and never submits them. **Whitespace-only text is scored normally.**
- **D12 Encoding:**
  - **R:** strings marked latin1 or native are converted with `enc2utf8`, which is legitimate transcoding, not repair. If `validUTF8()` fails, the item becomes `invalid_encoding`.
  - **Python:** strings with lone surrogates become `invalid_encoding`; passing `bytes` is a `TypeError`.
- **D13 IDs:**
  - `ids = NULL` produces `"1".."N"`.
  - Character, integer and factor ids are accepted. Doubles are accepted only when they are whole numbers below 2^53, and are formatted without scientific notation.
  - `NA` or duplicate ids raise an error before any request is sent.
  - The category ids `id` and `expected_value` are reserved, because they would collide with tibble columns.
- **D14 Prepared tasks are required:** `score_texts()` and `engine.score()` accept only a prepared task. A raw task raises an error with a hint to call `prepare_task()`. Before submitting, the client compares the prepared fingerprint with the one `/v1/info` reported at connect time.
- **D15 Connect behavior:** `scorer_connect()` / `Scorer.connect()` eagerly call `/v1/info`, which checks auth, protocol major version and fingerprint. A `wait` argument (default 300 s) polls `/health` through `503` responses from cold starts, reporting "service starting" progress.
- **D16 Retry policy (same in both clients):**
  - Up to 6 tries, with exponential backoff capped at 60 s, full jitter, and `Retry-After` honored.
  - **Retried:** `429`, `502`, `503`, `504`, timeouts and connection resets.
  - **Never retried:** `400`, `401`, `403`, `404`, `409`, `422`.
  - **`413`:** halve the HTTP chunk size, down to 1. This does not change semantics.
  - Non-JSON proxy bodies are reduced to status plus a short sanitized excerpt.
- **D17 Chunking:** a fixed default of `chunk_size = 16`, clamped to the server limits. There is no adaptive sizing in v0.1.
- **D18 Saved result:** **one cross-language `result.v1.json`** via R `write_result()` / `read_result()` and Python `ScoreResult.save()` / `.load()`. A checkpoint directory can be read offline with `read_checkpoint()`. For CSV, users call `write.csv()` on the tibble.

**Packaging and naming**
- **D19 Accessor names (confirmed):** `llikert_result_diagnostics()`, `llikert_result_candidate_probs()`, `llikert_result_candidate_log_probs()`, `llikert_result_categories()`, `llikert_result_manifest()`. The `llikert_result_` prefix avoids masking functions in other packages.
- **D20 Python client stack:** frozen stdlib `dataclasses` plus `httpx`, on Python ≥ 3.10, synchronous only. Pydantic stays in the server extra. `to_pandas()` and `to_numpy()` import lazily.
- **D21 License and names (confirmed):** MIT for both packages. M0 checks whether `llikert` is available on PyPI, CRAN, r-universe and GitHub.
- **D22 Backend currency (confirmed: update from the start):**
  - **Build:** in M0, build llama-cpp-python from source with its `vendor/llama.cpp` submodule moved to a recent tagged llama.cpp release (`bNNNN`), and pin that tag.
  - **Local checkout:** the existing `~/llama.cpp` checkout (May 2025) is left untouched; the build happens in the project's own build directory.
  - **Where the adapter touches the binding:** only through the low-level `llama_cpp.llama_cpp` module and the roughly 20 native functions it needs: model and context init, vocab and token attributes, tokenize, batch, decode, `llama_get_logits_ith`, memory clear and free.
  - **Check at startup and in M0:** those signatures must match `llama.h` at the pinned tag.
  - **If the tag's signatures differ from llama-cpp-python's declarations:** the adapter declares argtypes/restype for just those functions against the `libllama` shipped with the build. That is a narrow shim, not a fork. It is recorded in a decision record and covered by a signature test that parses the pinned `llama.h`.
  - **Scope check:** this stays within "llama.cpp through pinned llama-cpp-python bindings". I'll escalate if more than that shim is needed.
  - **Fingerprint:** records the llama-cpp-python version and the vendored llama.cpp tag and commit separately.

## Repository layout

```text
llikert/
  python/
    pyproject.toml                 # base deps: httpx; [server]: fastapi, uvicorn, pydantic, numpy, jinja2, llama-cpp-python==<pin>
    src/llikert/
      __init__.py                  # ScoringTask, Category, Scorer, PreparedTask, ScoreResult, read_checkpoint
      task.py  client.py  result.py  checkpoint.py  _http.py (retry/errors)  _ids.py  _version.py
      cli.py                       # `llikert serve|selfcheck|export-schemas`; server imports are lazy
      server/
        app.py  config.py  schemas.py (pydantic)  canonical.py (JCS)  profiles.py (template allowlist)
        render.py  tokenize.py  prepare.py  numerics.py  fingerprint.py  errors.py
        adapter.py (private llama.cpp)  fake_adapter.py (tiny vocab, deterministic logits)
    tests/
  r/llikert/
    DESCRIPTION  NAMESPACE
    R/ task.R connect.R prepare.R score.R result.R checkpoint.R http.R conditions.R io.R print.R ids.R
    tests/testthat/ (+ fixtures/ synced copy)
    vignettes/ quickstart-hosted.Rmd  quickstart-local.Rmd
  schemas/            task.v1, prepared-task.v1, score-request.v1, score-response.v1, info.v1, error.v1, result.v1, checkpoint-manifest.v1 (.json)
  tests/fixtures/     numerics/  tasks/{valid,invalid}/  protocol/  checkpoints/  prompts/ (golden)
  tests/integration/  real-model and container tests (explicit opt-in)
  tools/sync_fixtures.py           # copies fixtures into the R package; CI fails on drift
  deploy/             Dockerfile.cuda, hf-endpoint.md
  docs/               scoring-spec.md, protocol.md, checkpoint-format.md, compatibility.md, decisions/NNNN-*.md
  benchmarks/         corpus generator, runner, reports/
  spikes/             M0 only; deleted in M5
```

Pydantic models in `server/schemas.py` generate `schemas/*.json`. `llikert export-schemas --check` fails CI when the committed schemas drift from the models.

## M0: Feasibility (spike code, no public API)

1. Run `git init`. Write `docs/decisions/0001-…` covering C1–C4 and D1–D22 as they are accepted.
2. Create a venv.
   - Clone the latest llama-cpp-python release and move `vendor/llama.cpp` to the newest stable llama.cpp release tag (D22).
   - Build a CUDA wheel (`CMAKE_ARGS="-DGGML_CUDA=on"`) and a CPU wheel side by side.
   - Diff the pinned `llama.h` against the binding's declarations for the functions we use.
   - Record the llama-cpp-python version and the llama.cpp tag and commit.
   - If the build fails on the newest tag, step back to the most recent tag that builds and record why.
3. Download the Qwen3-4B-Instruct-2507 Q8_0 and Qwen2.5-0.5B-Instruct Q8_0 GGUFs from their official repos, about 5 GB in total. This is network use only, nothing billable. Record SHA-256 and revision for each.
4. Low-level spike, `spikes/m0_logits.py`:
   - Load the model and read `tokenizer.chat_template` from its metadata.
   - Render the template with jinja2 and compare against the HF `apply_chat_template` golden output.
   - Tokenize in segments (D2). Check boundary tokens for `A/B/C` and `1..5`, and confirm that ` A` and `10` behave as expected.
   - Decode in `n_batch` chunks with logits only at the final position, then copy `llama_get_logits_ith(-1)` into a float64 array.
   - Apply the §4 math, and cross-check it against a scalar-loop logsumexp.
5. Probes:
   - Is the logits pointer still valid after the next decode? Copy immediately either way.
   - Exact limits: `n_ctx` tokens should decode; `n_ctx + 1` should be rejected.
   - Reset: find the memory/KV clear call in this version (the API renamed `kv_cache_clear` to `llama_memory_clear`).
   - Special-token attributes: which API exposes control and EOG flags.
   - Repeatability: run the same prompt 5 times, and compare batch sizes 1 vs 512, reporting max absolute differences.
6. Save one nominal and one numeric task result as JSON in `spikes/out/`.
7. Container probe:
   - Minimal CUDA Dockerfile with a stub `/health`; run it locally with `--gpus all`.
   - Read the current HF custom-container docs for port, health route, the `/repository` mount and request timeouts.
   - Record anything not verified.
8. Name availability check (PyPI, CRAN, GitHub).

**Exit check:** spike numbers are reproducible; golden prompt equality holds; any API gap is written up in a decision record.

**M0 outcome (see `docs/decisions/0002-m0-findings.md`):**
- **ABI shim:** the adapter declares its own ctypes structs and functions, and an ABI check compares them with the C header.
- **Development Python:** system Python, not conda.
- **Adapter preconditions:** checked in Python before native calls, because native asserts abort the process.
- **Weights:** F32 on CUDA (D23), subject to the reference-fidelity gate (D24).
- **Container probe:** not executed, because there is no Docker daemon access.

## M1: Core service

**Order:** numerics → task/canonical → render/tokenize/prepare → fake adapter → HTTP → real adapter → CLI.

- `numerics.py`
  - Takes a float32 logits vector and token ids.
  - If any value in the full vector is nonfinite, the item gets `numerical_error`.
  - Otherwise returns `log_q`, `log_coverage`, `q`, `p`, `coverage` and the optional `expected_value = Σ w·p`, all in float64 with no rounding.
  - Fixtures cover a hand-computed 5-token vocabulary, underflow (log_q ≈ −800 with q = 0 exactly), and very low coverage with a concentrated `p`.
- `suggest.py` builds the D3 code suggestions (prefix and generic). It is tested with the vocab-only GGUFs, including a collision case such as `"question"` and `"quest"` sharing a first token.
- `canonical.py` and `task` schemas cover the validation rules in §5.1:
  - at least 2 categories;
  - unique, nonempty ids;
  - values finite and given for all categories or none;
  - example category ids exist;
  - no trimming or Unicode normalization.
- `profiles.py` holds the allowlist: template SHA-256 → profile, which fixes the renderer settings, the tail-segment definition and BOS policy.
- `prepare.py`
  - Builds the prompt: system = instructions + code mapping + "answer with exactly one code"; few-shot examples as user/assistant turns; user text in a delimited field.
  - Runs the D3 checks and returns the prepared artifact: canonical task, `task_hash`, `prepared_hash`, `engine_fingerprint`, and per-category `{id, label, response, token_id, token_piece, value}`.
  - Also returns renderer id, boundary policy, diagnostics and the preview (D8).
- `adapter.py`
  - Owns the model and context behind a lock. `score_prompt(tokens) -> float64 logits`, taking bounded batches with logits requested at the last position only.
  - Clears memory before each item. After a failed decode it reinitializes the context; if that fails too, it marks the service unhealthy.
- `app.py` exposes `/health`, `/v1/info`, `/v1/prepare`, `/v1/score`:
  - strict schemas (`extra="forbid"`), protocol version in body and header, and D4 checks before scoring;
  - per-item statuses: `ok | missing_input | empty_input | invalid_encoding | context_limit | boundary_error | numerical_error | inference_error`;
  - failed items carry K-length null arrays; there are never NaN values in the JSON;
  - request metadata includes elapsed time and effective settings.
- `cli.py serve` accepts `--model --host --port --n-ctx --n-gpu-layers --n-batch --auth --model-sha256 --max-items --queue`.
  - It fails if GPU use was requested but no GPU is available.
  - It prints identity and readiness.
  - `selfcheck` runs the same checks without starting the server.
- Logging records request id, item count, status counts and timing only; never text, prompts or tokens.

**Exit check:** pytest suites pass with the fake adapter (numerics, tasks, prepare, protocol, errors). A real-model integration test scores a chunk and matches the M0 spike to within 1e-6 in `p` and log_q.

## M2: Clients (R and Python in parallel, once M1 schemas are frozen)

### R API (C1, C2)

```r
task <- scoring_task(name, instructions, categories, responses,
                     values = NULL, ordered = FALSE, ids = categories,
                     examples = NULL)      # examples: data.frame(text, category)
write_task(task, path); read_task(path)
engine   <- scorer_connect(url = Sys.getenv("LLIKERT_URL"),
                           token = Sys.getenv("LLIKERT_TOKEN"), wait = 300)
scorer_info(engine)
prepared <- prepare_task(task, engine, preview_text = NULL)
print(prepared)                           # mapping table: id, label, response, token_id, piece, value
preview_prompt(prepared)                  # renders stored preview; exact export via write_prompt_preview()
result <- score_texts(texts, ids = NULL, task = prepared, engine = engine,
                      chunk_size = 16, checkpoint = NULL, resume = FALSE,
                      progress = interactive())
llikert_result_diagnostics(result); llikert_result_candidate_probs(result)
llikert_result_candidate_log_probs(result); llikert_result_categories(result)
llikert_result_manifest(result)
write_result(result, path); read_result(path); read_checkpoint(dir)
```

- **Object structure:** `llikert_result` is built with `tibble::new_tibble(..., class = "llikert_result")`. Its attributes are tibbles keyed by `id`.
- **Accessors after dplyr operations:** accessors subset their attribute tables to the ids still present, in their current order, so `filter()` or `arrange()` don't leave them misaligned. If an attribute was dropped, the accessor errors with advice to keep the original object.
- **Print header:** `tbl_sum.llikert_result` prints the task name, ok/failed counts and the coverage median and minimum. Source text is never stored, so it is never printed.
- **Failed rows:** probabilities are `NA`; `llikert_result_diagnostics()` gives the status and error code.
- **Errors:** raised with `rlang::abort(class = "llikert_error_<code>")`. `interrupt` is caught between chunks, and already committed work is kept.
- **Dependencies:**
  - Imports: httr2, jsonlite, openssl, cli, rlang, tibble. Most arrive as httr2 dependencies anyway.
  - Suggests: testthat, withr, dplyr.
- **Tests:** they never need Python. They use `httr2::local_mocked_responses()` with the synced protocol fixtures.

### Python API

```python
task = ScoringTask(name=..., instructions=..., categories=[...], responses=[...],
                   values=None, ordered=False, ids=None, examples=None)
ScoringTask.from_json(path); task.to_json(path)
with Scorer.connect(url=os.environ["LLIKERT_URL"], token=os.environ.get("LLIKERT_TOKEN"), wait=300) as engine:
    engine.info()
    prepared = engine.prepare(task, preview_text=None)   # PreparedTask; .mapping, .preview, .save/.load
    result = engine.score(texts, ids=None, task=prepared, chunk_size=16,
                          checkpoint=None, resume=False, progress=True)
result.ids; result.category_ids; result.probabilities      # list[list[float|None]]
result.expected_values; result.diagnostics; result.manifest
result.to_pandas()   # same column order as the R tibble; lazy import
ScoreResult.load(path); result.save(path); read_checkpoint(dir)
```

- **`progress`:** accepts a bool (a plain stderr line printer) or a callable. There is no tqdm dependency.
- **Import guard:** a test asserts that `import llikert` never loads numpy, fastapi, llama_cpp, torch or transformers (checked through `sys.modules`).
- **Tests:** `httpx.MockTransport` serving the same protocol fixtures.

### Checkpoint format (shared, `docs/checkpoint-format.md`)

```text
run-dir/
  manifest.json      # format version, client+protocol version, canonical task, task/prepared hash,
                     # engine fingerprint, ordered ids, sha256(utf8 text) per id, dataset hash, created_at
  .lock/             # created with mkdir (atomic in both R and Python); holds owner.json; stale lock needs force_unlock = TRUE
  chunks/000001.json # {fingerprint, prepared_hash, results:[...]} written as tmp then renamed in the same dir
```

- **Resume check:** the ids in order, the text hashes, the prepared hash and the fingerprint must all match. Otherwise the run fails with an error that says to start a new run directory.
- **Merging:** chunks are merged by id. Later chunks override retryable failures only; successes are never re-scored.
- **Errors:** deterministic input errors are stored in the checkpoint; retryable ones (`inference_error`, transport) are retried.
- **Durability:** R has no `fsync`, so the docs will state that R commits are atomic but not crash-durable across power loss.
- **Cross-language:** Python can resume a checkpoint that R wrote, and R can resume one that Python wrote.

**Exit check:**
- Both clients pass the shared fixture run with identical normalized `result.v1.json`, excluding timestamps.
- Tests cover interruption before rename, between rename and manifest update, and after; each is resumed by the other language.
- An all-success checkpoint makes 0 requests.
- A clean-install check passes: an R-only Docker image (rocker) running `R CMD check`, and a Python venv with only the base install.
- Against the real local service, both clients produce the same probabilities to within 1e-12, since the server computes them once.

## M3–M5 (outline)

- **M3:**
  - Multi-sequence batching via `llama_batch` with per-sequence last-position logits.
  - Shared instruction-prefix reuse, adopted only if the benchmark shows a gain.
  - Parity with the sequential path is measured and reported. The gate is the D24 reference-fidelity check: max |Δp| ≤ 0.005 and max |Δlog q| ≤ 0.1 against transformers float32. M0 showed that on CUDA, changing `n_ubatch` or flash attention moves results beyond 1e-4, so bit-parity across batch configurations is not a requirement; each configuration is a separate fingerprinted profile.
  - Stress tests for context, OOM and state isolation.
  - Benchmark on a synthetic corpus with short, medium and long texts and K ∈ {2, 5, 10}, and a written compatibility matrix.
- **M4:**
  - `deploy/Dockerfile.cuda` with pinned base-image digest and lockfile; the model is read from `/repository/<file>.gguf`.
  - Local container tests, then an HF endpoint guide.
  - A live HF endpoint only after you explicitly authorize it; otherwise the guide is labeled "experimental".
- **M5:**
  - Delete `spikes/`.
  - License and security review, reproducible examples, a methods-paragraph template, and a handoff report of commands, results, unverified items and owner actions.

## Verification (end to end)

1. `pytest python/tests`: numerics, schema drift, prepare, protocol and client tests, all without a model.
2. `LLIKERT_TEST_MODEL=… pytest tests/integration -m model`: real Qwen2.5-0.5B, then Qwen3-4B-Instruct-2507 on CPU and CUDA.
3. `R CMD check r/llikert` in an R-only container, without Python.
4. Manual vertical slice:
   - Run `llikert serve --model …/Qwen3-4B-Instruct-2507-Q8_0.gguf`.
   - The R quickstart runs a nominal task (no `expected_value` column) and a numeric 1–5 task (with an `expected_value` column).
   - Interrupt with Ctrl-C mid-run, resume, and `write_result()`.
   - `ScoreResult.load()` in Python shows identical values.
5. `python tools/sync_fixtures.py --check` finds no drift.
