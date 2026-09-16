# LLikert: Development Plan

## Minimal research-grade candidate-probability scorer

**Target release:** v0.1  
**Document version:** 1.0  
**Date:** September 16, 2026  
**Audience:** Coding agent, maintainer, and research collaborators  
**Working name:** LLikert (`llikert` in examples; check registry availability before publication)

## 1. Product objective and implementation mandate

Build a small, reliable tool for researchers scoring fixed text datasets with a causal language model. For each text, calculate the model's next-token probabilities for explicitly specified response codes. Return both their original probability mass and their distribution conditional on choosing one of those codes. Support nominal categories, ordered categories, and numeric scales without making numeric summaries mandatory.

The product comprises one Python-based scoring service using llama.cpp, a lightweight R HTTP client, and a lightweight Python HTTP client. The same service must run locally or in a versioned container on Hugging Face Inference Endpoints. Connecting from R to a hosted service must require neither Python nor a local inference runtime.

**Implement this design as a new product, not as a compatibility-preserving refactor of the supplied scripts.** The existing client demonstrates a shared HTTP scoring path, and the existing server demonstrates full-vocabulary candidate scoring. Neither its model loader, optimization machinery, names, defaults, nor output format is binding. See [L1] and [L2].

Treat the requirements marked **MUST** as acceptance conditions. Recommendations marked **SHOULD** can be changed with a short engineering rationale. Function names and internal file names below are proposed implementation defaults; scoring semantics and scope are settled. Resolve routine implementation details without reopening the product decisions.

The first deliverable is a working vertical slice with mathematical tests, not an elaborate backend framework. Do not replace missing functionality with sampled ratings, structured-output generation, or truncated top-token probabilities.

## 2. Settled scope

| Area | v0.1 decision |
| --- | --- |
| Scoring event | The next token is one of the explicitly configured candidate tokens. |
| Response codes | One ordinary token per category; distinct token IDs; one response string per category. |
| Categories | Arbitrary descriptive names; nominal or ordinal; mutually exclusive within a task. |
| Numeric summaries | Optional, using explicitly supplied finite values; no automatic equal spacing. |
| Inference | llama.cpp through pinned, tested `llama-cpp-python` bindings. |
| Interfaces | R and Python clients use one versioned HTTP protocol. |
| Service ownership | An operator starts the service explicitly; clients connect without spawning or stopping it. |
| Deployment | Local service and one custom-container deployment recipe for Hugging Face. |
| Workload | Synchronous dataset chunks, with client-side checkpoints and resume. |
| Reproducibility claim | Sampling-free probability scoring, with a recorded execution manifest. |

**Out of scope:** Multi-token response codes; overlap correction; terminators; multiple verbalizations per category; length normalization; Transformers, Ollama, SGLang, or vLLM backends; `reticulate`; native R inference; generic provider adapters; embeddings; fine-tuning; tools or retrieval; generated explanations; multimodal inputs; a graphical interface; a distributed task queue; and automatic cloud provisioning.

Do not add placeholder public arguments for these deferred features. Keep the backend adapter internal and narrow. A nominal task with simultaneously applicable labels is not one categorical distribution: document separate binary tasks as the initial approach. Multi-construct scoring is a loop over tasks, not a new orchestration subsystem.

Long category names and values such as 10 or 100 remain supported because their model response codes can be distinct single tokens such as `A`, `B`, and `C`.

## 3. Intended researcher workflow

A researcher defines instructions, categories, response codes, and optionally numeric values and few-shot examples. They connect to a running service, prepare the task, inspect the mapping and prompt preview, and score a text vector or data-frame column. Results retain input order and identifiers and can be saved independently of an active connection.

Preparation is a visible validation step. An invalid response code should fail before an expensive dataset run. Preparation must not imply that every future input is valid: per-item context and boundary checks remain necessary.

For hosted use, the lab deploys the published service container once. Collaborators receive an endpoint URL and appropriate credentials and install only their language's client. For local use, the operator installs the service environment, obtains a compatible model, and starts the documented command. Package import and client connection never trigger model downloads or inference installation.

Documentation should lead with one hosted quickstart and one local quickstart. Avoid asking beginners to choose among runtimes. Explain the unavoidable local costs separately: model files, memory, and a tested native-library installation.

A successful first-run example must show meaningful categories, original candidate mass, conditional probabilities, diagnostics, and saving results. It must also show that a nominal task returns no expected-value score.

## 4. Mathematical contract

Let `x` be the exact rendered prompt token sequence ending immediately before the answer. Let `z[v]` be the unmodified next-token logit for vocabulary token `v`. A prepared task maps its `K` categories to distinct token IDs `t[1], ..., t[K]`.

The required computation is:

```text
log_Z       = logsumexp(z[v] for every vocabulary token v)
log_q[j]    = z[t[j]] - log_Z
log_coverage = logsumexp(log_q[1], ..., log_q[K])
p[j]        = exp(log_q[j] - log_coverage)
q[j]        = exp(log_q[j])
coverage    = exp(log_coverage)
```

Here `q[j]` is the original next-token probability of response code `j`, `coverage` is the mass assigned to all configured codes, and `p[j]` is the distribution conditional on selecting one of them. The service MUST retain `log_q` and `log_coverage`, even when exponentiation underflows to zero.

Compute normalization in a numerically stable form. Use float64 accumulation for the small postprocessing calculation when available without changing model inference precision. Do not divide by a clamped coverage floor. Do not round scores internally; display rounding belongs in the clients.

For explicitly provided numeric values `w[j]`, the optional expected value is `sum(w[j] * p[j])`. An ordinal declaration alone does not provide these values. Expected values are absent, not zero, for tasks without values.

### 4.1 Prohibited substitutions

The service MUST NOT sample an answer, apply temperature scaling, use top-k/top-p filtering, apply repetition penalties or logit bias, mask the vocabulary to force a valid code, or call a constrained-decoding API as a substitute for raw logits. No answer token needs to be evaluated after the prompt. There may still be multiple computational batches while processing a long prompt.

Softmax over selected logits alone can recover `p`, but cannot supply original mass without the full-vocabulary normalizer. Every requested token must be available regardless of its rank. Tokens absent from a top-N response must never be assigned zero by assumption.

### 4.2 Interpretation and validity

These are probabilities of particular token events, not calibrated probabilities that a research category is objectively correct. Coverage is a response-format diagnostic, not a validity score. It is possible to have a concentrated conditional distribution with very low coverage.

Changing a response code, category order, wording, model, template, or quantization can change the measurement. Document this as a reason to retain task and model identities and validate the instrument against appropriate reference judgments. Do not advertise universal equivalence to human ratings or cross-machine bitwise determinism.

## 5. Task and preparation objects

### 5.1 Portable task

Use a versioned, JSON-serializable object with these fields:

| Field | Meaning and validation |
| --- | --- |
| `schema_version` | Task schema version, distinct from package and HTTP versions. |
| `name` | Human-readable task name; not a unique database identifier. |
| `instructions` | Nonempty construct definition and decision instructions. |
| `categories` | Ordered array of at least two category records. Preserve order everywhere. |
| Category `id` | Unique, nonempty stable string used for machine-readable joins. |
| Category `label` | Human-readable name, independent of response code and numeric value. |
| Category `response` | Nonempty exact response string; whitespace is meaningful. |
| `ordered` | Boolean; preserves ordinal interpretation without imposing numeric spacing. |
| Category `value` | Optional finite number; supplied for every category or for none. |
| `examples` | Optional ordered list of example text and category ID, not arbitrary code. |

Values may be nonintegers or negative. Reject nonfinite values and partial mappings. Preserve user strings without silently trimming or Unicode-normalizing them. Validate encoded text explicitly. Empty datasets are valid inputs to the client, but an empty category set is not a valid task.

The server owns authoritative task canonicalization and hashing. Publish canonicalization rules and fixtures so portable files remain stable across client languages. Do not derive task identity from an R-specific serialization or incidental JSON field order.

### 5.2 Prepared task

Preparation resolves the task against the service's fixed model and renderer. Return the canonical task, prepared-task identifier, model/engine fingerprint, category mapping, exact token IDs and token pieces, renderer identity, answer-boundary policy, and validation diagnostics.

Provide a preview containing instructions, category-code mapping, and a harmless example input. Actual research text is shown only when the researcher explicitly requests a preview of that text. Preparing a task does not send the full dataset.

A prepared object MUST be portable across service restarts with the same fingerprint. It must not depend on an opaque identifier stored only in one process. Each scoring request carries sufficient task information for validation on a fresh replica. Cache prepared tasks only as an optimization.

Changing a code, category order, example, numeric value, renderer, or model invalidates preparation. A code that becomes multi-token under another model requires re-preparation and an actionable error, not automatic substitution.

## 6. Prompt rendering and token-boundary correctness

**Treat rendering and tokenization as part of the measurement instrument.** These are the highest-risk parts of the implementation, even with single-token response codes.

Use one service-side prompt builder: instructions and the category mapping in the instruction context, optional examples in a fixed order, and the current text in a clearly delimited user-content field. Explicitly ask for exactly one listed response code and no explanation. Apply the selected model's tested chat template and append its assistant-generation boundary.

The first supported model profile should be a text-only instruction model that can answer directly without a generated reasoning preamble. Do not generate or discard reasoning and then score a later position. Do not insert undocumented answer-prefill text to increase coverage.

### 6.1 Token validation

For each response code, verify all of the following:

- The code corresponds to exactly one ordinary token at the intended answer boundary, with no added beginning/end token and no hidden normalization.
- Appending the code does not retokenize or replace the already fixed prompt prefix under the documented validation procedure.
- Its token ID is distinct from every other category's ID, in range, and not a control, unknown, or end-of-turn token.
- Instructions and examples use the declared response strings and category mapping; preparation and inference use the same validated answer-position token IDs.

A useful boundary test is to compare tokenization of the rendered prompt with tokenization of that prompt followed directly by the response string, without a closing assistant-turn marker. The accepted token difference must be exactly the intended one-token suffix. Document the backend's beginning-of-sequence and special-token settings.

Preparation validates representative boundary cases; per-item validation ensures unusual text cannot alter the answer boundary. Tokenize the final prompt once for inference and verify any proposed optimization against the same reference construction. Do not assume a one-character string is always one token, or that a multi-character string cannot be one token.

### 6.2 Unsupported inputs and templates

Fail with a structured error for unsupported templates, multi-token codes, duplicate token IDs, or boundary instability. Suggest alternative response codes without silently remapping them. Do not infer or switch to a generic chat format when the configured template fails.

The native `llama_chat_apply_template` interface documents a limited template set rather than an unrestricted Jinja interpreter. Verify the actual installed binding's rendering route; do not assume every GGUF template is supported. [S1]

Untrusted dataset text must not introduce special template control tokens. Use a tested ordinary-text encoding path for content, or explicitly reject conflicting reserved-marker sequences. Do not claim that delimiters alone prevent prompt injection. Model-generated text is never executed; no tool or filesystem action follows a scored input.

## 7. Architecture and responsibility boundaries

```text
R client ---------\
                   HTTP/JSON --> Scoring service --> llama.cpp --> GGUF
Python client ----/

Local and hosted deployments expose the same protocol.
```

The service contains five small components: task validation/preparation, prompt rendering, a private llama.cpp adapter, numerical postprocessing, and HTTP request handling. Native-logit access is available through llama.cpp and its Python binding; the exact tested versions must be pinned during the first milestone. [S1, S2]

**Service ownership:** Prompt assembly, tokenization, candidate validation, all probabilities, optional expected values, and authoritative fingerprints belong on the service. Clients must not maintain independent scoring algorithms. Convenience recalculation from saved probabilities may exist only with cross-language tests.

**Client ownership:** Input validation, chunk submission, progress, bounded retries, checkpoint/resume, restoring order, presentation, and local result export belong in each language client. Both use shared protocol fixtures.

**Process model:** One loaded model and one protected inference context per service process. Use one inference worker initially. Prevent concurrent mutation of the context; keep health handling responsive while inference runs. Do not start multiple web workers that each load the same model accidentally.

There is no per-request model switching, no remote model path parameter, no public shutdown endpoint, and no client-side auto-launch. The operator starts and stops the service through its CLI or container supervisor. Native objects must be released on normal shutdown.

Recommended implementation defaults are FastAPI/Pydantic for request validation, Uvicorn for serving, NumPy for numerical work, and a private low-level `llama-cpp-python` adapter. These are engineering choices, not additional public integration modes. Do not import the binding's generic OpenAI-compatible server as the application itself.

## 8. Inference, memory, and performance

First implement a transparent sequential reference path: render one prompt, validate it, evaluate its tokens in bounded native batches, read the final-position raw logits, and calculate candidate results. The reference remains available to tests, not as a second advertised backend.

The production path MUST evaluate all candidates from one answer-position distribution. Request only logits needed at final prompt positions, not a vocabulary matrix for every prompt token. Never retain a NumPy view into native output memory after that memory can be reused; copy necessary values or complete the reduction before the next evaluation.

Validate actual rendered token lengths before inference. Use the effective configured context limit, including model/template overhead. The implementation must determine the correct last-position boundary empirically for the pinned API and test exact-limit and over-limit cases. Do not reserve arbitrary terminator tokens or silently truncate, summarize, context-shift, or extrapolate positions.

Use bounded prompt-prefill chunks and explicit memory limits. Distinguish HTTP chunk size, native token batch size, and simultaneous sequence count. llama.cpp exposes multi-sequence batches and output flags, but the adapter must use them correctly. [S1]

A correct bounded sequential implementation can establish the first vertical slice. Native multi-sequence batching and shared instruction-prefix reuse SHOULD follow only after parity tests exist. Shared-prefix caching is an optimization, not a new public mode; use it when it materially improves the benchmark. Its absence must be visible in performance results, not concealed as "batching."

Bound or avoid persistent caches. A failed native evaluation must not leave state attached to the next item. Retry at a smaller native batch only when safe state recovery is demonstrated; otherwise mark the service unhealthy or reinitialize its context. Never respond to memory pressure by changing weights, quantization, context semantics, or device silently.

Keep full-vocabulary output inside the service. Transport only category-sized results. Provide a reproducible benchmark measuring throughput, warm latency, startup, peak memory, prompt length, native batches, and HTTP overhead under an identical model and task. No cross-runtime speed guarantee is a release claim.

## 9. HTTP protocol

Publish versioned schemas and generated OpenAPI documentation. Use strict UTF-8 JSON and reject unsupported fields rather than accepting ignored scoring options. Carry protocol version and request IDs explicitly. Authentication secrets never appear in request bodies or returned manifests.

| Endpoint | Contract |
| --- | --- |
| `GET /health` | Minimal readiness: 200 only after successful model initialization and smoke check; 503 otherwise. |
| `GET /v1/info` | Service identity, protocol version, capabilities, limits, and effective model/engine fingerprint. |
| `POST /v1/prepare` | Validate a task against the model and return the portable prepared artifact. |
| `POST /v1/score` | Score one synchronous chunk of identified inputs with a prepared task and expected fingerprint. |

For non-loopback access, only `/health` may be unauthenticated; it must not disclose paths or model details. The other routes follow the deployment's authentication configuration. Verify that the hosted ingress preserves these routes; route rewriting, when necessary, must stay in deployment configuration rather than alter scoring semantics.

A score request includes the task/preparation artifact, expected fingerprint, request ID, and an ordered array of `{id, text}` records. IDs are strings; duplicate IDs in one dataset are rejected. Empty text, missing text, and nonstring values have distinct validation outcomes. A prepared task is never trusted solely because the client supplies token IDs: the service verifies it against the model and canonical task.

Return the actual engine/prepared-task fingerprints and one result per submitted input, in the same order. A mismatch produces `409` before scoring rather than mixing model versions in a run. Replicas must share the same effective fingerprint; process UUIDs and startup timestamps are not part of that stable identity.

Use `400`/`422` for malformed or invalid requests, `401`/`403` for access errors, `413` for body limits, `429` for bounded capacity, and `503` for temporary unavailability. Separate request-level failure from per-item failure. A completed chunk with some invalid texts can return `200` with explicit per-item errors.

There is no asynchronous job API, server-side dataset database, or streaming token output in v0.1.

## 10. Result and error contract

Use explicit category ordering from the prepared artifact. Each successful item contains `id`, `status`, `candidate_log_probs`, `candidate_probs`, `probabilities`, `log_coverage`, `coverage`, rendered prompt token count, prompt-token hash, optional `expected_value`, and structured warnings. Request-level metadata contains elapsed time and effective execution settings.

The arrays are always length `K`. Clients construct fixed-shape `N x K` tables/matrices, including when all rows fail. Do not rely on automatic coercion of a mixture of vectors and null rows.

A failed item has a non-success status, a machine-readable error code, a safe message, and `K` null entries for each probability array. Scalar quantities are null and the expected-value field is present only for numeric tasks. JSON never contains NaN or Infinity. Keep `missing_input` distinct from `empty_input`, `context_limit`, `boundary_error`, `invalid_encoding`, `numerical_error`, and `inference_error`.

Nonfinite native logits are a numerical error in the v0.1 contract, not a reason to emit a uniform distribution. Finite, extremely negative log probabilities may legitimately exponentiate to zero; retaining their finite log values distinguishes underflow from missing information. Small floating-point tolerances must be documented and tested, not used to hide substantial probability violations.

R results SHOULD expose `probabilities`, `candidate_log_probs`, `candidate_probs`, a per-item `diagnostics` data frame, optional `scores`, and `manifest`. Preserve names and order; do not coerce arbitrary labels into invalid or mangled column names without a reversible category-ID map. Python returns equivalent structured data without requiring pandas.

Print methods summarize successes, failures, and coverage without dumping source text. Optional numeric output must remain traceable to the supplied values. A single winner/argmax is not required for v0.1; the distribution is the primary result.

## 11. R and Python client interfaces

The public workflow is intentionally small. Names below describe the target API and are not claims about an already published package.

```r
library(llikert)

task <- scoring_task(
  name = "Communicative function",
  instructions = "Classify the text's primary communicative function.",
  categories = c("description", "question", "request"),
  responses = c("A", "B", "C"),
  ordered = FALSE
)

engine <- scorer_connect(
  url = Sys.getenv("SCORER_URL"),
  token = Sys.getenv("SCORER_TOKEN")
)
prepared <- prepare_task(task, engine)
result <- score_texts(
  texts = dat$text, ids = as.character(dat$id),
  task = prepared, engine = engine,
  checkpoint = "function-scoring-run", resume = TRUE
)
result$probabilities
result$diagnostics
```

```python
from llikert import ScoringTask, Scorer

# Pass an endpoint token through the environment, not source code.
import os

task = ScoringTask(
    name="Communicative function",
    instructions="Classify the text's primary communicative function.",
    categories=["description", "question", "request"],
    responses=["A", "B", "C"],
    ordered=False,
)
with Scorer.connect(
    url=os.environ["SCORER_URL"],
    token=os.environ.get("SCORER_TOKEN"),
) as engine:
    prepared = engine.prepare(task)
    result = engine.score(
        texts=texts, ids=ids, task=prepared,
        checkpoint="function-scoring-run", resume=True,
    )
```

A task constructor may derive initial category IDs from unique labels while retaining explicit IDs in its portable schema. Labels used this way must be unique; advanced callers can supply explicit records. Numeric examples add `values` rather than interpreting numeric-looking names.

Support task import/export, result saving, and a prompt/mapping inspection method. Avoid separate APIs for each categorical scale type. Use string IDs or generate deterministic row IDs; never send large numeric identifiers through JSON floating-point conversion.

For R, use `httr2`, `jsonlite`, and a small hashing dependency if needed; base matrices/data frames are sufficient. `httr2` provides bounded retry and `Retry-After` handling, but the package must configure a shared policy explicitly. [S7] Python can use `httpx` and standard-library data structures. The base Python client must not import NumPy, llama.cpp, FastAPI, Torch, or Transformers.

Neither client starts a server, manages a Python environment, or shuts down a remote model. Closing a client closes only its HTTP resources.

## 12. Dataset integrity, retries, and checkpoint/resume

Preserve the input dataset unchanged. Reject invalid encodings rather than applying automatic text repair. Missing and empty inputs retain their original row positions and become explicit diagnostic rows; the clients need not submit them for inference. Duplicate texts with different IDs remain separate observations. Do not silently deduplicate or reorder returned data.

Use modest configurable request chunks constrained by advertised server limits. Submit one chunk at a time initially. Expose progress without text payloads. A user interrupt stops further submissions and preserves already committed work; it does not guarantee cancellation of computation already running remotely.

Retry documented transient statuses, cold starts, and selected transport failures with bounded exponential backoff and jitter, honoring `Retry-After`. Do not retry authentication, schema, token-validation, or fingerprint failures. Proxy errors may be HTML rather than JSON; error handling must preserve useful status information safely.

A timeout after submission is ambiguous: the service might have completed the computation. Retrying is acceptable for this side-effect-free scoring operation, but can duplicate computation and cloud cost. Never claim exactly-once execution. Merge accepted results by input ID and fingerprint, not by appending every retry response.

### 12.1 Checkpoint format and behavior

Use a simple documented local directory containing a JSON manifest and atomically committed JSON chunk files. A server database is unnecessary. Write each chunk to a temporary file, flush it, and rename on the same filesystem; incomplete temporary files are not completed work. Reject concurrent writers to the same checkpoint.

Record ordered IDs, per-item hashes of exact UTF-8 text, task/prepared-task identity, engine fingerprint, client/protocol versions, and completed status. Raw source texts and credentials are excluded by default. Hashes and scores may still be sensitive and are not anonymization.

Resume MUST validate the same dataset order/content and scoring configuration. A changed task, input, model, or effective numerical configuration requires a new run rather than silently combining outputs. Cold restarts with matching fingerprints are allowed. A completed run can be read offline without contacting a service.

Successful rows are never rescored automatically. Persist deterministic input errors; pending work and explicitly retryable failures can be retried. An all-success checkpoint should produce zero scoring requests. Test interruption between response receipt and checkpoint commit and before/after atomic rename.

## 13. Reproducibility and provenance

Every saved result includes the canonical task, ordered categories and values, response strings and token IDs, task/prepared-task hashes, renderer/template identity, and full effective model identity.

Model provenance MUST identify the actual GGUF artifact by content hash, its quantization, and any available repository/revision metadata. Do not identify a model only as a mutable repository name. Record the binding version, llama.cpp build/commit where available, runtime platform, device/offload configuration, context settings, cache precision, and numerical postprocessing version. Include container image digest for hosted runs when available.

Separate stable method/execution fingerprints from administrative metadata such as timestamps, request IDs, hostnames, and process instance IDs. A service restart must not by itself invalidate a prepared task. Conversely, a different weight file, template, or effective numerical setting must invalidate it.

Server-generated prompt-token hashes enable auditing without retaining every full prompt. Provide an explicit local export path for exact prompt previews when needed; do not write them to server logs by default.

Define repeatability tolerances through tests on the pinned stack. Quantization and alternate hardware are not asserted to be numerically identical. The product is sampling-free; it does not offer a separate "strict determinism" mode in v0.1. A seed alone is not a reproducibility mechanism for this method.

Documentation must distinguish software correctness from instrument validation. Encourage researchers to report model/task identity, code mapping, coverage behavior, and any exclusion rules, and to evaluate agreement with relevant reference judgments before substantive use.

## 14. Packaging, model acquisition, and local execution

Maintain one repository with a Python distribution and an R package. Suggested structure:

```text
python/src/llikert/   # client plus lazily imported service modules
r/llikert/           # R package
schemas/                   # task, prepared task, HTTP and checkpoint schemas
tests/fixtures/            # shared protocol and numerical fixtures
tests/integration/         # explicit real-model and deployment checks
deploy/                    # container definitions and Hugging Face guide
docs/                      # quickstarts, scoring specification, compatibility
benchmarks/                # fixed synthetic workloads and manifests
```

Use one Python base installation for the HTTP client and one `server` extra for inference and serving. Extras and installed CLI entry points are supported by Python packaging metadata. [S6] No backend menu or collection of optional provider SDKs is needed. Keep server imports lazy and pin the inference stack in reproducible deployment lockfiles rather than depending on an unversioned branch.

Provide an operator CLI such as:

```text
llikert serve --model /models/model.gguf --host 127.0.0.1 --port 8080
```

The CLI accepts an explicit local GGUF path and a small documented set of execution settings. It reports effective model identity, device/offload, context capacity, and readiness. No silent CPU fallback when GPU operation was explicitly requested. A separate self-check command should diagnose missing native libraries, invalid model paths, template incompatibility, and unavailable requested acceleration.

Do not bundle weights or download them on import. Document explicit model acquisition from an approved source and verification of its filename/revision/hash. Start with a single-file GGUF workflow; automatic model conversion and automatic selection among quantizations are out of scope.

`llama-cpp-python` documents both source builds and selected prebuilt wheels; wheel availability must be verified for the supported platform matrix. [S3] Target local Linux CPU and a tested Linux NVIDIA configuration first. R/Python remote clients should be tested on Windows, macOS, and Linux. Local Apple Metal or Windows inference is not a release promise unless verified in the actual build matrix.

Build the R package so ordinary installation and examples do not require Python, model files, a GPU, or network inference. Use package checks and mocked tests. Publish installation instructions for a development release before treating public-registry acceptance as a completed milestone. Confirm package names, ownership, redistribution permissions, and model licensing before publishing.

## 15. Hugging Face deployment and operational security

Publish one versioned service application image for the initial supported hosted hardware profile, with no model weights baked in. Local execution uses the same service code and protocol. Additional image variants are a later distribution concern, not alternate inference backends.

Hugging Face supports custom containers and mounts the selected model repository at `/repository`. Configure an explicit GGUF filename below that mount rather than downloading a second copy or selecting an arbitrary file. Configure the container port, readiness route, and ingress paths. [S4]

The deployment guide must identify a tested image digest, model artifact, compatible hardware, memory requirements established by testing, startup settings, endpoint visibility, and token setup. Use platform-managed endpoint authentication when hosting there; verify how it interacts with application authentication rather than requiring an unrelated second secret by accident.

For local service use, bind to loopback by default. Non-loopback deployment requires an explicit authentication mode and protected transport through the hosting platform or reverse proxy. Verify TLS in clients and do not forward bearer tokens across redirects to another host. No inference credentials, private paths, tracebacks, or input texts appear in routine logs.

Scale-to-zero can cause cold starts and `503` responses while initialization occurs. [S5] Keep prepared tasks reconstructible, expose honest readiness, and have clients distinguish startup from permanent failure. Use bounded retries, not indefinite waiting. Keep request chunks below the provider's verified request-duration and payload limits; do not assume long synchronous jobs survive an ingress timeout.

Document the operational procedure to pause or remove an endpoint after a dataset run and how to inspect applicable charges. Do not promise a free tier or embed changing prices. The coding agent must not create billable infrastructure without owner authorization.

Remote execution transmits study text to a third-party environment. Explain this plainly. Apply minimal logs, resource limits, and no intentional server-side result persistence. Treat institutional approvals and data handling as deployment responsibilities, not as guarantees conferred by using a particular provider.

## 16. Verification and acceptance tests

Most tests MUST run without a model download, GPU, endpoint credentials, or Python in an R-only environment. Use a fake logits adapter and a mock HTTP service for numerical, protocol, and client tests. Real-model checks are explicit integration jobs using a pinned small model and checksum.

| Test group | Required evidence |
| --- | --- |
| Numerical core | Hand-calculated vocabulary fixtures; full-vocabulary mass; candidate normalization; optional expected values; log-space underflow; nonfinite errors. |
| Candidate validation | One-token success; multi-token and duplicate-ID rejection; whitespace/boundary cases; control-token rejection; nonnumeric categories. |
| Prompt construction | Golden rendered prompts and tokens; exact assistant position; few-shot consistency; Unicode and reserved-marker behavior. |
| Reference parity | Native output matches independent full-vocabulary reduction; no candidate-specific inference; raw logits remain unmodified. |
| Optimized parity | Sequential versus batched/chunked/cached runs, reordered inputs, repeated requests, and single-item batches. |
| Data integrity | Zero rows, one row, all-failed rows, missing/empty inputs, duplicate IDs, duplicate texts, fixed matrix shapes, and preserved order. |
| Error recovery | Oversized input, memory failure, native-state reset, bounded queue, startup errors, bad JSON, and safe messages. |
| Transport | Authentication, TLS policy, timeouts, transient status retries, non-JSON proxy errors, protocol and fingerprint mismatch. |
| Checkpointing | Interrupt/restart, partial writes, no rescore of successes, changed inputs/tasks rejected, offline read, and writer conflicts. |
| Cross-language | Identical fixtures yield the same categories, values, probabilities, missingness, manifests, and checkpoint interpretation. |
| Installation | Clean remote-only R/Python environments; no heavy imports; service boot on each claimed inference platform. |
| Hosted service | Authenticated prepare/score, readiness, path routing, model mount, cold restart, and matching fingerprints. |

For numerical parity, report maximum absolute differences in conditional probabilities, candidate log probabilities, coverage/log coverage, and expected values when defined. Correlation alone is insufficient. State tolerances before judging a test; isolate postprocessing precision from backend arithmetic differences.

Use a synthetic nonpolitical benchmark corpus with short, medium, and long texts and several category counts. Measure warm and cold behavior separately and compare identical weights and prompts. Include actual native batching configuration, not merely HTTP list length.

Integration reports must distinguish executed/passed, failed, and not executed. A mocked endpoint does not establish Hugging Face deployment compatibility. A tiny smoke-test model does not establish production scoring quality. Unverified GPU or paid-cloud environments must remain explicitly unverified.

## 17. Development milestones and handoff outputs

### M0. Feasibility and compatibility proof

Pin a llama.cpp/binding combination and select a small, license-compatible direct-answer GGUF model. Demonstrate model loading, tested chat rendering, exact candidate token IDs, final-position raw logits, and full-vocabulary normalization. Show one nominal and one numeric task with saved outputs.

Probe template support, output-buffer ownership, context limits, and the local build path before writing public wrappers. Inspect current Hugging Face custom-container requirements and establish a container boot/route smoke test locally. Record open environment dependencies without claiming cloud execution.

**Exit gate:** A standalone numerical proof and smoke test pass. Any backend limitation affecting the scoring contract is raised explicitly; it is not bypassed with generation.

### M1. Core scorer and versioned service

Implement task schemas, canonicalization, preparation, the sequential reference adapter, result/error contracts, readiness/info/prepare/score routes, authentication configuration, and resource limits. Publish shared JSON fixtures and the first engine fingerprint definition.

**Exit gate:** A real local service scores a dataset chunk correctly; malformed tasks and invalid texts behave as specified; the numerical and schema suites pass.

### M2. Research-ready R and Python clients

Implement the same prepare/score workflow, fixed-shape result objects, task/result serialization, progress, retries, checkpoint/resume, and empty/missing handling. Package the lightweight clients without inference dependencies. These clients can be developed in parallel after M1's schemas stabilize.

**Exit gate:** Both clients complete and resume the same fixture run, preserve identical semantics, and pass remote-only clean-install checks.

### M3. Bounded performance and real-model conformance

Add safe prefill batching, tune request/native batch defaults, and implement multi-sequence or shared-prefix optimizations only where justified. Stress-test context and memory failures, repeated runs, state isolation, and cross-language parity. Publish a benchmark and a supported model/platform matrix.

**Exit gate:** The chosen practical execution path passes reference parity and completes the benchmark within documented memory limits. Performance is measured, not inferred from architecture.

### M4. Distribution and hosted deployment

Produce the versioned service image, locked build, local guide, Hugging Face deployment recipe, troubleshooting guide, and release artifacts. Exercise a real hosted endpoint only with authorized access. Record any deployment tests not executed.

**Exit gate:** Local and hosted installation evidence supports every compatibility claim. If live cloud validation is unavailable, label that recipe experimental rather than presenting it as verified.

### M5. Release review

Review scoring semantics, security defaults, provenance, licenses, package checks, example reproducibility, and outstanding defects. Remove unused scaffolding and undocumented fallback behavior. Provide a short agent handoff containing commands run, test results, artifact locations, known limitations, and remaining owner actions.

**Exit gate:** The definition of done below is satisfied for the published support matrix. There is no requirement to add another backend to reach v0.1.

## 18. Definition of done and instructions to the coding agent

The release is complete when a researcher can define a nominal or numeric task, inspect its validated response tokens, score an identified dataset through either client, interrupt and resume safely, and save a self-describing result. Hosted R use has no local Python requirement. Local inference has an explicit, tested installation path.

Every returned probability must have the meaning in Section 4. Unsupported codes and model/template combinations fail clearly. No input is silently dropped, repaired, truncated, or scored under a different model. All supported execution paths pass the reference tests; advertised environments have corresponding evidence.

Deliver the source repository, installable client packages, service CLI, schemas/fixtures, tests, container recipe/image identity where built, deployment guide, examples, benchmark report, and compatibility matrix. Include a concise methods-description example for a research paper without claiming validation results that were not measured.

**Begin with M0, then implement the shortest complete path through M1 and M2.** Do not spend the first iteration building optional integrations, abstract backend registries, dashboards, or public plugin APIs. Keep routine implementation decisions in short decision records. Escalate only contradictions affecting probability semantics, demonstrable backend blockers, sensitive-data handling, external publishing, or billable actions.

A methods-correct smaller release is preferable to a larger package with hidden approximations. Unimplemented optional features should remain absent, not appear as accepted arguments that do nothing.

## 19. Deferred extension: multi-token candidates

This section records the future mathematical direction; it is not a v0.1 implementation requirement.

For a candidate token sequence, evaluate its conditional token probabilities and multiply them, preferably by summing log probabilities. No terminator is required for prefix-event scoring. Prefix overlaps can be disambiguated by assigning continuations to their longest matching candidate prefix.

For candidates `[1]` and `[1, 0]`, define:

```text
q_long  = P(1 | x) * P(0 | x, 1)
q_short = P(1 | x) - q_long
```

Subtract the joint longer-prefix probability, not the conditional probability `P(0 | x, 1)` alone. For nested candidates, subtract only the masses of immediate descendant candidate prefixes to avoid double subtraction. The residual shorter label includes unlisted continuations that do not match a longer candidate; it is not an exact terminated-answer probability.

Any extension must keep the current single-token behavior unchanged, specify these events explicitly, and pass new reference tests. It must not dictate a different v0.1 backend or delay the initial release.

## 20. Technical references

Technical capability references were checked on September 16, 2026. They establish available interfaces, not that this planned application has already been implemented or benchmarked. Pin actual versions and record exact compatibility evidence during development. The product requirements above are design decisions, not claims made by these sources.

**[L1] Supplied `likert_client.py`.** Module description and HTTP client: supplied file lines 2-7 and 32-78. Demonstrates using a shared server instead of loading the model in each language process. Its automatic process-management behavior is not retained.

**[L2] Supplied `likert_server.py`.** Candidate aggregation and full-vocabulary log-softmax: supplied file lines 621-633 and 655-685. Provides conceptual background only; no existing implementation or benchmark claim is a release requirement.

**[S1] llama.cpp native API, `include/llama.h`.** Batch representation, output logits, tokenization, and chat-template interface. The template helper documents a limited template set.  
`https://raw.githubusercontent.com/ggml-org/llama.cpp/master/include/llama.h`

**[S2] llama-cpp-python API reference.** Low-level bindings and model/tokenization operations.  
`https://llama-cpp-python.readthedocs.io/en/latest/api-reference/`

**[S3] llama-cpp-python installation documentation.** Source builds, native dependencies, and available wheel paths.  
`https://llama-cpp-python.readthedocs.io/en/latest/`

**[S4] Hugging Face Inference Endpoints: Deploy with your own container.** Custom service deployment, readiness, model mounting, and image configuration.  
`https://huggingface.co/docs/inference-endpoints/engines/custom_container`

**[S5] Hugging Face Inference Endpoints: Autoscaling.** Scale-to-zero behavior and cold-start responses.  
`https://huggingface.co/docs/inference-endpoints/guides/autoscaling`

**[S6] Python Packaging User Guide: Writing your pyproject.toml.** Optional dependencies and installed CLI entry points.  
`https://packaging.python.org/en/latest/guides/writing-pyproject-toml/`

**[S7] httr2: Automatically retry a request on failure.** Retry limits, transient failures, and Retry-After behavior.  
`https://httr2.r-lib.org/reference/req_retry.html`
