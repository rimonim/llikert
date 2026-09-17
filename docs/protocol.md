# LLikert HTTP protocol v1

This is the contract between the scoring service and the R and Python clients. The machine-readable schemas are in `schemas/*.v1.json`; the OpenAPI document is served at `/openapi.json` with `--auth none`, and the same schemas are exported by `llikert export-schemas`. The reasons behind individual choices are in `docs/decisions/0003-m1-service.md`.

Every response carries the header `LLikert-Protocol: 1`. Request bodies must be strict UTF-8 JSON: no `NaN` or `Infinity`, no duplicate keys, no unknown fields. Send credentials as `Authorization: Bearer <token>`; they never belong in a body.

## Endpoints

| Method and path | Auth | Purpose |
|---|---|---|
| `GET /health` | never | `200 {"status":"ready"}` once the model is loaded and the smoke check has passed; otherwise `503` with `Retry-After` |
| `GET /v1/info` | yes | Protocol version, engine fingerprint and identity, execution details, limits |
| `POST /v1/prepare` | yes | Validate a task against the loaded model; returns the portable prepared artifact |
| `POST /v1/score` | yes | Score one chunk of identified items with a prepared artifact |

## Task (`task.v1.json`)

```json
{
  "schema_version": 1,
  "name": "Communicative function",
  "instructions": "Classify the text's primary communicative function.",
  "categories": [
    {"id": "description", "label": "description", "response": "A", "value": null},
    {"id": "question",    "label": "question",    "response": "B", "value": null},
    {"id": "request",     "label": "request",     "response": "C", "value": null}
  ],
  "ordered": false,
  "examples": [{"item": "Where is it?", "category_id": "question"}],
  "prompt": {
    "system": "{instructions}\n\n{scale}\n\n{answer_instruction}",
    "user": "Text:\n<text>\n{item}\n</text>",
    "scale": "Response codes:\n{codes}",
    "code": "{response} = {label}",
    "code_separator": "\n",
    "answer_instruction": "Answer with exactly one of the response codes listed above and nothing else."
  }
}
```

- **Categories:** at least two, in a meaningful order.
- **`id`:** unique; `id` and `expected_value` are reserved.
- **`response`:** nonempty and unique; whitespace counts.
- **`value`:** a finite number for every category or null for all of them.
- **Strings:** never trimmed or Unicode-normalized. `instructions` may be empty.
- **`prompt`:** optional; missing fields take the defaults shown. `system` may be null for no system message. Templates, placeholders and validation rules are in `docs/prompts.md`; invalid formats are `422 invalid_task`.
- **Identity:** the task hash is the SHA-256 of the RFC 8785 (JCS) canonical form with defaults filled in, written `sha256:<hex>`. Fixtures are in `tests/fixtures/tasks/`.

## Prepare

Request:

```json
{"protocol_version": 1, "request_id": "optional", "task": { ... }, "preview_text": "optional research text"}
```

On `200` the response contains:

- **`prepared`:** the artifact to store and send back with every score request. It holds:
  - `prepared_schema_version`, `task`, `task_hash`, `engine_fingerprint`
  - `profile {name, template_sha256}`
  - `boundary_policy`, `tail_tokens`
  - `categories[] {id, label, response, value, token_id, token_piece}`
  - `prepared_hash`
- **`diagnostics`:** `n_prompt_tokens_without_item`, `n_ctx`, `max_item_tokens_approx`, `warnings[]` (prompt-format warnings: `unused_instructions`, `empty_instructions`, `unused_answer_instruction`, `scale_not_in_prompt`; `response_whitespace`)
- **`preview`:** `example {messages, prompt, n_prompt_tokens}` rendered with a made-up example item, plus `text` only when `preview_text` was sent. `messages` are the chat messages (`role`, `content`), and `prompt` is the exact text in the model's chat format. Previews are never logged and never part of the artifact.
- **`request_id`**

On `422 invalid_response_codes`, `error.details` holds:

- **`problems[]`:** one entry per failing code, with `category_id`, `response`, `problem` and the `tokens` actually produced. `problem` is one of `multi_token`, `no_token`, `prefix_unstable`, `piece_mismatch`, `duplicate_token`, `control_token`, `end_of_generation_token`, `byte_token`, `unknown_token`, `unused_token` or `user_defined_token`.
- **`suggestions.prefix.per_category[]`:** `category_id`, `from_response`, `from_label` and `collision`. The two `from_` fields are the first token of the supplied response or label, offered only when that token is itself a valid, distinct code.
- **`suggestions.prefix.complete`:** `{source, responses}` when every category has a unique valid prefix, otherwise null.
- **`suggestions.generic[]`:** `{kind: "letters"|"digits", responses}`.

Nothing is remapped automatically. A prefix code changes the prompt the model sees, and so changes the instrument.

## Score

Request:

```json
{
  "protocol_version": 1,
  "request_id": "optional",
  "prepared": { ...exactly as returned by prepare... },
  "items": [{"id": "r1", "text": "Where is the station?"}, {"id": "r2", "text": null}]
}
```

- **Items:** `id` is a nonempty string, unique within the request. `text` is a string or `null`; any other type is a 422.
- **Chunk size:** at most `limits.max_items_per_request` items per request.
- **Checks before any scoring:** first the fingerprint (`409 engine_fingerprint_mismatch`), then the full artifact against the service's own preparation (`409 prepared_task_mismatch`).

Response `200`:

```json
{
  "protocol_version": 1,
  "request_id": "...",
  "engine_fingerprint": "sha256:...",
  "prepared_hash": "sha256:...",
  "category_ids": ["description", "question", "request"],
  "results": [
    {
      "id": "r1", "status": "ok", "error": null,
      "candidate_log_probs": [-9.1, -0.52, -0.9],
      "candidate_probs": [0.00011, 0.59, 0.40],
      "probabilities": [0.00011, 0.5959, 0.4040],
      "log_coverage": -0.00002, "coverage": 0.99998,
      "n_prompt_tokens": 71,
      "prompt_token_sha256": "sha256:...",
      "warnings": []
    },
    {
      "id": "r2", "status": "missing_input",
      "error": {"code": "missing_input", "message": "item is missing"},
      "candidate_log_probs": [null, null, null], "candidate_probs": [null, null, null],
      "probabilities": [null, null, null], "log_coverage": null, "coverage": null,
      "n_prompt_tokens": null, "prompt_token_sha256": null, "warnings": []
    }
  ],
  "meta": {"elapsed_ms": 83.1, "execution": { ... }}
}
```

- **Result order:** one result per item, in request order.
- **Arrays:** always length K, in `category_ids` order.
- **`expected_value`:** present only when the task has values (`Σ value·p`), and null for failed items.
- **Log values:** kept even when a probability underflows to 0.

| Item status | Meaning |
|---|---|
| `ok` | Scored |
| `missing_input` | `text` (the item) was null |
| `empty_input` | `text` was `""`; a whitespace-only item is scored |
| `invalid_encoding` | `text` contained a lone surrogate |
| `context_limit` | The rendered prompt is longer than `n_ctx`; `n_prompt_tokens` gives the length |
| `boundary_error` | The item's prompt does not end at the prepared answer boundary |
| `numerical_error` | The model produced non-finite logits |
| `inference_error` | Native evaluation failed; the service recovers or reports itself unavailable |

Item warnings: `reserved_marker_text` means the text contained special-token text, which was scored as ordinary text.

## Errors

Every non-200 response has the shape `{"error": {"code", "message", "details"?, "request_id"?}}`.

| Status | Codes | Client action |
|---|---|---|
| 400 | `invalid_json` | Fix the request; don't retry |
| 401 | `unauthorized` | Fix credentials; don't retry |
| 404, 405 | `not_found`, `method_not_allowed` | Check the URL or proxy routing |
| 409 | `engine_fingerprint_mismatch`, `prepared_task_mismatch` | Re-prepare the task, or start a new run; don't retry |
| 413 | `body_too_large`, `too_many_items` | Use smaller chunks (`body_too_large` has no details) |
| 422 | `invalid_request`, `invalid_task`, `invalid_response_codes`, `task_exceeds_context`, `render_error`, `boundary_error`, `duplicate_ids`, `invalid_encoding` | Fix the input; don't retry |
| 429 | `queue_full` | Retry after `Retry-After` |
| 503 | `starting`, `startup_failed`, `service_unavailable` | Retry after `Retry-After`, with bounded waiting |
| 500 | `internal_error` | Report it; bounded retry is acceptable |

A proxy in front of the service may return non-JSON bodies, for example on 502/504 or on a cold start; clients must cope with that.
