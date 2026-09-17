# Result and checkpoint files (schema v1)

The R and Python clients write the same two file formats and can each read what the other wrote. The machine-readable schemas are `schemas/result.v1.json`, `schemas/checkpoint-manifest.v1.json` and `schemas/checkpoint-chunk.v1.json`. Tests validate files written by both clients against them.

## Result file

`write_result()` (R) and `ScoreResult.save()` (Python) write a single JSON object with these fields:

| Field | Content |
|---|---|
| `result_schema_version` | `1` |
| `protocol_version` | `1` |
| `created_at` | UTC time of writing (ISO 8601) |
| `client` | `{language, package, version}` |
| `engine_fingerprint` | The engine the scores came from |
| `engine`, `execution` | Engine identity and execution details from `/v1/info` |
| `prepared` | The full prepared task: canonical task, response strings, token ids, hashes, template profile |
| `category_ids` | Column order for every probability array |
| `items` | One record per input, in input order |

Each item record is the service's result record (see `docs/protocol.md`) plus `text_sha256`, the SHA-256 hex digest of the item's UTF-8 text, or null when the item is missing. An item that has not been scored yet has status `pending`; this appears only when an incomplete checkpoint is read with `allow_incomplete`.

Numbers are written so that every double round-trips exactly. The R client uses `jsonlite` with `digits = I(17)`; the jsonlite default `digits = NA` keeps only 15 significant digits and would change values.

A result file contains no texts and no credentials. Hashes and scores can still be sensitive, and hashing is not anonymization.

## Checkpoint directory

```text
run-dir/
  manifest.json            run identity (written once, atomically)
  .lock/owner.json         present while a client is writing
  chunks/chunk-000001.json committed results, one file per commit
```

### `manifest.json`

- **Version and provenance:** `checkpoint_schema_version`, `created_at`, `client`, `protocol_version`
- **Engine:** `engine_fingerprint`, `engine`, `execution`
- **Task:** `prepared`
- **Dataset:** `ids`, all item ids in input order, and `text_sha256`, one hash per id with null for missing text

### Chunk files

Each chunk file holds `checkpoint_schema_version`, `sequence`, `engine_fingerprint`, `prepared_hash`, and `results`, a list of item records.

### How writes are committed

1. A chunk is written to `chunks/.tmp-*.json` in the same directory, then renamed. Python also flushes to disk before renaming; base R cannot, so R commits are atomic but not guaranteed to survive a power loss.
2. Files named `.tmp-*` are never read. A write that was interrupted before the rename counts as not done.
3. Results count as completed only after the rename. If a client stops between receiving a response and committing it, that chunk is scored again on resume. Scoring has no side effects, so the only cost is repeated computation.

### Merging chunks

Chunks are read in `sequence` order and merged by item id:

- A later record replaces an earlier one only when the earlier status is retryable. At present that is `inference_error`.
- Successes and deterministic input errors are final: `missing_input`, `empty_input`, `invalid_encoding`, `context_limit`, `boundary_error` and `numerical_error`.

The first chunk of a run holds the input errors the client decides itself (missing, empty, invalid encoding), so a resumed run does not recompute them.

### Resuming

With `resume = TRUE` or `resume=True`, the client loads the manifest and compares it with the current call:

- **Checks:** the prepared hash, the engine fingerprint, the ids in order, and every text hash must be identical. Otherwise the client stops with an error that names what changed, and you start a new checkpoint.
- **What is scored:** only items without a record, or whose record has a retryable status. If every item is already final, the resume sends no scoring requests.
- **Chunk size:** it may differ between runs, because merging is by id.
- **Service restarts:** a cold-started service with the same engine fingerprint is fine.

A checkpoint that exists is never overwritten. Without `resume`, the client reports the existing checkpoint and stops. A non-empty directory without `manifest.json` is refused.

### Locking

A client creates `.lock/` with an atomic directory creation, writes `owner.json` (host, pid, time, client) into it, and removes it on normal completion, on error and on interrupt.

If `.lock/` already exists, the client refuses to write. After a crash, pass `force_unlock` once you are sure no other session is using the checkpoint.

### Reading offline

`read_checkpoint(path)` (R and Python) builds a result from the manifest and chunks without contacting a service. An incomplete checkpoint is an error unless `allow_incomplete` is set, in which case unscored items appear as `pending`.
