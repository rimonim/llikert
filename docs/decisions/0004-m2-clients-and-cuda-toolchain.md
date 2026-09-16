# 0004: M2 clients; CUDA toolchain correction and supported execution settings

**Status:** accepted 2026-09-16
**Context:** M2 built the R and Python clients. On the way, the container probe exposed a CUDA toolchain error in the M0/M1 build that changes the evidence for the supported profile.

## A. CUDA toolchain correction (supersedes parts of 0002 F8/D23 and 0003 evidence)

**Finding.** The M0 and M1 CUDA wheel was compiled with CUDA 12.9 `nvcc`, but CMake linked the Ubuntu CUDA **11** runtime and cuBLAS from `/lib/x86_64-linux-gnu` (`libcudart.so.11.5`, `libcublas.so.11.7`). The container probe exposed this, because CUDA 12.9 images cannot load CUDA 11 libraries. Every M0 and M1 CUDA measurement therefore used cuBLAS 11.7.

**Fix.**
1. `tools/build-llama-cpp-python.sh` pins `CUDAToolkit_ROOT` and `CMAKE_CUDA_COMPILER` to `CUDA_HOME`. The rebuilt wheel links `libcudart.so.12.9.37` and `libcublas.so.12.9.0.13`.
2. The engine identity now includes `math_libraries`: the resolved file names of the loaded CUDA runtime and BLAS libraries. A toolchain change therefore changes the engine fingerprint.

**Re-measurement on the correct toolchain.** The D24 reference-fidelity gate (`tools/fidelity_gate.py`) was rerun with the Qwen3-4B-Instruct-2507 F32 weights on an RTX 4090, 28 items, tolerances max |Δp| ≤ 0.005 and max |Δlog q| ≤ 0.1:

| Flash attention | KV cache | Batch | max \|Δp\| | max \|Δlog q\| | ms/item | Gate |
|---|---|---|---|---|---|---|
| auto (the M1 default) | f16 | 512 | 0.0037 | 0.102 | 27 | **fail** |
| off | f16 | 512 | 0.0030 | 0.121 | 28 | fail |
| auto | f32 | 512 | 0.0037 | 0.102 | 28 | fail |
| **off** | **f32** | **512** | **0.0044** | **0.045** | **28** | **pass** |
| off | f32 | 1 | 0.00001 | 0.0002 | 1155 | pass |

**Decision (refines D23).** The supported profile is F32 weights on CUDA 12.9 / cuBLAS 12.9 with `--flash-attn off --kv-type f32 --batch-size 512`. Those are now the defaults for `llikert serve`, `llikert selfcheck` and `LlamaConfig`. The tolerances were not changed after seeing results. The margin on |Δp| is small (0.0044 against 0.005), so M3 will re-examine the gate on a larger corpus before the release claim.

**Test change.** The small-model integration test no longer compares probabilities with the M0 spike, which was recorded under the old settings. It checks exact token parity with the spike, plus the service's numbers against an independent scalar full-vocabulary reduction of the same logits, to 1e-12.

## B. Container probe (M0 item completed)

- **Base image:** `nvidia/cuda:12.9.0-runtime-ubuntu22.04`, plus `libgomp1`, which the runtime image lacks. `12.9.1` needs a newer driver than the host's 575.51.03.
- **GPU access:** containers run with `--runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility`. `--gpus all` fails until the Docker daemon is restarted after installing the toolkit.
- **Result:** the probe loaded the model from a `/repository` mount, offloaded to CUDA, and served `/health` with 200 after 2 s.

## C. Client decisions

1. **Both clients implement the same algorithm**, tested against the same replayed fixtures (`tests/fixtures/protocol/fake-service.json`). The fixtures come from the real service code running over the deterministic fake adapter (`tools/generate_client_fixtures.py`). Both clients must reproduce `tests/fixtures/runs/expected-result-*.json` exactly, apart from `created_at` and `client`, for chunk sizes 1, 5 and 64.
2. **Retry policy (D16):**
   - Up to 6 tries, with full-jitter exponential backoff capped at 60 s. `Retry-After` is honored and capped at 60 s.
   - Retried: 429, 502, 503, 504, connection failures and timeouts.
   - A 413 halves the chunk size.
   - Non-JSON error bodies are reduced to the status plus 200 characters with markup stripped.
   - Redirects are not followed.
3. **Connect (D15)** polls `/v1/info` while it gets 503 or no connection, up to `wait` seconds (default 300). `scale_up_timeout` optionally sends `X-Scale-Up-Timeout`.
4. **Client-side statuses.**
   - `NA`/`None` (and pandas NaN or NA) become `missing_input`; `""` becomes `empty_input`; invalid UTF-8 in R, or a lone surrogate in Python, becomes `invalid_encoding`. These records match the service's own records exactly, messages included, and are never sent.
   - Whitespace-only text is sent.
   - R transcodes strings marked latin1 or native to UTF-8, which is transcoding, not repair.
5. **Ids (D13).**
   - Character, factor, integer and whole-number doubles below 2^53 are accepted; doubles are formatted without scientific notation.
   - Duplicate or missing ids are an error.
   - The default ids are `"1"…"N"`.
6. **Checkpoints** follow `docs/checkpoint-format.md`. `inference_error` is the only retryable item status. Python writes checkpoint and result files with owner-only permissions (0600, from `mkstemp`); R uses the process umask.
7. **R result (C2, D19).**
   - The main object is a tibble subclass `llikert_result`: `id`, then `expected_value` (numeric tasks only), then one column per category id.
   - The item records live in the `"llikert"` attribute.
   - Accessors (`llikert_result_*`) align to the ids present, so `dplyr::filter()` and `arrange()` keep them consistent. Dropping `id` makes the accessors error.
   - The printed header shows status counts and coverage, never texts.
8. **R JSON uses `digits = I(17)`.** jsonlite's `digits = NA` writes 15 significant digits and silently changes doubles such as `0.30000000000000004`.
9. **R tokens** are stored in an environment inside the engine object, so `print()` and `str()` never show them. Saving an engine object with `saveRDS()` would still store the token, so the documentation says not to.
10. **Prepared task file:** `{"prepared": <artifact>}`, via `write_prepared_task()`/`read_prepared_task()` in R and `PreparedTask.save()`/`load()` in Python.
11. **Python client stack:** `httpx` only, with frozen dataclasses. `ScoreResult.to_pandas()` imports pandas lazily and uses the same column order as the R tibble.
12. **Test layout:** Python tests are split into `tests/server` (needs the server extra) and `tests/client` (base install only). `tools/sync_fixtures.py` copies the shared fixtures into the R package, and a test fails if the copies drift.

## D. Evidence (2026-09-16)

- **Python, full environment:** 202 passed (client, server, fixture sync and schema validation), 5 model tests deselected.
- **Python, clean base-only install** (fresh venv, `pip install ./python`, only `httpx` installed): 63 client tests passed, and no heavy imports.
- **R, in an R-only container** (`rocker/r-ver:4.6.1`, no Python): `R CMD check --no-manual` finished with Status: OK, tests included.
- **Cross-language:**
  - R resumes a partial checkpoint written by Python, and Python resumes one written by R. In both directions exactly the 11 remaining items are re-sent, and the result equals the shared expected result.
  - Files written by both clients validate against the published schemas.
- **Real service (F32, CUDA 12.9, new defaults):** the Python client (chunk size 7) and the R client (chunk size 5) scored the same 24 items with checkpoints. Their saved results are identical down to exact float equality, and the missing, empty and reserved-marker items were handled identically.
- **Model integration:** `pytest -m model`, 5 passed, including the D24 gate under the new defaults (max |Δp| 0.0044, max |Δlog q| 0.045).
