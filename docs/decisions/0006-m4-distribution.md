# 0006: M4 — service image, locked build, deployment recipe

**Status:** accepted 2026-09-17
**Context:** M4 packages the service for local and hosted use. Hosting on Hugging Face is deferred by the owner, so no image or model has been published and no endpoint created.

## Decisions

1. **One service image** for the supported hosted profile: `deploy/Dockerfile.cuda` → `llikert-service:0.1.0.dev0-cuda12.9`.
   - **Builder stage:** compiles llama-cpp-python v0.3.35 against llama.cpp v0.4.1 with the image's own CUDA 12.9.0 toolkit, for GPU architectures 8.6 (Ampere, e.g. A10G) and 8.9 (Ada, e.g. L4, RTX 4090). It uses the same `tools/build-llama-cpp-python.sh` as local builds.
   - **Runtime stage:** the CUDA 12.9.0 runtime, CPython 3.10, `libgomp1`, and an unprivileged user (uid 10001).
   - Weights are never included.
2. **Everything pinned:**
   - CUDA base images by digest (devel `sha256:26cd7d00…`, runtime `sha256:d1ad87e9…`);
   - build tools in `deploy/requirements-build.lock` and server dependencies in `deploy/requirements-server.lock`, both installed with `--require-hashes`;
   - the llama-cpp-python and llama.cpp commits, verified by the build script.

   Local wheel builds were previously pulling unpinned build tools through pip's isolated environment. The locked tools now apply to the image build, and local builds can use them with `PIP_WHEEL_ARGS=--no-build-isolation`.
3. **The image build compiles its own native wheel** instead of copying the developer's wheel. A rebuilt native library is a different engine build, so the image is verified on its own with the HTTP fidelity gate.
4. **Configuration by environment** (`deploy/entrypoint.sh`):
   - Required: `LLIKERT_MODEL`. Recommended: `LLIKERT_MODEL_SHA256`.
   - Optional: `LLIKERT_AUTH` (default `token`), `LLIKERT_API_TOKEN`, `LLIKERT_PORT`, `LLIKERT_N_CTX`, `LLIKERT_MAX_ITEMS`, `LLIKERT_QUEUE`.
   - Numerics-affecting execution settings are not configurable in the image.
   - Passing arguments runs `llikert <args>`, for example `selfcheck`.
   - `NVIDIA_TF32_OVERRIDE=0` is also set in the image, in addition to the service setting it itself.
5. **A failed startup exits the process** (status 3) after the error is logged. The HTTP server still answers `/health` with 503 while loading, but a model hash mismatch, missing library, unsupported template or memory failure no longer leaves a server that is never ready. A container supervisor or endpoint platform sees the failed start in its logs. The underlying `create_app()` keeps the process alive by default, for embedding and tests.
6. **`tools/fidelity_gate_http.py`** runs the D24 gate against any running service, over HTTP, using only the base client. Prompt-token parity with the reference is checked remotely through `prompt_token_sha256`. It is the verification step for containers and hosted endpoints on their own hardware.
7. **Hugging Face recipe** (`deploy/hf-endpoint.md`) is marked **experimental and not executed**:
   - a protected endpoint with `LLIKERT_AUTH=platform`, the image deployed by digest, the model in a private repository mounted at `/repository`, max 1 replica until the gate passes;
   - collaborators connect with their Hugging Face token and `scale_up_timeout`;
   - a nine-item verification checklist and cost-control steps.
8. **Release artifacts** (`tools/build-release.sh` → `dist/`, not committed): Python sdist and wheel, R source package, `SHA256SUMS`, and `release-manifest.json` linking the artifacts to the source revision, lockfile hashes, image id, pinned stack and supported model hash. Versions stay at development numbers (`0.1.0.dev0`, `0.1.0.9000`) until the M5 release review.

## Evidence (2026-09-17, RTX 4090, driver 575.51.03)

- **Image:** `llikert-service:0.1.0.dev0-cuda12.9`, 5.0 GB. The acceptance test ran on image id `sha256:08efc067e80f…`. After the commit, the image was rebuilt only to set the revision label to `116cef8`, giving id `sha256:89bf6a14596b…`; all layers are otherwise cached and identical. A rebuild after a source-only change reuses the cached native stage and takes about 7 s.
- **`deploy/test-container.sh`:** 16 of 16 checks passed.
  - **Startup:** a missing `LLIKERT_MODEL`, `LLIKERT_AUTH=none` on 0.0.0.0 and a short token are all refused. A wrong `LLIKERT_MODEL_SHA256` exits with status 3 and a message.
  - **Readiness:** `/health` is not 200 before the service is ready, and becomes ready after 15 s (hashing, loading, warm-up, smoke check). It needs no credentials and returns only `{"status":"ready"}`.
  - **Auth and user:** `/v1/info` returns 401 without a token, `/openapi.json` is not served in token mode, and the process runs as uid 10001.
  - **HTTP fidelity gate:** 450 items, 0 prompt-token mismatches, max |Δp| 0.000093, max |Δlog q| 0.0011.
  - **Clients:** the Python and R clients scored the same texts with checkpoints, including missing and empty items, and saved identical results.
  - **Logs:** the service log contains neither a canary text nor the token.
  - **Restart:** the engine fingerprint is identical after a restart.
- **Same engine as the host.** The container's engine fingerprint (`sha256:3f0496367bd3…`) equals the host service's, and its gate results are identical, to the last digit, to the host run. The image's own native build is the same engine as the verified local build.
- **Release artifacts:** `tools/build-release.sh` produces the Python wheel and sdist, the R source package, `SHA256SUMS` and `release-manifest.json`.

## Not executed

- Pushing the image to a registry, uploading the model to Hugging Face, and creating an endpoint (deferred by the owner). The Hugging Face checklist in `deploy/hf-endpoint.md` is entirely unverified.
- Any GPU other than the RTX 4090, including the L4 and A10G proposed for hosting.
- The CPU-only service build.
