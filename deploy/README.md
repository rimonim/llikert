# Service container

Written for: operators running the LLikert scoring service in a container, locally or on a GPU host.

The image holds the service code and its pinned native build. Model weights are never included; you mount them.

| Image | Contents |
|---|---|
| `llikert-service:0.1.0.dev0-cuda12.9` | CUDA 12.9.0 runtime (digest-pinned), CPython 3.10, llama-cpp-python 0.3.35 built against llama.cpp v0.4.1 for GPU architectures 8.6 and 8.9, and server dependencies from `deploy/requirements-server.lock` (hash-checked) |

Host requirements:
- an NVIDIA driver supporting CUDA 12.9.0 (575.51 or newer);
- the NVIDIA Container Toolkit;
- a GPU with at least 24 GB of memory for the F32 profile.

Only an RTX 4090 has passed the fidelity gate so far (`docs/compatibility.md`).

## Build

```bash
docker build -f deploy/Dockerfile.cuda -t llikert-service:0.1.0.dev0-cuda12.9 \
  --build-arg SOURCE_REVISION=$(git rev-parse HEAD) .
```

The build compiles CUDA kernels and takes roughly 15 minutes. Base images, build tools and Python dependencies are all pinned.

## Run

```bash
export LLIKERT_API_TOKEN=$(python3 -c "import secrets; print(secrets.token_hex(32))")
docker run --rm --gpus all -p 127.0.0.1:8080:8080 \
  -v "$PWD/models:/repository:ro" \
  -e LLIKERT_MODEL=/repository/qwen3-4b-instruct-2507-f32.gguf \
  -e LLIKERT_MODEL_SHA256=a5733d5a25e8b824e1ef9d7e75449ffd8f8ec582de8ac253420e4423a7f0e357 \
  -e LLIKERT_API_TOKEN \
  llikert-service:0.1.0.dev0-cuda12.9
```

`/health` returns 503 while the model is hashed, loaded and warmed up (about 25 s), then 200. The Docker `HEALTHCHECK` uses the same route.

| Variable | Default | Meaning |
|---|---|---|
| `LLIKERT_MODEL` | required | GGUF path inside the container |
| `LLIKERT_MODEL_SHA256` | none | Expected file hash; startup fails on mismatch. Recommended |
| `LLIKERT_AUTH` | `token` | `token` (bearer token from `LLIKERT_API_TOKEN`), or `platform` (only behind a platform that authenticates every request) |
| `LLIKERT_API_TOKEN` | none | Required for `token`; at least 16 characters |
| `LLIKERT_PORT` | `8080` | Listening port |
| `LLIKERT_N_CTX` | `4096` | Context length in tokens. A larger value changes the engine fingerprint |
| `LLIKERT_MAX_ITEMS` | `64` | Items per request |
| `LLIKERT_QUEUE` | `4` | Pending requests before `429` |

Execution settings that affect the numbers (device, batch size, flash attention, KV type, TF32) stay fixed at the supported profile.

Diagnostics with the same image:

```bash
docker run --rm --gpus all -v "$PWD/models:/repository:ro" llikert-service:0.1.0.dev0-cuda12.9 \
  selfcheck --model /repository/qwen3-4b-instruct-2507-f32.gguf
```

## Verify on your hardware

A different GPU model, driver or CUDA library build is a different engine. Before research use, run the fidelity gate against the running container:

```bash
LLIKERT_TOKEN=$LLIKERT_API_TOKEN python tools/fidelity_gate_http.py --url http://127.0.0.1:8080 --json gate.json
```

The gate needs only the base Python client. It checks that prompt tokens match the reference and that max |Δp| ≤ 0.005 and max |Δlog q| ≤ 0.1 over 450 items. Keep `gate.json` with the engine fingerprint it reports.

## Security notes

- The container runs as an unprivileged user (uid 10001) and writes nothing to disk.
- Logs contain request ids, item counts, statuses and timings, never texts or tokens.
- Publish the port on `127.0.0.1` or behind a TLS-terminating proxy; the service does not terminate TLS itself.
- `LLIKERT_AUTH=platform` disables the service's own authentication. Use it only where the platform rejects unauthenticated requests before they reach the container.
