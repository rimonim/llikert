# Hugging Face Inference Endpoint recipe (EXPERIMENTAL — not executed)

Written for: the lab operator who will deploy one LLikert endpoint for collaborators.

**Status:** nothing in this recipe has been executed against Hugging Face. It is based on the Hugging Face custom-container documentation (read 2026-09-16) and on local container tests. Treat every step as unverified until the checklist at the end has been completed on a real endpoint. Creating an endpoint is billable.

## What you need

- **Owner authorization** to publish a container image, upload model weights to a Hugging Face repository, and create a billable endpoint.
- **A container registry** Hugging Face can pull from (Docker Hub, GHCR, ECR, ACR or GCR).
- **A GPU with at least 24 GB**, such as an NVIDIA L4 or A10G. The F32 profile uses about 17.3 GB. Neither of those GPUs has passed the fidelity gate yet; only an RTX 4090 has.
- **Data-handling approval.** Study texts are sent to a third-party environment. Institutional approvals and data-handling rules are your responsibility; using a particular provider does not grant them.

## 1. Publish the image

```bash
git clone https://github.com/rimonim/llikert.git && cd llikert
docker build -f deploy/Dockerfile.cuda -t llikert-service:0.1.0.dev0-cuda12.9 --build-arg SOURCE_REVISION=$(git rev-parse HEAD) .
docker tag llikert-service:0.1.0.dev0-cuda12.9 <registry>/<namespace>/llikert-service:0.1.0.dev0-cuda12.9
docker push <registry>/<namespace>/llikert-service:0.1.0.dev0-cuda12.9
docker inspect --format '{{index .RepoDigests 0}}' <registry>/<namespace>/llikert-service:0.1.0.dev0-cuda12.9
```

Deploy by digest (`…@sha256:…`), not by tag, so the endpoint cannot change underneath a study.

## 2. Put the model file in a model repository

Create a **private** model repository and upload `qwen3-4b-instruct-2507-f32.gguf`, prepared with `deploy/prepare-model.sh` (SHA-256 `a5733d5a…e357`). The source model is Apache-2.0, which permits redistribution; keep its license and model card in the repository.

Hugging Face mounts the selected repository at `/repository`. Put only this one GGUF there (plus license and README), and name the file explicitly in `LLIKERT_MODEL`.

## 3. Create the endpoint

| Setting | Value |
|---|---|
| Model repository | the private repository from step 2 |
| Hardware | a 24 GB GPU (L4 or A10G), 1 replica |
| Security | **Protected** (requires a Hugging Face token); never Public |
| Container type | Custom container |
| Image | `<registry>/<namespace>/llikert-service@sha256:<digest>` |
| Container port | `8080` |
| Health route | `/health` |
| Environment | `LLIKERT_MODEL=/repository/qwen3-4b-instruct-2507-f32.gguf`, `LLIKERT_MODEL_SHA256=a5733d5a25e8b824e1ef9d7e75449ffd8f8ec582de8ac253420e4423a7f0e357`, `LLIKERT_AUTH=platform` |
| Autoscaling | min 0, max 1 replica; scale to zero after an idle period you choose |

`LLIKERT_AUTH=platform` relies on the protected endpoint to reject unauthenticated requests. If you cannot confirm that it does (checklist item 3), use `LLIKERT_AUTH=token` with a `LLIKERT_API_TOKEN` secret instead. Collaborators then need that token, and must check whether the platform token and the service token can be sent together.

Keep `max replicas = 1` until the gate has run: every replica must report the same engine fingerprint.

## 4. Connect

Collaborators set the endpoint URL and their Hugging Face token:

```r
engine <- scorer_connect(url = "https://<endpoint>.endpoints.huggingface.cloud",
                         token = Sys.getenv("HF_TOKEN"), wait = 900, scale_up_timeout = 600)
```

```python
scorer = Scorer.connect("https://<endpoint>.endpoints.huggingface.cloud", token=os.environ["HF_TOKEN"],
                        wait=900, scale_up_timeout=600)
```

- **Cold start:** while the endpoint scales up from zero, the proxy returns 503, and the clients wait up to `wait` seconds. Hashing, loading and warming up a 16 GB model takes about 25 s locally, plus the platform's own scale-up time.
- **`scale_up_timeout`:** sends `X-Scale-Up-Timeout` so the proxy holds requests while scaling up.

## 5. Verify before research use

```bash
LLIKERT_TOKEN=$HF_TOKEN python tools/fidelity_gate_http.py \
  --url https://<endpoint>.endpoints.huggingface.cloud --json hf-gate.json
```

If the gate fails, do not use the endpoint for research. Record the result, the reported engine fingerprint and the GPU in `docs/compatibility.md`.

## 6. Control cost

- Pause the endpoint, or let it scale to zero, when a dataset run is finished. Delete it when the study phase ends.
- Check charges on the Hugging Face billing page for your organization. This guide does not quote prices, because they change.
- A timeout after a request is sent is ambiguous: the service may have finished the work. The clients retry with bounded backoff, which can repeat computation and cost. Keep checkpoints on, so completed chunks are never rescored.

## Verification checklist (all NOT executed)

| # | Check | How |
|---|---|---|
| 1 | The image is pulled by digest and the container starts | Endpoint status reaches Running |
| 2 | `/health` readiness gates traffic | Status stays Initializing until the model has loaded |
| 3 | A protected endpoint rejects requests without a token before the container | `curl` without `Authorization` returns 401 or 403 from the platform |
| 4 | The `Authorization` header and request paths reach the container unchanged (`/v1/info`, `/v1/prepare`, `/v1/score`) | `scorer_connect()` succeeds; an invalid-task request returns the service's 422 envelope |
| 5 | Request duration and body limits allow the default chunks | Score 64 long texts in one request |
| 6 | Cold start from zero works within the client `wait` | Scale to zero, then connect |
| 7 | Reference-fidelity gate passes on the endpoint GPU | `tools/fidelity_gate_http.py` |
| 8 | The engine fingerprint is stable across restarts | Compare `/v1/info` before and after a restart |
| 9 | Logs contain no texts | Inspect endpoint logs after scoring |
