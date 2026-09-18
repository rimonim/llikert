# Troubleshooting

Written for: researchers using the R or Python client, and operators running the service.

## Client (R and Python)

| Symptom | Cause | What to do |
|---|---|---|
| `llikert_error_not_ready` / `ServiceUnavailable` when connecting | The service is still starting (loading or warming up the model, or a hosted endpoint scaling up), or is down | Increase `wait`; for hosted endpoints also set `scale_up_timeout`. Ask the operator whether `/health` ever returns 200 |
| `llikert_error_authentication` / `AuthenticationError` (401 or 403) | Missing or wrong token | Set `LLIKERT_TOKEN` (or pass `token`). For Hugging Face endpoints, use a Hugging Face token with access to the endpoint |
| `llikert_error_invalid_response_codes` / `PrepareError` | A response code is not a single distinct token for this model (for example `"10"`, or a word that splits) | Look at `problems` and `suggestions`. Choose new codes yourself, and remember the prompt changes with them |
| `422 invalid_task` naming a field the service "does not know" | The service is older than the R or Python package (for example it predates `answer_instruction` as a task field) | Ask whoever runs the service to update it: `git pull`, rebuild the image (step 3 of `deploy/README.md`), then `deploy/service.sh recreate` |
| `422 invalid_task` naming another field | The task itself is invalid | The message lists each field and what is wrong with it |
| `llikert_error_engine_fingerprint_mismatch` / `FingerprintMismatch` (409) | The service now runs a different model, library build or execution setting than when the task was prepared | Prepare the task again. For a checkpointed run, start a new checkpoint; never mix engines in one run |
| `prepared_task_mismatch` (409) | The prepared task file was edited, or it came from another service version | Prepare the task again from the task definition |
| `Checkpoint ... already exists` | You reused a checkpoint path | Pass `resume = TRUE` / `resume=True`, or choose a new path |
| `The checkpoint belongs to a different run (changed: ...)` | The texts, their order, the ids, the task or the engine changed since the checkpoint was written | Start a new checkpoint. The message names what changed |
| `The checkpoint is locked by another writer` | Another session is using the checkpoint, or a crashed session left a lock | Make sure no other session is running, then pass `force_unlock` |
| Status `context_limit` for some items | The rendered prompt is longer than the service's context (`n_prompt_tokens` shows by how much) | Shorten or split those texts under a documented rule, or ask for a service with a larger `n_ctx`, which is a different engine |
| Status `inference_error` | Native evaluation failed for that item | Resume the checkpoint; only these items are retried. If it persists, tell the operator |
| `HTTP 502/504 without a protocol error body` | A proxy in front of the service timed out or failed | Retries happen automatically. If they persist, use a smaller `chunk_size` |
| Many `429` retries | Several clients share one service | Retries and waits happen automatically; run fewer clients in parallel |
| Low `coverage` | The model's next token is often not one of your codes | Not an error. Coverage is a response-format diagnostic: review the instructions and codes with `preview_prompt()` |

## Service

| Symptom | Cause | What to do |
|---|---|---|
| `llama.cpp ... is installed; llikert requires the pinned 0.4.1 build` | A PyPI llama-cpp-python wheel, or a different build | Build and install the pinned wheel (`docs/local-service.md`) |
| `GLIBCXX_3.4.30 not found` | The service runs from a conda-based Python | Use a distribution or python.org Python for the service venv |
| `selfcheck` shows `math_libraries` with `libcublas.so.11` | The wheel linked a distribution CUDA runtime | Rebuild with `CUDA_HOME` set to the intended toolkit |
| `device 'cuda' was requested but no GPU backend is available` | No GPU visible, driver problem, or a CPU-only wheel | `nvidia-smi` must work. In containers, use `--gpus all` (restart the Docker daemon after installing the NVIDIA Container Toolkit) |
| Container: `system has unsupported display driver / cuda driver combination` | The host driver is older than the image's CUDA version needs | Upgrade the driver, or use an image built on a CUDA base the driver supports (for example 12.9.0 for driver 575.51) |
| Container: `libgomp.so.1: cannot open shared object file` | A custom runtime image lacks OpenMP | Install `libgomp1` (the provided Dockerfile does) |
| `llama.cpp could not create a context with these settings` | GPU memory is too small for `--n-ctx`, or other processes use the GPU | Lower `--n-ctx`, or free the GPU. The service fails at startup instead of mid-run |
| `chat template sha256 ... is not a supported template profile` | The model is not an allowlisted profile | Use a supported model file (`docs/compatibility.md`). Adding a profile requires rendering verification and the fidelity gate |
| `model file sha256 does not match --model-sha256` | Wrong, corrupt or re-converted model file | Re-download or re-convert, then verify the hash |
| `deploy/prepare-model.sh` reports `MISMATCH` | The download was incomplete, or the conversion ran with different inputs (the folder name and extra files such as a model card end up in the file's metadata) | Delete the output file and rerun the script without changes. If the mismatch persists, report the script's output and don't use the file |
| `--auth none is only allowed on a loopback address` | Tried to expose an unauthenticated service | Use `--auth token` with `LLIKERT_API_TOKEN`, or bind to 127.0.0.1 |
| The process aborts with `CUDA error: out of memory` during scoring | Another process took GPU memory after startup | Give the service exclusive use of the GPU. Restart it; completed chunks are safe in client checkpoints |
| Another program cannot get GPU memory, or a model build fails with out of memory | The service holds its graphics memory whenever it runs, even when idle | Stop it with `deploy/service.sh stop`, and start it again afterwards with `deploy/service.sh start` |
| Results differ from a colleague's for the same task | Different engine fingerprint: model file, GPU, library build or settings | Compare `llikert_result_manifest()` / `result.manifest`. Only identical fingerprints are expected to agree bitwise |
