# Running the scoring service locally

To run the service in a container instead of installing it on the host, see `deploy/README.md`. For a Hugging Face Inference Endpoint, see `deploy/hf-endpoint.md` (experimental).

Written for: operators setting up a LLikert service on a Linux machine with an NVIDIA GPU. Researchers who only connect to a running service need nothing on this page.

Before you start, know the unavoidable local costs: the model file (16 GB for the supported F32 profile), a GPU with at least 24 GB of memory, and a native library build that takes a few minutes.

## 1. Code and Python environment

```bash
git clone https://github.com/rimonim/llikert.git
cd llikert
```


Use a distribution or python.org CPython 3.10 or newer, **not a conda Python**. Conda's bundled `libstdc++` prevents the CUDA backend from loading.

```bash
/usr/bin/python3 -m venv .venv
.venv/bin/pip install -e "python[server]"
```

## 2. Pinned llama.cpp build

The service requires llama-cpp-python 0.3.35 built against llama.cpp v0.4.1. A llama-cpp-python wheel from PyPI is rejected at startup.

```bash
CUDA_HOME=/usr/local/cuda-12.9 PYTHON=.venv/bin/python tools/build-llama-cpp-python.sh cuda
.venv/bin/pip install --no-deps build/wheels/cuda/llama_cpp_python-0.3.35-*.whl
```

Set `CUDA_HOME` to the toolkit you mean to use. The script pins CMake to it, because otherwise a distribution CUDA runtime can be linked silently. Check the result with `llikert selfcheck`: `math_libraries` should list `libcublas.so.12.9…`. For other GPU generations, set `CUDA_ARCHITECTURES`; the default of 89 is Ada.

## 3. Model file

The simplest way is `deploy/prepare-model.sh models`: it runs the download and conversion in a container and verifies the checksum. The manual equivalent follows.

No official F32 GGUF exists, so convert the official weights with the converter from the same llama.cpp tag. The conversion is deterministic, but the name of the weights folder and any extra files in it (for example a model card) become metadata in the file. Use a folder named `Qwen3-4B-Instruct-2507` containing only the files below.

```bash
# Qwen/Qwen3-4B-Instruct-2507 at revision cdbee75f17c01a7cc42f958dc650907174af0554
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --local-dir models/hf/Qwen3-4B-Instruct-2507
pip install torch --index-url https://download.pytorch.org/whl/cpu   # conversion only; any venv
pip install -e build/llama-cpp-python/vendor/llama.cpp/gguf-py
python build/llama-cpp-python/vendor/llama.cpp/convert_hf_to_gguf.py models/hf/Qwen3-4B-Instruct-2507 \
  --outtype f32 --outfile models/qwen3-4b-instruct-2507-f32.gguf
sha256sum models/qwen3-4b-instruct-2507-f32.gguf
# expected: a5733d5a25e8b824e1ef9d7e75449ffd8f8ec582de8ac253420e4423a7f0e357
```

## 4. Check, then serve

```bash
.venv/bin/llikert selfcheck --model models/qwen3-4b-instruct-2507-f32.gguf

.venv/bin/llikert serve --model models/qwen3-4b-instruct-2507-f32.gguf \
  --model-sha256 a5733d5a25e8b824e1ef9d7e75449ffd8f8ec582de8ac253420e4423a7f0e357
```

`selfcheck` reports the native library, the template profile, the smoke check and the engine fingerprint.

`serve` binds to `127.0.0.1:8080` by default. `/health` returns 503 while the model file is hashed, loaded and warmed up with one full-context evaluation (about 22 s for the F32 file), then 200. A context that does not fit in GPU memory fails at startup, not during a run. Stop the service with Ctrl-C.

The defaults (`--device cuda --batch-size 512 --flash-attn off --kv-type f32 --n-ctx 4096`) are the supported profile. The service also always disables TF32 in cuBLAS (`NVIDIA_TF32_OVERRIDE=0`), because TF32 moves probabilities by up to 0.03 (decision 0005). Changing any execution setting changes the engine fingerprint, and a profile you change is not covered by the fidelity evidence in `docs/compatibility.md`.

## Serving beyond this machine

- **Loopback only:** `--auth none` refuses to start on any other address.
- **Other machines:** set a random token of at least 16 characters, then run with `--auth token`:

  ```bash
  export LLIKERT_API_TOKEN=$(python3 -c "import secrets; print(secrets.token_hex(32))")
  .venv/bin/llikert serve --model ... --host 0.0.0.0 --auth token
  ```

  Put the service behind a TLS-terminating reverse proxy, because it doesn't terminate TLS itself.
- **Hosting platforms:** `--auth platform` is only for platforms that authenticate every request before it reaches the container.
- **Sensitive data:** logs contain request ids, item counts, statuses and timings, never texts, prompts or tokens. The service does not store results.
