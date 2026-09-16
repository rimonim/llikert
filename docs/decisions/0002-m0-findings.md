# 0002: M0 feasibility findings

**Status:** accepted 2026-09-16 (F8 resolved by D23/D24)
**Evidence:** `spikes/` (scripts) and `spikes/out/` (JSON outputs). The larger model files stay local and are not committed.

## Pinned stack

| Item | Value |
|---|---|
| llama.cpp | `v0.4.1`, tag commit `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4` (released 2026-09-14) |
| llama-cpp-python | `v0.3.35`, source build with `vendor/llama.cpp` moved to the tag above |
| CUDA build | CUDA 12.9, `-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89`; RTX 4090, driver 575.51.03 |
| CPU build | `-DGGML_CUDA=off -DGGML_NATIVE=on` (not portable; release images need a fixed ISA baseline) |
| Dev Python | system CPython 3.10.12 (see F2) |
| Small model | `Qwen/Qwen2.5-0.5B-Instruct-GGUF` @ `9217f5db`, `qwen2.5-0.5b-instruct-q8_0.gguf`, sha256 `ca59ca7f…844e` |
| First model (D23) | `qwen3-4b-instruct-2507-f32.gguf`, converted from the official safetensors (see "Model acquisition"), sha256 `a5733d5a…e357` |
| Q8_0 compared in M0 | `ggml-org/Qwen3-4B-Instruct-2507-Q8_0-GGUF` @ `e6f794d4`, sha256 `ae916ede…d5f1`. Qwen publishes no official GGUF for 2507. |
| Reference weights | `Qwen/Qwen3-4B-Instruct-2507` @ `cdbee75f` (safetensors and tokenizer); Apache-2.0 |

Name check (2026-09-16): `llikert` is unused on PyPI, CRAN (no current package or archive), r-universe and GitHub.

## Findings

**F1: The binding's struct layout is wrong for the pinned llama.cpp.**
- v0.4.1 added `lazy_mode` to `llama_model_params`. llama-cpp-python 0.3.35 declares a 72-byte struct where C has 80 bytes, and 13 fields sit at the wrong offsets. `n_gpu_layers` is unaffected, but `main_gpu`, `tensor_split`, `vocab_only` and the fields after them would be read incorrectly, with no error raised.
- `llama_context_params` and `llama_batch` still match.
- **Decision:** the adapter declares its own argtypes, restypes and structs for the 32 functions it uses (`spikes/native.py`). The binding is used only to build, package and locate `libllama`. Importing `llama_cpp` itself would also pull in `diskcache` and more.
- An ABI test compiles `spikes/abi/abi_offsets.c` against the pinned header and compares `sizeof`/`offsetof` with the ctypes layout (`spikes/abi_check.py`). Current result: 0 mismatches. This becomes a CI test in M1.

**F2: conda's libstdc++ breaks the CUDA backend.**
- In a venv based on miniconda Python, importing `numpy` loads conda's older `libstdc++`. `libggml-cuda.so` then fails with `GLIBCXX_3.4.30 not found`.
- Development uses a system-Python venv. The docs will state that the server must not run from a conda-based Python unless libstdc++ is compatible. Release images use a distribution Python.

**F3: Native asserts abort the whole process.**
- llama.cpp silently clamps `n_batch` to `n_ctx`. Submitting a batch larger than the effective `n_batch` triggers `GGML_ASSERT`, which aborts the process instead of returning an error code.
- The adapter must read back the effective `llama_n_ctx`, `llama_n_batch` and `llama_n_ubatch`, and check every precondition (batch size, positions, token ids in range, context length) in Python before any native call.
- Also: the context default is only 4 threads, so the adapter sets `n_threads` and `n_threads_batch` explicitly. They go in the manifest, not the fingerprint, since thread count had no numerical effect in these probes.

**F4: Output buffer ownership.**
- The pointer from `llama_get_logits_ith(ctx, -1)` holds the *next* evaluation's values after another decode.
- Final-position logits must be copied to float64 immediately. The float32 → float64 copy costs about 1 ms for the 151,936-entry vocabulary.

**F5: State, context limit and recovery.**
- `llama_memory_clear(mem, true)` before each item gives bit-identical logits: on repeat, and after a different prompt has been scored.
- With `n_ctx = 256`, prompts of 255 and 256 tokens decode fine, and 257 returns `rc = 1` from `llama_decode`. The next prompt after that failure is bit-identical to a fresh run.
- The context limit rule is `n_prompt_tokens <= n_ctx`. No tokens need to be reserved, since nothing is generated.

**F6: Rendering and tokenization work as designed (D1–D3).**
- The GGUF chat template for Qwen3-4B-Instruct-2507 is byte-identical to the official `tokenizer_config.json` (sha256 `64f85b19…c326`).
- For Qwen2.5-0.5B the template strings differ, but the rendered output is identical for our messages.
- The sandboxed jinja2 renderer (transformers settings) matches `apply_chat_template` exactly for all 10 test texts on both models, including leading space, trailing newline, Unicode and whitespace-only text.
- Segment tokenization matches HF tokenization token for token, except for `marker_injection`, where it differs as intended: HF parses the injected `<|im_end|>` as a control token, while ours keeps it ordinary text and preserves the tail.
- `add_bos = false` and `add_eos = false` for both Qwen models.
- The tail segment is `<|im_end|>\n<|im_start|>assistant\n`: tokens 151645 (control), 198, 151644 (control), 77091, 198.

**F7: Response codes on the Qwen tokenizer.**
- Single tokens at the answer boundary: `A`, `B`, `C`, ` A`, `1`, `5`, `Yes`, `description`, `question`, `request`.
- Not single tokens: `10` and `100`, because digits are split, and `very negative`.
- Numeric scales with more than 9 points therefore need letter codes or other single-token codes.

**F8: Execution settings and weight format change probabilities materially.**
- The reference is transformers with eager attention on CPU, on the official weights. float64 and float32 agree to 4 decimals.
- **Kernels are correct.**
  - F32 GGUF on CPU with flash attention off and an f32 KV cache matches the reference to max |Δp| ≤ 2e-5 and max |Δlog q| ≤ 2e-4, for every `n_ubatch` tried (1, 64, 512).
  - CUDA with `n_ubatch = 1` matches too.
- **Execution settings matter.** Even with F32 weights on CUDA, `n_ubatch` and flash attention change `p` by up to 0.03. With F16 weights on CUDA the change reaches 0.23 (`n_ubatch = 64`). The logical `n_batch` has no effect, and neither do the `GGML_CUDA_FORCE_*` environment variables.
- **Q8_0 changes the instrument.** Q8_0 deviates by up to 0.47 in `p` on near-tie items, and all Q8_0 configurations disagree with each other. The ggml-org file and our own Q8_0 conversion give identical numbers, so this is quantization, not a bad file.
- **Fidelity sweep** over 28 items, 20 of them deliberately ambiguous, with warm latency (`spikes/out/fidelity-summary.jsonl`):

  | Weights, device, settings | ms/item | max \|Δp\| | p90 \|Δp\| | max \|Δlog q\|* | items \|Δp\|>0.05 |
  |---|---|---|---|---|---|
  | Q8_0, CUDA, defaults (u512, FA auto, KV f16) | 13 | 0.128 | 0.003 | 2.22 | 3/28 |
  | Q8_0, CPU, u512, FA off, KV f32 | 1162† | 0.097 | 0.029 | 2.74 | 3/28 |
  | F16, CUDA, defaults | 17 | 0.014 | 0.001 | 0.81 | 0/28 |
  | F16, CUDA, u1, FA off, KV f32 | 622 | 0.003 | 0.001 | 0.11 | 0/28 |
  | F16, CPU, u512, FA off, KV f32 | 2456† | 0.003 | 0.000 | 0.04 | 0/28 |
  | **F32, CUDA, defaults** | **27** | **0.0008** | 0.0001 | 0.06 | 0/28 |
  | F32, CUDA, u1, FA off, KV f32 | 1154 | 0.0000 | 0.0000 | 0.0002 | 0/28 |
  | F32, CPU, u512, FA off, KV f32 | 2570† | 0.0000 | 0.0000 | 0.0002 | 0/28 |

  \*Only candidates with reference log q > −15 are counted. †CPU latency used the 4-thread default (F3), so it is pessimistic.

- **Consequences, whatever is decided:**
  1. The engine fingerprint must include weight format, device, `n_ubatch`, flash-attention mode and KV-cache types (already in D6). Prompt chunking must be deterministic, starting at position 0.
  2. Every supported model profile needs a **reference-fidelity gate**: transformers float32 on the official weights, fixed prompts, and a stated tolerance, recorded in the compatibility matrix.
  3. M3 optimizations (multi-sequence batching, shared-prefix reuse) change micro-batch composition. Each must pass the same gate against the reference, not just match the sequential path.
  4. The M3 tolerances in the implementation plan (|Δp| ≤ 1e-4) can't be met across batch configurations on CUDA. They must be set per execution profile from measurements.
  5. The docs and the methods paragraph must state the weight format and execution profile as part of the instrument.

## Decisions (owner, 2026-09-16)

- **D23 (supersedes the weight format in C3):**
  - The first supported profile is Qwen3-4B-Instruct-2507 as an **F32 GGUF on CUDA with default kernels**: `n_ubatch` 512, flash attention auto, f16 KV. Measured max |Δp| is 0.0008 at 27 ms per item.
  - Q8_0 is not a supported profile in v0.1.
  - Qwen2.5-0.5B stays the smoke and CI model and makes no fidelity claim.
- **D24: reference-fidelity gate.**
  - A model profile, meaning weights plus device plus execution settings, is listed as supported only after passing a check against transformers float32 on the official weights.
  - The check uses a fixed prompt set with tolerances stated in advance, and results go in the compatibility matrix.
  - M3 optimizations must pass the same gate.
  - Initial tolerances for the F32/CUDA profile are max |Δp| ≤ 0.005 and max |Δlog q| ≤ 0.1 for candidates with reference log q > −15. They are set with headroom over the observed 0.0008 and 0.06, and will be revisited with the M3 benchmark corpus.

### Model acquisition for the F32 profile

There is no official F32 GGUF, so the documented step is a deterministic conversion with the pinned llama.cpp converter:

```text
source   Qwen/Qwen3-4B-Instruct-2507 @ cdbee75f17c01a7cc42f958dc650907174af0554
         model-00001-of-00003.safetensors 75311d91bb08cf0b882913da464a1e722a31fb44db35208663487efb7a3d8ed6
         model-00002-of-00003.safetensors 0b48adbb1f60e901153d91907ba11ce63bd4b8b584482e730f48808d055dfba1
         model-00003-of-00003.safetensors 7dd39ccca5e4de123c74c14af44c9bf2eb75df33b4614382af0134528e060d5d
command  python vendor/llama.cpp/convert_hf_to_gguf.py <dir> --outtype f32   (llama.cpp v0.4.1, gguf-py from same tree)
result   qwen3-4b-instruct-2507-f32.gguf  sha256 a5733d5a25e8b824e1ef9d7e75449ffd8f8ec582de8ac253420e4423a7f0e357
```

The conversion is byte-reproducible: a second run produced the same SHA-256. Converting remains an operator step; the service never converts models itself.

- **Open for M4:**
  - Hosting the converted file in a lab-owned HF model repo, so `/repository` mounts it, is an external publishing action and needs owner authorization.
  - The Apache-2.0 license permits redistribution.

## Hugging Face custom container (docs read 2026-09-16; nothing deployed)

- **Confirmed in the docs:**
  - The model repository is mounted at `/repository`, and weights should not be baked into the image.
  - The container port is set in the endpoint UI.
  - A readiness probe polls `/health` about once per second.
  - The proxy returns `503` while scaling from zero. A client may send `X-Scale-Up-Timeout: <seconds>` to have the proxy hold the request, which should be added to the client connect policy (D15).
  - The image must be linux/amd64 in an accessible registry. The documented client sends `Authorization: Bearer <HF token>`.
- **Not in the docs (verify in M4):** whether the `Authorization` header is forwarded to the container, request duration and body limits, and path rewriting.
- **Local container probe: NOT EXECUTED.** The user account has no access to the Docker daemon. `spikes/container/` holds a CUDA runtime Dockerfile and a `/health` stub. The stub itself was run outside Docker: `/health` returned 503 during load, then 200, with GPU offload available.
