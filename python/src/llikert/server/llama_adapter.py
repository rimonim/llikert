"""Private llama.cpp adapter: one model, one context, sequential final-position logits.

Findings behind the rules here are in docs/decisions/0002-m0-findings.md:
- native asserts abort the process, so every precondition is checked in Python (F3);
- the logits buffer is reused by the next evaluation, so it is copied at once (F4);
- memory is cleared before every prompt; a failed evaluation clears it again, and a
  fatal one reinitializes the context or marks the adapter unhealthy (F5);
- n_batch is set equal to n_ubatch, the physical micro-batch that affects numerics (F8).
"""

from __future__ import annotations

import atexit
import ctypes
import hashlib
import importlib.metadata
import logging
import os
import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from llikert.server import native
from llikert.server.adapter import InferenceError, TokenKind

log = logging.getLogger("llikert.server.llama")
_LLAMA_LOG_LEVELS = {1: logging.DEBUG, 2: logging.DEBUG, 3: logging.INFO, 4: logging.WARNING, 5: logging.DEBUG}


class AdapterStartupError(RuntimeError):
    pass


@dataclass(frozen=True)
class LlamaConfig:
    model_path: pathlib.Path
    device: str = "cuda"  # "cuda" | "cpu"
    n_ctx: int = 4096
    batch_size: int = 512
    # defaults are the supported profile: with TF32 disabled, the only CUDA configuration that
    # passed the reference-fidelity gate (docs/decisions/0005-m3-performance-and-fidelity.md)
    flash_attn: str = "off"  # "auto" | "on" | "off"
    kv_type: str = "f32"  # "f16" | "f32"
    threads: int | None = None
    expected_sha256: str | None = None


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


class LlamaAdapter:
    def __init__(self, config: LlamaConfig, lib: native.Native | None = None):
        if config.device not in ("cuda", "cpu"):
            raise AdapterStartupError("device must be 'cuda' or 'cpu'")
        if config.flash_attn not in native.FLASH_ATTN_TYPES or config.kv_type not in native.GGML_TYPES:
            raise AdapterStartupError("invalid flash_attn or kv_type")
        if config.n_ctx < 16 or config.batch_size < 1:
            raise AdapterStartupError("n_ctx and batch_size must be positive")
        path = config.model_path
        if not path.is_file():
            raise AdapterStartupError("model file not found")

        self.config = config
        self.lib = lib or native.Native()
        L = self.lib.llama
        self._log_callback = native.ggml_log_callback(self._on_native_log)
        L.llama_log_set(self._log_callback, None)
        # native destructors may log during interpreter shutdown, after the ctypes callback
        # is gone; restore llama.cpp's default logger before that can happen
        atexit.register(self._restore_native_log)
        L.llama_backend_init()

        self.gguf_sha256 = sha256_file(path)
        if config.expected_sha256 and config.expected_sha256.lower() != self.gguf_sha256:
            raise AdapterStartupError("model file sha256 does not match --model-sha256")

        devices = self.lib.devices()
        gpus = [d for d in devices if d["type"] == "gpu"]
        if config.device == "cuda" and (not gpus or not L.llama_supports_gpu_offload()):
            raise AdapterStartupError("device 'cuda' was requested but no GPU backend is available")
        self._devices = gpus if config.device == "cuda" else [d for d in devices if d["type"] == "cpu"]

        mp = L.llama_model_default_params()
        mp.n_gpu_layers = -1 if config.device == "cuda" else 0
        self._model = L.llama_model_load_from_file(str(path).encode(), mp)
        if not self._model:
            raise AdapterStartupError("llama.cpp could not load the model file")
        self._vocab = L.llama_model_get_vocab(self._model)
        self._n_vocab = L.llama_vocab_n_tokens(self._vocab)
        self._math_libraries = native.Native.loaded_math_libraries()
        self._threads = config.threads or os.cpu_count() or 4
        self._ctx = None
        self._init_context()
        self._healthy = True
        self._specials: list[str] | None = None
        self.stats = {"prompts": 0, "decode_calls": 0, "tokens_evaluated": 0}
        self._warm_up()

    def _warm_up(self) -> None:
        """Evaluate one full-context prompt at startup, so memory problems surface before the
        service reports ready rather than in the middle of a dataset run."""
        filler = self.tokenize(" a", parse_special=False)
        try:
            self.final_logits(filler * self._eff_n_ctx)
        except (InferenceError, ValueError) as exc:
            self.close()
            raise AdapterStartupError(f"full-context warm-up failed: {exc}") from None
        finally:
            if self._ctx:
                self.lib.llama.llama_memory_clear(self.lib.llama.llama_get_memory(self._ctx), True)
        self.stats = {"prompts": 0, "decode_calls": 0, "tokens_evaluated": 0}

    # -- native lifecycle ---------------------------------------------------------------
    @staticmethod
    def _on_native_log(level: int, text: bytes | None, _user: Any) -> None:
        if text and log.isEnabledFor(_LLAMA_LOG_LEVELS.get(level, logging.DEBUG)):
            log.log(_LLAMA_LOG_LEVELS.get(level, logging.DEBUG), "%s", text.decode("utf-8", "replace").rstrip())

    def _init_context(self) -> None:
        L, c = self.lib.llama, self.config
        cp = L.llama_context_default_params()
        cp.n_ctx = c.n_ctx
        cp.n_batch = c.batch_size
        cp.n_ubatch = c.batch_size
        cp.n_seq_max = 1
        cp.n_threads = self._threads
        cp.n_threads_batch = self._threads
        cp.flash_attn_type = native.FLASH_ATTN_TYPES[c.flash_attn]
        cp.type_k = native.GGML_TYPES[c.kv_type]
        cp.type_v = native.GGML_TYPES[c.kv_type]
        cp.no_perf = True
        if c.device == "cpu":
            cp.offload_kqv = False
            cp.op_offload = False
        ctx = L.llama_init_from_model(self._model, cp)
        if not ctx:
            raise AdapterStartupError("llama.cpp could not create a context with these settings")
        self._ctx = ctx
        self._eff_n_ctx = L.llama_n_ctx(ctx)
        self._eff_n_batch = L.llama_n_batch(ctx)
        self._eff_n_ubatch = L.llama_n_ubatch(ctx)
        if self._eff_n_batch != self._eff_n_ubatch:
            raise AdapterStartupError("llama.cpp adjusted the batch size unequally; choose batch_size <= n_ctx")

    def _restore_native_log(self) -> None:
        self.lib.llama.llama_log_set(native.ggml_log_callback(), None)

    def close(self) -> None:
        L = self.lib.llama
        self._restore_native_log()
        if self._ctx:
            L.llama_free(self._ctx)
            self._ctx = None
        if getattr(self, "_model", None):
            L.llama_model_free(self._model)
            self._model = None
        self._healthy = False

    # -- vocabulary -----------------------------------------------------------------------
    @property
    def n_vocab(self) -> int:
        return self._n_vocab

    @property
    def n_ctx(self) -> int:
        return self._eff_n_ctx

    @property
    def healthy(self) -> bool:
        return self._healthy

    def _meta(self, key: str) -> str | None:
        buf = ctypes.create_string_buffer(1 << 20)
        n = self.lib.llama.llama_model_meta_val_str(self._model, key.encode(), buf, len(buf))
        return None if n < 0 else buf.value.decode("utf-8", "replace")

    def chat_template(self) -> str | None:
        raw = self.lib.llama.llama_model_chat_template(self._model, None)
        return raw.decode("utf-8") if raw else None

    def tokenize(self, text: str, *, parse_special: bool) -> list[int]:
        data = text.encode("utf-8")
        n_max = len(data) + 16
        while True:
            buf = (ctypes.c_int32 * n_max)()
            n = self.lib.llama.llama_tokenize(self._vocab, data, len(data), buf, n_max, False, parse_special)
            if n >= 0:
                return list(buf[:n])
            if -n <= n_max:
                raise RuntimeError("tokenization failed")
            n_max = -n

    def token_piece(self, token: int) -> bytes:
        self._check_token(token)
        size = 64
        while True:
            buf = ctypes.create_string_buffer(size)
            n = self.lib.llama.llama_token_to_piece(self._vocab, token, buf, size, 0, True)
            if n >= 0:
                return buf.raw[:n]
            size = -n

    def token_kind(self, token: int) -> TokenKind:
        self._check_token(token)
        L = self.lib.llama
        if L.llama_vocab_is_eog(self._vocab, token):
            return TokenKind.END_OF_GENERATION
        attr = L.llama_vocab_get_attr(self._vocab, token)
        A = native.TOKEN_ATTR
        for name, kind in (
            ("control", TokenKind.CONTROL),
            ("unknown", TokenKind.UNKNOWN),
            ("unused", TokenKind.UNUSED),
            ("byte", TokenKind.BYTE),
            ("user_defined", TokenKind.USER_DEFINED),
        ):
            if attr & A[name]:
                return kind
        return TokenKind.ORDINARY if attr & A["normal"] else TokenKind.UNKNOWN

    def special_token_texts(self) -> list[str]:
        if self._specials is None:
            texts = set()
            for t in range(self._n_vocab):
                if self.token_kind(t) in (TokenKind.CONTROL, TokenKind.END_OF_GENERATION):
                    piece = self.token_piece(t).decode("utf-8", "replace")
                    if piece:
                        texts.add(piece)
            self._specials = sorted(texts)
        return self._specials

    def _check_token(self, token: int) -> None:
        if not 0 <= token < self._n_vocab:
            raise ValueError("token id out of range")

    # -- inference --------------------------------------------------------------------------
    def final_logits(self, tokens: list[int]) -> np.ndarray:
        if not self._healthy or not self._ctx:
            raise InferenceError("adapter is not healthy", recovered=False)
        if not tokens:
            raise ValueError("empty prompt")
        if len(tokens) > self._eff_n_ctx:
            raise ValueError("prompt exceeds context")
        if min(tokens) < 0 or max(tokens) >= self._n_vocab:
            raise ValueError("token id out of range")

        L = self.lib.llama
        n_batch = self._eff_n_batch
        memory = L.llama_get_memory(self._ctx)
        L.llama_memory_clear(memory, True)
        self.stats["prompts"] += 1
        batch = L.llama_batch_init(n_batch, 0, 1)
        try:
            for start in range(0, len(tokens), n_batch):
                chunk = tokens[start : start + n_batch]
                batch.n_tokens = len(chunk)
                for i, tok in enumerate(chunk):
                    batch.token[i] = tok
                    batch.pos[i] = start + i
                    batch.n_seq_id[i] = 1
                    batch.seq_id[i][0] = 0
                    batch.logits[i] = 0
                if start + len(chunk) == len(tokens):
                    batch.logits[len(chunk) - 1] = 1
                rc = L.llama_decode(self._ctx, batch)
                self.stats["decode_calls"] += 1
                self.stats["tokens_evaluated"] += len(chunk)
                if rc != 0:
                    self._recover(rc)
            ptr = L.llama_get_logits_ith(self._ctx, -1)
            if not ptr:
                L.llama_memory_clear(memory, True)
                raise InferenceError("no logits at the final position", recovered=True)
            # copy immediately: the buffer is reused by the next evaluation
            return np.ctypeslib.as_array(ptr, shape=(self._n_vocab,)).astype(np.float64, copy=True)
        finally:
            L.llama_batch_free(batch)

    def _recover(self, rc: int) -> None:
        L = self.lib.llama
        if rc in (1, 2, -1):
            L.llama_memory_clear(L.llama_get_memory(self._ctx), True)
            raise InferenceError(f"llama_decode returned {rc}", recovered=True)
        log.error("fatal llama_decode return code %d; reinitializing context", rc)
        try:
            L.llama_free(self._ctx)
            self._ctx = None
            self._init_context()
        except Exception:  # noqa: BLE001
            self._healthy = False
            raise InferenceError(f"llama_decode returned {rc}; context reinitialization failed", recovered=False) from None
        raise InferenceError(f"llama_decode returned {rc}; context reinitialized", recovered=True)

    # -- identity ---------------------------------------------------------------------------
    def identity(self) -> dict[str, Any]:
        L = self.lib.llama
        try:
            binding_version = importlib.metadata.version("llama_cpp_python")
        except importlib.metadata.PackageNotFoundError:
            binding_version = None
        return {
            "backend": "llama.cpp",
            "llama_cpp_version": self.lib.llama_cpp_version,
            "ggml_version": self.lib.ggml_base.ggml_version().decode(),
            "ggml_commit": self.lib.ggml_base.ggml_commit().decode(),
            "llama_cpp_python_version": binding_version,
            "model": {
                "gguf_sha256": self.gguf_sha256,
                "architecture": self._meta("general.architecture"),
                "file_type": self._meta("general.file_type"),
                "n_params": int(L.llama_model_n_params(self._model)),
                "n_vocab": self._n_vocab,
                "add_bos": bool(L.llama_vocab_get_add_bos(self._vocab)),
            },
            "device": {"type": self.config.device, "devices": [d["description"] for d in self._devices]},
            "cuda_tf32": "disabled" if self.config.device == "cuda" else None,
            "math_libraries": self._math_libraries,
            "n_gpu_layers": -1 if self.config.device == "cuda" else 0,
            "n_ctx": self._eff_n_ctx,
            "n_ubatch": self._eff_n_ubatch,
            "flash_attn": self.config.flash_attn,
            "kv_type": self.config.kv_type,
        }

    def execution(self) -> dict[str, Any]:
        return {
            "backend": "llama.cpp",
            "model_file": self.config.model_path.name,
            "model_name": self._meta("general.name"),
            "n_threads": self._threads,
            "system_info": self.lib.llama.llama_print_system_info().decode("utf-8", "replace"),
        }
