"""ctypes declarations for the libllama functions llikert calls (decision D22, finding F1).

llama-cpp-python 0.3.35 declares struct layouts for its own vendored llama.cpp commit.
llikert pins llama.cpp v0.4.1, where ``llama_model_params`` gained ``lazy_mode``, so the
binding's declaration is shifted from ``main_gpu`` onward. This module declares only
what the adapter calls, against ``include/llama.h`` at the pinned tag; the binding is
used only to build and locate the shared libraries. Struct offsets are checked against
the C header by ``tests/test_native_abi.py``.
"""

from __future__ import annotations

import ctypes
import pathlib

c_bool, c_char_p, c_float, c_int8, c_int32, c_uint32, c_void_p, c_size_t = (
    ctypes.c_bool,
    ctypes.c_char_p,
    ctypes.c_float,
    ctypes.c_int8,
    ctypes.c_int32,
    ctypes.c_uint32,
    ctypes.c_void_p,
    ctypes.c_size_t,
)
c_enum = ctypes.c_int
llama_token = c_int32
llama_pos = c_int32
llama_seq_id = c_int32

llama_progress_callback = ctypes.CFUNCTYPE(c_bool, c_float, c_void_p)
ggml_backend_sched_eval_callback = ctypes.CFUNCTYPE(c_bool, c_void_p, c_bool, c_void_p)
ggml_abort_callback = ctypes.CFUNCTYPE(c_bool, c_void_p)


class llama_model_params(ctypes.Structure):
    _fields_ = [
        ("devices", c_void_p),
        ("tensor_buft_overrides", c_void_p),
        ("n_gpu_layers", c_int32),
        ("split_mode", c_enum),
        ("load_mode", c_enum),
        ("lazy_mode", c_enum),
        ("main_gpu", c_int32),
        ("tensor_split", ctypes.POINTER(c_float)),
        ("progress_callback", llama_progress_callback),
        ("progress_callback_user_data", c_void_p),
        ("kv_overrides", c_void_p),
        ("vocab_only", c_bool),
        ("check_tensors", c_bool),
        ("use_extra_bufts", c_bool),
        ("no_host", c_bool),
        ("no_alloc", c_bool),
        ("load_mtp", c_bool),
    ]


class llama_context_params(ctypes.Structure):
    _fields_ = [
        ("n_ctx", c_uint32),
        ("n_batch", c_uint32),
        ("n_ubatch", c_uint32),
        ("n_seq_max", c_uint32),
        ("n_rs_seq", c_uint32),
        ("n_outputs_max", c_uint32),
        ("n_outputs_max_per_seq", c_uint32),
        ("n_threads", c_int32),
        ("n_threads_batch", c_int32),
        ("ctx_type", c_enum),
        ("rope_scaling_type", c_enum),
        ("pooling_type", c_enum),
        ("attention_type", c_enum),
        ("flash_attn_type", c_enum),
        ("rope_freq_base", c_float),
        ("rope_freq_scale", c_float),
        ("yarn_ext_factor", c_float),
        ("yarn_attn_factor", c_float),
        ("yarn_beta_fast", c_float),
        ("yarn_beta_slow", c_float),
        ("yarn_orig_ctx", c_uint32),
        ("defrag_thold", c_float),
        ("cb_eval", ggml_backend_sched_eval_callback),
        ("cb_eval_user_data", c_void_p),
        ("type_k", c_enum),
        ("type_v", c_enum),
        ("abort_callback", ggml_abort_callback),
        ("abort_callback_data", c_void_p),
        ("embeddings", c_bool),
        ("offload_kqv", c_bool),
        ("no_perf", c_bool),
        ("op_offload", c_bool),
        ("swa_full", c_bool),
        ("kv_unified", c_bool),
        ("samplers", c_void_p),
        ("n_samplers", c_size_t),
        ("ctx_other", c_void_p),
    ]


class llama_batch(ctypes.Structure):
    _fields_ = [
        ("n_tokens", c_int32),
        ("token", ctypes.POINTER(llama_token)),
        ("embd", ctypes.POINTER(c_float)),
        ("pos", ctypes.POINTER(llama_pos)),
        ("n_seq_id", ctypes.POINTER(c_int32)),
        ("seq_id", ctypes.POINTER(ctypes.POINTER(llama_seq_id))),
        ("logits", ctypes.POINTER(c_int8)),
    ]


PINNED_LLAMA_CPP_VERSION = "0.4.1"

ggml_log_callback = ctypes.CFUNCTYPE(None, c_enum, c_char_p, c_void_p)

# name -> (argtypes, restype); prototypes copied from llama.h at v0.4.1
PROTOTYPES = {
    "llama_log_set": ([ggml_log_callback, c_void_p], None),
    "llama_model_n_params": ([c_void_p], ctypes.c_uint64),
    "llama_backend_init": ([], None),
    "llama_backend_free": ([], None),
    "llama_print_system_info": ([], c_char_p),
    "llama_supports_gpu_offload": ([], c_bool),
    "llama_model_default_params": ([], llama_model_params),
    "llama_context_default_params": ([], llama_context_params),
    "llama_model_load_from_file": ([c_char_p, llama_model_params], c_void_p),
    "llama_model_free": ([c_void_p], None),
    "llama_init_from_model": ([c_void_p, llama_context_params], c_void_p),
    "llama_free": ([c_void_p], None),
    "llama_model_get_vocab": ([c_void_p], c_void_p),
    "llama_model_chat_template": ([c_void_p, c_char_p], c_char_p),
    "llama_model_meta_val_str": ([c_void_p, c_char_p, c_char_p, c_size_t], c_int32),
    "llama_model_n_ctx_train": ([c_void_p], c_int32),
    "llama_model_desc": ([c_void_p, c_char_p, c_size_t], c_int32),
    "llama_n_ctx": ([c_void_p], c_uint32),
    "llama_n_batch": ([c_void_p], c_uint32),
    "llama_n_ubatch": ([c_void_p], c_uint32),
    "llama_vocab_n_tokens": ([c_void_p], c_int32),
    "llama_vocab_get_add_bos": ([c_void_p], c_bool),
    "llama_vocab_get_add_eos": ([c_void_p], c_bool),
    "llama_vocab_is_eog": ([c_void_p, llama_token], c_bool),
    "llama_vocab_is_control": ([c_void_p, llama_token], c_bool),
    "llama_vocab_get_attr": ([c_void_p, llama_token], c_enum),
    "llama_tokenize": (
        [c_void_p, c_char_p, c_int32, ctypes.POINTER(llama_token), c_int32, c_bool, c_bool],
        c_int32,
    ),
    "llama_token_to_piece": ([c_void_p, llama_token, c_char_p, c_int32, c_int32, c_bool], c_int32),
    "llama_get_memory": ([c_void_p], c_void_p),
    "llama_memory_clear": ([c_void_p, c_bool], None),
    "llama_batch_init": ([c_int32, c_int32, c_int32], llama_batch),
    "llama_batch_free": ([llama_batch], None),
    "llama_decode": ([c_void_p, llama_batch], c_int32),
    "llama_get_logits_ith": ([c_void_p, c_int32], ctypes.POINTER(c_float)),
}


# ggml-backend.h / ggml.h (libggml, libggml-base)
GGML_PROTOTYPES = {
    "ggml_backend_dev_count": ([], c_size_t),
    "ggml_backend_dev_get": ([c_size_t], c_void_p),
    "ggml_backend_dev_name": ([c_void_p], c_char_p),
    "ggml_backend_dev_description": ([c_void_p], c_char_p),
    "ggml_backend_dev_type": ([c_void_p], c_enum),
}
GGML_BASE_PROTOTYPES = {
    "ggml_version": ([], c_char_p),
    "ggml_commit": ([], c_char_p),
}
GGML_BACKEND_DEVICE_TYPES = {0: "cpu", 1: "gpu", 2: "igpu", 3: "accel", 4: "meta"}
GGML_TYPES = {"f32": 0, "f16": 1}
FLASH_ATTN_TYPES = {"auto": -1, "off": 0, "on": 1}
TOKEN_ATTR = {"unknown": 1 << 0, "unused": 1 << 1, "normal": 1 << 2, "control": 1 << 3, "user_defined": 1 << 4, "byte": 1 << 5}


class NativeLibraryError(RuntimeError):
    """The pinned llama.cpp build is missing or a different version is installed."""


def library_dir() -> pathlib.Path:
    import importlib.util

    spec = importlib.util.find_spec("llama_cpp")
    if spec is None or spec.origin is None:
        raise NativeLibraryError("llama-cpp-python is not installed (see the local installation guide)")
    return pathlib.Path(spec.origin).parent / "lib"


def library_version(path: pathlib.Path) -> str | None:
    """Version from the versioned soname the build installs (libllama.so -> libllama.so.X.Y.Z)."""
    name = path.resolve().name
    prefix = "libllama.so."
    if name.startswith(prefix):
        return name[len(prefix):]
    for candidate in sorted(path.parent.glob("libllama.so.*.*.*")):
        return candidate.name[len(prefix):]
    return None


def _declare(lib: ctypes.CDLL, prototypes: dict) -> None:
    for name, (argtypes, restype) in prototypes.items():
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = restype


class Native:
    """The loaded libraries with llikert's declarations applied."""

    def __init__(self, lib_dir: pathlib.Path | None = None, require_pinned: bool = True):
        lib_dir = lib_dir or library_dir()
        llama_path = lib_dir / "libllama.so"
        if not llama_path.exists():
            raise NativeLibraryError("libllama.so not found in the llama-cpp-python installation")
        self.llama_cpp_version = library_version(llama_path)
        if require_pinned and self.llama_cpp_version != PINNED_LLAMA_CPP_VERSION:
            raise NativeLibraryError(
                f"llama.cpp {self.llama_cpp_version or 'unknown'} is installed; llikert requires the pinned "
                f"{PINNED_LLAMA_CPP_VERSION} build of llama-cpp-python (see the local installation guide)"
            )
        # the ggml backends live next to libllama; RTLD_GLOBAL lets them resolve each other
        self.llama = ctypes.CDLL(str(llama_path), mode=ctypes.RTLD_GLOBAL)
        self.ggml = ctypes.CDLL(str(lib_dir / "libggml.so"), mode=ctypes.RTLD_GLOBAL)
        self.ggml_base = ctypes.CDLL(str(lib_dir / "libggml-base.so"), mode=ctypes.RTLD_GLOBAL)
        _declare(self.llama, PROTOTYPES)
        _declare(self.ggml, GGML_PROTOTYPES)
        _declare(self.ggml_base, GGML_BASE_PROTOTYPES)

    def devices(self) -> list[dict[str, str]]:
        out = []
        for i in range(self.ggml.ggml_backend_dev_count()):
            dev = self.ggml.ggml_backend_dev_get(i)
            out.append(
                {
                    "name": (self.ggml.ggml_backend_dev_name(dev) or b"").decode(),
                    "description": (self.ggml.ggml_backend_dev_description(dev) or b"").decode(),
                    "type": GGML_BACKEND_DEVICE_TYPES.get(self.ggml.ggml_backend_dev_type(dev), "other"),
                }
            )
        return out


def ctypes_offsets() -> dict:
    out = {}
    for struct in (llama_model_params, llama_context_params, llama_batch):
        out[struct.__name__] = {
            "sizeof": ctypes.sizeof(struct),
            "fields": {name: getattr(struct, name).offset for name, _ in struct._fields_},
        }
    return out
