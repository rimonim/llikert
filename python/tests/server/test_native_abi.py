"""ctypes struct layouts in llikert.server.native must match the pinned llama.h (finding F1).

Needs a C compiler and the llama.cpp headers for the pinned tag. Set LLIKERT_LLAMA_CPP_SOURCE
to the llama.cpp source tree, or keep the default development location build/llama-cpp-python/vendor/llama.cpp.
"""

import json
import os
import pathlib
import re
import shutil
import subprocess

import pytest

from llikert.server import native

HERE = pathlib.Path(__file__).parent
REPO = HERE.parents[2]
SOURCE = pathlib.Path(os.environ.get("LLIKERT_LLAMA_CPP_SOURCE", REPO / "build/llama-cpp-python/vendor/llama.cpp"))


@pytest.fixture(scope="module")
def c_layout(tmp_path_factory):
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None or not (SOURCE / "include/llama.h").is_file():
        pytest.skip("C compiler or pinned llama.cpp headers not available")
    exe = tmp_path_factory.mktemp("abi") / "abi_offsets"
    subprocess.run(
        [cc, "-std=c11", f"-I{SOURCE / 'include'}", f"-I{SOURCE / 'ggml/include'}", str(HERE / "native/abi_offsets.c"), "-o", str(exe)],
        check=True,
    )
    return json.loads(subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout)


def test_struct_layouts_match_header(c_layout):
    assert native.ctypes_offsets() == c_layout


def test_declared_functions_exist_in_header():
    header = SOURCE / "include/llama.h"
    if not header.is_file():
        pytest.skip("pinned llama.cpp headers not available")
    text = header.read_text()
    missing = [name for name in native.PROTOTYPES if not re.search(rf"\b{name}\s*\(", text)]
    assert missing == []


def test_header_is_pinned_version():
    cmake = SOURCE / "CMakeLists.txt"
    if not cmake.is_file():
        pytest.skip("pinned llama.cpp source not available")
    assert native.PINNED_LLAMA_CPP_VERSION == "0.4.1"


def test_installed_library_is_pinned():
    try:
        lib_dir = native.library_dir()
    except native.NativeLibraryError:
        pytest.skip("llama-cpp-python not installed")
    assert native.library_version(lib_dir / "libllama.so") == native.PINNED_LLAMA_CPP_VERSION
    lib = native.Native()
    assert all(hasattr(lib.llama, name) for name in native.PROTOTYPES)
