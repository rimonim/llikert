#!/usr/bin/env bash
# Build the pinned llama-cpp-python wheel against llama.cpp v0.4.1 (decision D22).
#
#   tools/build-llama-cpp-python.sh cuda   # -> build/wheels/cuda/llama_cpp_python-0.3.35-*.whl
#   tools/build-llama-cpp-python.sh cpu    # -> build/wheels/cpu/...
#
# Environment: CUDA_ARCHITECTURES (default 89), PYTHON (a non-conda Python >= 3.10, see F2).
set -euo pipefail

BINDING_TAG=v0.3.35
LLAMA_CPP_TAG=v0.4.1
LLAMA_CPP_COMMIT=b29c606e28a01b1bc8c1351026a0fa6e616bf6c4
VARIANT=${1:?usage: $0 cuda|cpu}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC=$ROOT/build/llama-cpp-python
PYTHON=${PYTHON:-/usr/bin/python3}

if [ ! -d "$SRC" ]; then
  git clone --depth 1 --branch "$BINDING_TAG" --recurse-submodules --shallow-submodules \
    https://github.com/abetlen/llama-cpp-python.git "$SRC"
fi
git -C "$SRC/vendor/llama.cpp" fetch --depth 1 origin tag "$LLAMA_CPP_TAG"
git -C "$SRC/vendor/llama.cpp" checkout -q "$LLAMA_CPP_TAG"
actual=$(git -C "$SRC/vendor/llama.cpp" rev-parse HEAD)
[ "$actual" = "$LLAMA_CPP_COMMIT" ] || { echo "llama.cpp $LLAMA_CPP_TAG is $actual, expected $LLAMA_CPP_COMMIT" >&2; exit 1; }
rm -rf "$SRC/build"

case "$VARIANT" in
  cuda)
    # pin the toolkit root: otherwise CMake can link a distribution CUDA runtime from
    # /lib/x86_64-linux-gnu (CUDA 11 on Ubuntu 22.04) while compiling with a newer nvcc
    CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
    export CUDACXX="$CUDA_HOME/bin/nvcc"
    export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES:-89} -DCUDAToolkit_ROOT=$CUDA_HOME -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc"
    ;;
  cpu)  export CMAKE_ARGS="-DGGML_CUDA=off -DGGML_NATIVE=off -DGGML_AVX2=on -DGGML_F16C=on -DGGML_FMA=on" ;;
  *) echo "variant must be cuda or cpu" >&2; exit 2 ;;
esac

"$PYTHON" -m pip wheel --no-deps -w "$ROOT/build/wheels/$VARIANT" "$SRC"
ls -1 "$ROOT/build/wheels/$VARIANT"
