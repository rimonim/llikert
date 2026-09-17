#!/usr/bin/env bash
# Prepare the supported model file, qwen3-4b-instruct-2507-f32.gguf, in a throwaway container.
#
#   deploy/prepare-model.sh [output-dir]           # default output: ./models
#
# Downloads the official Qwen/Qwen3-4B-Instruct-2507 weights at a fixed revision (about 8 GB),
# converts them with the converter from llama.cpp v0.4.1 (the version the service uses), and
# checks the result against the expected SHA-256. Needs Docker and about 25 GB of free disk space;
# no GPU. If the weights were already downloaded, set HF_WEIGHTS_DIR to that folder to skip the download.
set -euo pipefail

OUT=$(mkdir -p "${1:-models}" && cd "${1:-models}" && pwd)
IMAGE=python:3.10-slim-bookworm@sha256:68d914ec641a0b69267ce65184d000a2bc3a9ee2590ab702b82250ab2385735a
EXPECTED=a5733d5a25e8b824e1ef9d7e75449ffd8f8ec582de8ac253420e4423a7f0e357
MOUNT_WEIGHTS=()
if [ -n "${HF_WEIGHTS_DIR:-}" ]; then
  MOUNT_WEIGHTS=(-v "$(cd "$HF_WEIGHTS_DIR" && pwd):/weights/Qwen3-4B-Instruct-2507:ro")
fi

docker run --rm -v "$OUT:/out" "${MOUNT_WEIGHTS[@]}" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" bash -euo pipefail -c '
REVISION=cdbee75f17c01a7cc42f958dc650907174af0554
pip install --quiet --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.14.0+cpu
pip install --quiet --no-cache-dir transformers==5.17.0 tokenizers==0.23.2 safetensors==0.8.0 sentencepiece==0.2.2 \
    protobuf==7.36.1 numpy==2.2.6 huggingface_hub==1.31.0 pyyaml==6.0.3 requests==2.34.2 tqdm==4.70.1
python - <<PY
import io, tarfile, urllib.request
url = "https://github.com/ggml-org/llama.cpp/archive/refs/tags/v0.4.1.tar.gz"
tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(url).read()), mode="r:gz").extractall("/src")
PY
pip install --quiet --no-cache-dir --no-deps /src/llama.cpp-0.4.1/gguf-py
# the converter names the model after this folder, so the folder name is part of the checksum
WEIGHTS=/weights/Qwen3-4B-Instruct-2507
if [ ! -d "$WEIGHTS" ]; then
  echo "Downloading Qwen/Qwen3-4B-Instruct-2507 at revision $REVISION ..."
  # only the files used for the verified conversion (a model card would add metadata and change the checksum)
  python -c "from huggingface_hub import snapshot_download; snapshot_download(\"Qwen/Qwen3-4B-Instruct-2507\", revision=\"$REVISION\", local_dir=\"$WEIGHTS\", allow_patterns=[\"*.safetensors\", \"model.safetensors.index.json\", \"config.json\", \"generation_config.json\", \"tokenizer.json\", \"tokenizer_config.json\", \"vocab.json\", \"merges.txt\", \"LICENSE\"])"
fi
echo "Converting to F32 GGUF ..."
python /src/llama.cpp-0.4.1/convert_hf_to_gguf.py "$WEIGHTS" --outtype f32 --outfile /out/qwen3-4b-instruct-2507-f32.gguf > /tmp/convert.log 2>&1 \
  || { tail -20 /tmp/convert.log; exit 1; }
chown "$HOST_UID:$HOST_GID" /out/qwen3-4b-instruct-2507-f32.gguf
'

echo "Checking the result ..."
actual=$(sha256sum "$OUT/qwen3-4b-instruct-2507-f32.gguf" | cut -d" " -f1)
if [ "$actual" = "$EXPECTED" ]; then
  echo "OK: $OUT/qwen3-4b-instruct-2507-f32.gguf (sha256 $actual)"
else
  echo "MISMATCH: got sha256 $actual, expected $EXPECTED. Do not use this file." >&2
  exit 1
fi
