#!/bin/sh
# Container entrypoint for the LLikert scoring service.
#
# Configuration comes from the environment; weights are never baked into the image.
#   LLIKERT_MODEL         required: path to the GGUF file, e.g. /repository/qwen3-4b-instruct-2507-f32.gguf
#   LLIKERT_MODEL_SHA256  recommended: expected SHA-256 of that file (startup fails on mismatch)
#   LLIKERT_AUTH          token (default) | platform (only behind a platform that authenticates requests)
#   LLIKERT_API_TOKEN     bearer token for LLIKERT_AUTH=token (at least 16 characters)
#   LLIKERT_PORT          default 8080
#   LLIKERT_N_CTX         default 4096
#   LLIKERT_MAX_ITEMS     default 64
#   LLIKERT_QUEUE         default 4
# Execution settings (device, batch size, flash attention, KV type) stay at the supported defaults.
# Passing arguments runs `llikert <args>` instead, e.g. `selfcheck --model ...`.
set -eu

if [ "$#" -gt 0 ]; then
  exec llikert "$@"
fi

: "${LLIKERT_MODEL:?set LLIKERT_MODEL to the GGUF file path, for example /repository/qwen3-4b-instruct-2507-f32.gguf}"
if [ ! -f "$LLIKERT_MODEL" ]; then
  echo "llikert: LLIKERT_MODEL does not name a file inside the container" >&2
  exit 2
fi

set -- serve \
  --model "$LLIKERT_MODEL" \
  --host 0.0.0.0 \
  --port "${LLIKERT_PORT:-8080}" \
  --auth "${LLIKERT_AUTH:-token}" \
  --n-ctx "${LLIKERT_N_CTX:-4096}" \
  --max-items "${LLIKERT_MAX_ITEMS:-64}" \
  --queue "${LLIKERT_QUEUE:-4}"
if [ -n "${LLIKERT_MODEL_SHA256:-}" ]; then
  set -- "$@" --model-sha256 "$LLIKERT_MODEL_SHA256"
fi
exec llikert "$@"
