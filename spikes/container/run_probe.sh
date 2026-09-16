#!/usr/bin/env bash
# M0 container probe: build image, run with GPU and /repository mount, poll /health.
# Requires docker daemon access (run via `sg docker -c` if group membership is new).
set -euo pipefail
cd "$(dirname "$0")/../.."
MODEL=${1:-qwen2.5-0.5b-instruct-q8_0.gguf}
PORT=${PORT:-18081}
docker build -f spikes/container/Dockerfile -t llikert-probe:m0 . > build/logs/docker-probe-build.log 2>&1
echo "image: $(docker image inspect llikert-probe:m0 --format '{{.Id}} {{.Size}}')"
docker run -d --rm --name llikert-probe ${GPU_ARGS:---gpus all} -p "$PORT:8080" \
  -v "$PWD/models:/repository:ro" -e LLIKERT_MODEL="/repository/$MODEL" llikert-probe:m0 > /dev/null
trap 'docker stop llikert-probe > /dev/null 2>&1 || true' EXIT
first=""
for i in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "localhost:$PORT/health" || true)
  [ -z "$first" ] && [ "$code" != "000" ] && first="t=${i}s health=$code"
  if [ "$code" = 200 ]; then echo "first response: $first; ready at t=${i}s"; break; fi
  sleep 1
done
echo "probe: $(curl -s "localhost:$PORT/probe")"
docker logs llikert-probe 2>&1 | grep -iE "CUDA0|error|found [0-9]+ CUDA" | head -5
