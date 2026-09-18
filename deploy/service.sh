#!/usr/bin/env bash
# Run the LLikert scoring service in a container: the one command for everything.
#
#   deploy/service.sh start       start the service (creating the container the first time)
#   deploy/service.sh stop        stop it and free the graphics card
#   deploy/service.sh status      running? ready? how much graphics memory? recent activity?
#   deploy/service.sh restart     stop, then start
#   deploy/service.sh recreate    apply a new image, key or setting (removes and creates the container)
#   deploy/service.sh logs [n]    the last n log lines (default 50)
#   deploy/service.sh selfcheck   check the model, graphics card and library without serving
#
# Settings come from deploy/llikert.env if that file exists, otherwise from these defaults.
# Run from the repository folder.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
[ -f "$ROOT/deploy/llikert.env" ] && . "$ROOT/deploy/llikert.env"

CONTAINER=${LLIKERT_CONTAINER:-llikert}
IMAGE=${LLIKERT_IMAGE:-llikert-service:0.1.0.dev0-cuda12.9}
MODELS_DIR=${LLIKERT_MODELS_DIR:-$ROOT/models}
MODEL_FILE=${LLIKERT_MODEL_FILE:-qwen3-4b-instruct-2507-f32.gguf}
MODEL_SHA256=${LLIKERT_MODEL_SHA256:-a5733d5a25e8b824e1ef9d7e75449ffd8f8ec582de8ac253420e4423a7f0e357}
KEY_FILE=${LLIKERT_KEY_FILE:-$ROOT/llikert-access-key.txt}
PORT=${LLIKERT_PORT:-8080}
BIND=${LLIKERT_BIND:-127.0.0.1}
N_CTX=${LLIKERT_N_CTX:-4096}

ACTION=${1:-status}

exists() { docker inspect "$CONTAINER" > /dev/null 2>&1; }
running() { [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" = "true" ]; }
health_code() { curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:$PORT/health" 2>/dev/null || echo 000; }
requests_since() { docker logs --since "$1" "$CONTAINER" 2>&1 | grep -c "score request_id="; }

gpu_memory() {
  local pid
  pid=$(docker inspect -f '{{.State.Pid}}' "$CONTAINER" 2>/dev/null) || return
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null |
    awk -v pid="$pid" -F', ' '$1 == pid {print $2}'
}

require_token() {
  if [ -z "${LLIKERT_API_TOKEN:-}" ]; then
    if [ -f "$KEY_FILE" ]; then
      LLIKERT_API_TOKEN=$(tr -d '\r\n' < "$KEY_FILE")
    else
      echo "No access key found. Create one with:" >&2
      echo "  python3 -c \"import secrets; print(secrets.token_hex(32))\" > $KEY_FILE && chmod 600 $KEY_FILE" >&2
      exit 1
    fi
  fi
}

create() {
  require_token
  if [ ! -f "$MODELS_DIR/$MODEL_FILE" ]; then
    echo "Model file not found: $MODELS_DIR/$MODEL_FILE" >&2
    echo "Prepare it with: deploy/prepare-model.sh $MODELS_DIR" >&2
    exit 1
  fi
  if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
    echo "Image not found: $IMAGE" >&2
    echo "Build it with: docker build -f deploy/Dockerfile.cuda -t $IMAGE --build-arg SOURCE_REVISION=\$(git rev-parse HEAD) ." >&2
    exit 1
  fi
  docker run -d --name "$CONTAINER" --restart unless-stopped --gpus all \
    -p "$BIND:$PORT:8080" \
    -v "$MODELS_DIR:/repository:ro" \
    -e LLIKERT_MODEL="/repository/$MODEL_FILE" \
    -e LLIKERT_MODEL_SHA256="$MODEL_SHA256" \
    -e LLIKERT_API_TOKEN="$LLIKERT_API_TOKEN" \
    -e LLIKERT_N_CTX="$N_CTX" \
    "$IMAGE" > /dev/null || exit 1
  echo "created container '$CONTAINER' from $IMAGE"
}

wait_ready() {
  local seconds=${1:-300}
  for i in $(seq 1 "$seconds"); do
    [ "$(health_code)" = 200 ] && { echo "ready after ${i}s; address: http://$BIND:$PORT"; return 0; }
    running || { echo "the service stopped while starting; see: $0 logs" >&2; return 1; }
    sleep 1
  done
  echo "not ready after ${seconds}s; see: $0 logs" >&2
  return 1
}

case "$ACTION" in
  start)
    if running; then
      echo "service '$CONTAINER' is already running"
      exit 0
    fi
    if exists; then
      docker start "$CONTAINER" > /dev/null || exit 1
      echo "starting '$CONTAINER' (loading the model takes about 20 seconds)"
    else
      create
      echo "starting '$CONTAINER' (loading the model takes about 20 seconds)"
    fi
    wait_ready
    ;;
  stop)
    exists || { echo "service '$CONTAINER' does not exist yet"; exit 0; }
    running || { echo "service '$CONTAINER' is already stopped (using no graphics memory)"; exit 0; }
    recent=$(requests_since 5m)
    [ "$recent" -gt 0 ] && echo "note: $recent scoring requests in the last 5 minutes; someone may be working. Runs with a checkpoint can be resumed afterwards."
    docker stop "$CONTAINER" > /dev/null && echo "stopped '$CONTAINER'; the graphics card is free"
    ;;
  restart)
    "$0" stop && "$0" start
    ;;
  recreate)
    if exists; then
      recent=$(requests_since 5m)
      [ "$recent" -gt 0 ] && echo "note: $recent scoring requests in the last 5 minutes; someone may be working."
      docker rm -f "$CONTAINER" > /dev/null && echo "removed the old container"
    fi
    create
    echo "starting '$CONTAINER' (loading the model takes about 20 seconds)"
    wait_ready
    ;;
  status)
    if ! exists; then
      echo "service '$CONTAINER': not created yet; start it with: $0 start"
      exit 0
    fi
    if running; then
      code=$(health_code)
      echo "service '$CONTAINER': running since $(docker inspect -f '{{.State.StartedAt}}' "$CONTAINER" | cut -c1-19)"
      echo "  address:         http://$BIND:$PORT"
      echo "  health:          $([ "$code" = 200 ] && echo ready || echo "not ready yet (HTTP $code)")"
      echo "  graphics memory: $(gpu_memory || echo unknown)"
      echo "  recent scoring:  $(requests_since 1h) requests in the last hour"
      echo "  stop it with:    $0 stop"
    else
      echo "service '$CONTAINER': stopped (using no graphics memory)"
      echo "  start it with:   $0 start"
    fi
    ;;
  logs)
    exists || { echo "service '$CONTAINER' does not exist yet" >&2; exit 1; }
    docker logs --tail "${2:-50}" "$CONTAINER"
    ;;
  selfcheck)
    docker run --rm --gpus all -v "$MODELS_DIR:/repository:ro" "$IMAGE" selfcheck --model "/repository/$MODEL_FILE"
    ;;
  *)
    echo "usage: $0 [start|stop|restart|recreate|status|logs [n]|selfcheck]" >&2
    exit 2
    ;;
esac
