#!/usr/bin/env bash
# Start, stop and check the LLikert service container.
#
#   deploy/service.sh status     is it running, is it ready, how much graphics memory it uses
#   deploy/service.sh stop       stop it and free the graphics card
#   deploy/service.sh start      start it again (same settings as when it was created)
#   deploy/service.sh restart    stop, then start
#   deploy/service.sh logs [n]   the last n log lines (default 50)
#
# The container is created once with the `docker run` command in deploy/README.md; this
# script only manages it afterwards. Set LLIKERT_CONTAINER (default "llikert") and
# LLIKERT_PORT (default 8080) if you used different values.
set -uo pipefail

NAME=${LLIKERT_CONTAINER:-llikert}
PORT=${LLIKERT_PORT:-8080}
ACTION=${1:-status}

running() { [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" = "true" ]; }
exists() { docker inspect "$NAME" > /dev/null 2>&1; }
health_code() { curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:$PORT/health" 2>/dev/null || echo 000; }

gpu_memory() {
  local pid
  pid=$(docker inspect -f '{{.State.Pid}}' "$NAME" 2>/dev/null) || return
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null |
    awk -v pid="$pid" -F', ' '$1 == pid {print $2}'
}

wait_ready() {
  local seconds=${1:-180}
  for i in $(seq 1 "$seconds"); do
    [ "$(health_code)" = 200 ] && { echo "ready after ${i}s"; return 0; }
    running || { echo "the container stopped while starting; see: $0 logs" >&2; return 1; }
    sleep 1
  done
  echo "not ready after ${seconds}s; see: $0 logs" >&2
  return 1
}

case "$ACTION" in
  status)
    if ! exists; then
      echo "container '$NAME': not created yet (see the 'Start the service' step in deploy/README.md)"
      exit 0
    fi
    if running; then
      code=$(health_code)
      memory=$(gpu_memory)
      echo "container '$NAME': running since $(docker inspect -f '{{.State.StartedAt}}' "$NAME" | cut -c1-19)"
      echo "  health:          $([ "$code" = 200 ] && echo "ready" || echo "not ready yet (HTTP $code)")"
      echo "  graphics memory: ${memory:-none reported}"
      echo "  recent scoring:  $(docker logs --since 1h "$NAME" 2>&1 | grep -c "score request_id=") requests in the last hour"
      echo "  stop it with:    $0 stop"
    else
      echo "container '$NAME': stopped (using no graphics memory)"
      echo "  start it with:   $0 start"
    fi
    ;;
  stop)
    exists || { echo "container '$NAME' does not exist"; exit 1; }
    if ! running; then
      echo "container '$NAME' is already stopped"
      exit 0
    fi
    recent=$(docker logs --since 5m "$NAME" 2>&1 | grep -c "score request_id=")
    [ "$recent" -gt 0 ] && echo "note: $recent scoring requests in the last 5 minutes; someone may be working. Runs with a checkpoint can be resumed afterwards."
    docker stop "$NAME" > /dev/null && echo "stopped '$NAME'; the graphics card is free"
    ;;
  start)
    exists || { echo "container '$NAME' does not exist; create it with the 'Start the service' step in deploy/README.md" >&2; exit 1; }
    running && { echo "container '$NAME' is already running"; exit 0; }
    docker start "$NAME" > /dev/null && echo "starting '$NAME' (loading the model takes about 20 seconds)"
    wait_ready 300
    ;;
  restart)
    "$0" stop && "$0" start
    ;;
  logs)
    docker logs --tail "${2:-50}" "$NAME"
    ;;
  *)
    echo "usage: $0 [status|stop|start|restart|logs [n]]" >&2
    exit 2
    ;;
esac
