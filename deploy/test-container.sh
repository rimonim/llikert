#!/usr/bin/env bash
# Local acceptance test for the service image (run from the repository root, with GPU access).
#
#   deploy/test-container.sh [image] [models-dir]
#
# Checks startup failures, readiness, auth, the HTTP fidelity gate, both clients, logs,
# non-root execution, and fingerprint stability across a restart.
set -uo pipefail
IMAGE=${1:-llikert-service:0.1.0.dev0-cuda12.9}
MODELS=${2:-$PWD/models}
MODEL_FILE=qwen3-4b-instruct-2507-f32.gguf
SHA=a5733d5a25e8b824e1ef9d7e75449ffd8f8ec582de8ac253420e4423a7f0e357
PORT=18095
TOKEN=$(python3 -c "import secrets; print(secrets.token_hex(24))")
PY=${PYTHON:-.venv/bin/python}
OUT=build/container-test
mkdir -p "$OUT"
failures=0
check() { if eval "$2"; then echo "PASS  $1"; else echo "FAIL  $1"; failures=$((failures + 1)); fi; }
run() { docker run --rm --gpus all -v "$MODELS:/repository:ro" "$@"; }

# -- startup failures (no long-running container) --
run "$IMAGE" > "$OUT/no-model.log" 2>&1; rc=$?
check "missing LLIKERT_MODEL exits non-zero with a message" '[ $rc -ne 0 ] && grep -q "set LLIKERT_MODEL" "$OUT/no-model.log"'
run -e LLIKERT_MODEL=/repository/$MODEL_FILE -e LLIKERT_AUTH=none "$IMAGE" > "$OUT/auth-none.log" 2>&1; rc=$?
check "auth none on 0.0.0.0 is refused" '[ $rc -ne 0 ] && grep -q "only allowed on a loopback" "$OUT/auth-none.log"'
run -e LLIKERT_MODEL=/repository/$MODEL_FILE -e LLIKERT_API_TOKEN=short "$IMAGE" > "$OUT/short-token.log" 2>&1; rc=$?
check "short token is refused" '[ $rc -ne 0 ] && grep -q "at least 16 characters" "$OUT/short-token.log"'

start() {
  docker run -d --rm --name llikert-acceptance --gpus all -p 127.0.0.1:$PORT:8080 -v "$MODELS:/repository:ro" \
    -e LLIKERT_MODEL=/repository/$MODEL_FILE -e LLIKERT_MODEL_SHA256="$1" -e LLIKERT_API_TOKEN="$TOKEN" "$IMAGE" > /dev/null
}
wait_health() {
  for i in $(seq 1 180); do
    code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" || true)
    [ "$code" = 200 ] && { echo "$i"; return 0; }
    docker inspect llikert-acceptance > /dev/null 2>&1 || return 1
    sleep 1
  done
  return 1
}
stop() { docker stop llikert-acceptance > /dev/null 2>&1 || true; }
trap stop EXIT

# -- wrong model hash: the process exits instead of staying unready --
timeout 180 docker run --rm --gpus all -v "$MODELS:/repository:ro" -e LLIKERT_MODEL=/repository/$MODEL_FILE \
  -e LLIKERT_MODEL_SHA256=0000000000000000000000000000000000000000000000000000000000000000 -e LLIKERT_API_TOKEN="$TOKEN" \
  "$IMAGE" > "$OUT/wrong-sha.log" 2>&1; rc=$?
check "wrong model sha256 exits with status 3 and a message" '[ $rc -eq 3 ] && grep -q "sha256 does not match" "$OUT/wrong-sha.log"'

# -- normal start --
start "$SHA"
code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" || true)
check "health is 503 or unreachable before ready" '[ "$code" != 200 ]'
ready=$(wait_health); rc=$?
check "service becomes ready (${ready:-?} s)" '[ $rc -eq 0 ]'
check "health needs no credentials and reveals nothing" '[ "$(curl -s http://127.0.0.1:$PORT/health)" = "{\"status\":\"ready\"}" ]'
check "info without token is 401" '[ "$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/v1/info)" = 401 ]'
check "openapi is not exposed with token auth" '[ "$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/openapi.json)" = 404 ]'
check "container runs as non-root" '[ "$(docker exec llikert-acceptance id -u)" = 10001 ]'
curl -s -H "Authorization: Bearer $TOKEN" "http://127.0.0.1:$PORT/v1/info" > "$OUT/info-1.json"

LLIKERT_TOKEN=$TOKEN $PY tools/fidelity_gate_http.py --url "http://127.0.0.1:$PORT" --json "$OUT/gate.json" > /dev/null 2>&1; rc=$?
check "HTTP fidelity gate passes (450 items)" '[ $rc -eq 0 ]'

LLIKERT_URL="http://127.0.0.1:$PORT" LLIKERT_TOKEN=$TOKEN $PY - > "$OUT/python-client.log" 2>&1 <<'PYEOF'
from llikert import Scorer, ScoringTask
task = ScoringTask(name="Communicative function", instructions="Classify the text's primary communicative function.",
                   categories=["description", "question", "request"], responses=["A", "B", "C"])
texts = ["Where is the station?", "Please close the door.", None, "", "CONTAINER-LOG-CANARY The museum opens at ten."]
with Scorer.connect(wait=60) as engine:
    result = engine.score(texts, task=engine.prepare(task), checkpoint="build/container-test/py-run", progress=False)
    result.save("build/container-test/python-result.json")
    print(result)
PYEOF
check "Python client scores against the container" 'grep -q "3 ok" "$OUT/python-client.log"'
rm -rf "$OUT/r-run"
LLIKERT_URL="http://127.0.0.1:$PORT" LLIKERT_TOKEN=$TOKEN Rscript -e '
suppressMessages(devtools::load_all("r/llikert", quiet = TRUE))
task <- scoring_task("Communicative function", "Classify the text'"'"'s primary communicative function.",
                     c("description", "question", "request"), c("A", "B", "C"))
engine <- scorer_connect(wait = 60)
texts <- c("Where is the station?", "Please close the door.", NA, "", "CONTAINER-LOG-CANARY The museum opens at ten.")
result <- score_items(texts, task = prepare_task(task, engine), engine = engine, checkpoint = "build/container-test/r-run", progress = FALSE)
write_result(result, "build/container-test/r-result.json")
print(table(llikert_result_diagnostics(result)$status))
' > "$OUT/r-client.log" 2>&1
check "R client scores against the container" 'grep -q "ok" "$OUT/r-client.log" && [ -f "$OUT/r-result.json" ]'
check "R and Python results are identical" "$PY -c \"
import json
a = json.load(open('$OUT/python-result.json')); b = json.load(open('$OUT/r-result.json'))
[d.pop(k) for d in (a, b) for k in ('created_at', 'client')]
raise SystemExit(0 if a == b else 1)\""

docker logs llikert-acceptance > "$OUT/service.log" 2>&1
check "service logs contain no texts or token" '! grep -q "CONTAINER-LOG-CANARY" "$OUT/service.log" && ! grep -q "$TOKEN" "$OUT/service.log"'

stop; sleep 3
start "$SHA"
wait_health > /dev/null
curl -s -H "Authorization: Bearer $TOKEN" "http://127.0.0.1:$PORT/v1/info" > "$OUT/info-2.json"
check "engine fingerprint is stable across restarts" "$PY -c \"
import json
a = json.load(open('$OUT/info-1.json'))['engine_fingerprint']; b = json.load(open('$OUT/info-2.json'))['engine_fingerprint']
raise SystemExit(0 if a == b else 1)\""
cp "$OUT/info-1.json" "$OUT/engine-info.json"

echo "failures: $failures"
exit $failures
