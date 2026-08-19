#!/bin/sh
set -eu

LLAMA_SERVER_BIN=${LLAMA_SERVER_BIN:-$(command -v llama-server || true)}
LLAMA_HF_REPO=${LLAMA_HF_REPO:-Qwen/Qwen3-4B-GGUF:Q4_K_M}
LLAMA_MODEL_ID=${LLAMA_MODEL_ID:-Qwen/Qwen3-4B-GGUF}
LLAMA_PORT=${LLAMA_PORT:-$((20000 + $$ % 10000))}
GATEWAY_PORT=${GATEWAY_PORT:-$((LLAMA_PORT + 1))}
GATEWAY_BIN=${GATEWAY_BIN:-gateway/build/sanitize/forge_gateway}
TEST_TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/forge-gateway-llama.XXXXXX")

cleanup() {
  if [ -n "${GATEWAY_PID:-}" ]; then
    kill "$GATEWAY_PID" 2>/dev/null || true
    wait "$GATEWAY_PID" 2>/dev/null || true
  fi
  if [ -n "${LLAMA_PID:-}" ]; then
    kill "$LLAMA_PID" 2>/dev/null || true
    wait "$LLAMA_PID" 2>/dev/null || true
  fi
  rm -rf "$TEST_TMP_DIR"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

if [ -z "$LLAMA_SERVER_BIN" ]; then
  echo "llama-server not found; install llama.cpp or set LLAMA_SERVER_BIN" >&2
  exit 2
fi
if [ ! -x "$GATEWAY_BIN" ]; then
  echo "gateway binary not found at $GATEWAY_BIN; build it first" >&2
  exit 2
fi

if [ -n "${LLAMA_MODEL:-}" ]; then
  "$LLAMA_SERVER_BIN" \
    --model "$LLAMA_MODEL" \
    --host 127.0.0.1 \
    --port "$LLAMA_PORT" \
    --ctx-size 1024 \
    --parallel 2 \
    >"$TEST_TMP_DIR/llama.log" 2>&1 &
else
  "$LLAMA_SERVER_BIN" \
    --hf-repo "$LLAMA_HF_REPO" \
    --host 127.0.0.1 \
    --port "$LLAMA_PORT" \
    --ctx-size 1024 \
    --parallel 2 \
    >"$TEST_TMP_DIR/llama.log" 2>&1 &
fi
LLAMA_PID=$!

attempt=0
until curl --silent --fail --max-time 2 "http://127.0.0.1:$LLAMA_PORT/health" >/dev/null; do
  attempt=$((attempt + 1))
  if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
    echo "llama.cpp exited before becoming healthy" >&2
    tail -n 80 "$TEST_TMP_DIR/llama.log" >&2
    exit 1
  fi
  if [ "$attempt" -ge 3600 ]; then
    echo "llama.cpp did not become healthy within 60 minutes" >&2
    tail -n 80 "$TEST_TMP_DIR/llama.log" >&2
    exit 1
  fi
  sleep 1
done

"$GATEWAY_BIN" \
  --listen-port "$GATEWAY_PORT" \
  --primary-port "$LLAMA_PORT" \
  --primary-token-capacity 4096 \
  --max-queue-requests 4 \
  --max-queue-tokens 8192 \
  --pool-size 2 \
  --request-timeout-ms 120000 \
  --disable-fallback \
  >"$TEST_TMP_DIR/gateway.log" 2>&1 &
GATEWAY_PID=$!

attempt=0
until curl --silent --fail --max-time 2 "http://127.0.0.1:$GATEWAY_PORT/healthz" >/dev/null; do
  attempt=$((attempt + 1))
  if ! kill -0 "$GATEWAY_PID" 2>/dev/null; then
    echo "gateway exited before becoming healthy" >&2
    tail -n 80 "$TEST_TMP_DIR/gateway.log" >&2
    exit 1
  fi
  if [ "$attempt" -ge 120 ]; then
    echo "gateway did not become healthy within 2 minutes" >&2
    tail -n 80 "$TEST_TMP_DIR/gateway.log" >&2
    exit 1
  fi
  sleep 1
done

curl --silent --show-error --fail --max-time 120 \
  -H 'Content-Type: application/json' \
  -H 'X-Client-ID: llama-nonstream' \
  -d "{\"model\":\"$LLAMA_MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with OK.\"}],\"max_tokens\":16,\"stream\":false}" \
  "http://127.0.0.1:$GATEWAY_PORT/v1/chat/completions" \
  >"$TEST_TMP_DIR/nonstream.json"
grep -q '"choices"' "$TEST_TMP_DIR/nonstream.json"

curl --silent --show-error --fail --no-buffer --max-time 120 \
  -H 'Content-Type: application/json' \
  -H 'X-Client-ID: llama-stream' \
  -d "{\"model\":\"$LLAMA_MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with OK.\"}],\"max_tokens\":16,\"stream\":true}" \
  "http://127.0.0.1:$GATEWAY_PORT/v1/chat/completions" \
  >"$TEST_TMP_DIR/stream.sse"
grep -q '^data:' "$TEST_TMP_DIR/stream.sse"
grep -q 'data: \[DONE\]' "$TEST_TMP_DIR/stream.sse"

curl --silent --show-error --fail --max-time 5 \
  "http://127.0.0.1:$GATEWAY_PORT/metrics" \
  >"$TEST_TMP_DIR/metrics.txt"
grep -q 'forge_gateway_routing_decisions_total' "$TEST_TMP_DIR/metrics.txt"
grep -q 'decision="primary"} 2' "$TEST_TMP_DIR/metrics.txt"

echo "llama.cpp integration passed: $LLAMA_HF_REPO (non-stream + SSE + metrics)"
