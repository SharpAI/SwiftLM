#!/bin/bash
# test-omni.sh — Omni (Vision + Audio) Integration tests for SwiftLM
#
# Usage:
#   ./tests/test-omni.sh [binary_path] [port]

set -euo pipefail

BINARY="${1:-.build/release/SwiftLM}"
PORT="${2:-15413}"
HOST="127.0.0.1"
MODEL="mlx-community/gemma-4-e4b-it-4bit" # CI Small Omni
URL="http://${HOST}:${PORT}"
PASS=0
FAIL=0
TOTAL=0

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${YELLOW}[test-omni]${NC} $*"; }
pass() { PASS=$((PASS + 1)); TOTAL=$((TOTAL + 1)); echo -e "  ${GREEN}✅ PASS${NC}: $*"; }
fail() { FAIL=$((FAIL + 1)); TOTAL=$((TOTAL + 1)); echo -e "  ${RED}❌ FAIL${NC}: $*"; }

cleanup() {
    if [ -n "${SERVER_PID:-}" ]; then
        log "Stopping server (PID $SERVER_PID)"
        kill -9 "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ── Start server ─────────────────────────────────────────────────────
log "Starting server: $BINARY --model $MODEL --port $PORT --vision --audio"
"$BINARY" --model "$MODEL" --port "$PORT" --host "$HOST" --vision --audio &
SERVER_PID=$!

log "Waiting for server to be ready (this may take a while on first run)..."
MAX_WAIT=600  # 10 minutes for model download
for i in $(seq 1 "$MAX_WAIT"); do
    if curl -sf "$URL/health" >/dev/null 2>&1; then
        log "Server ready after ${i}s"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Error: Server process died"
        exit 1
    fi
    sleep 1
done

if ! curl -sf "$URL/health" >/dev/null 2>&1; then
    echo "Error: Server did not become ready in ${MAX_WAIT}s"
    exit 1
fi

# ── Test Omni ──────────────────────────────────────────────────────────
mkdir -p /tmp/omni_test

# Gen Audio
cat << 'EOF' > /tmp/omni_test/gen.py
import wave, struct, math
with wave.open('/tmp/omni_test/test.wav', 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(16000)
    for i in range(16000):
        v = int(math.sin(i * 440.0 * 2.0 * math.pi / 16000.0) * 10000.0)
        w.writeframes(struct.pack('<h', v))
EOF
python3 /tmp/omni_test/gen.py
BASE64_AUDIO=$(base64 -i /tmp/omni_test/test.wav | tr -d '\n')

# 28x28 black PNG (requires multiple of 28 for Qwen2-VL patch embedder)
BASE64_IMG="iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAIAAAD9b0jDAAAAGUlEQVR4nO3BMQEAAADCoPVPbQdvoAAA6DQJTAABRMAOLAAAAABJRU5ErkJggg=="

# Payload with both audio and image
cat <<EOF > /tmp/omni_test/payload.json
{"model":"$MODEL","max_tokens":100,"messages":[{"role":"user","content":[{"type":"text","text":"What color is the image? Please also respond to the audio beep."},{"type":"image_url","image_url":{"url":"data:image/png;base64,${BASE64_IMG}"}},{"type":"input_audio","input_audio":{"data":"${BASE64_AUDIO}","format":"wav"}}]}]}
EOF

COMPLETION=$(curl -sf -X POST "$URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d @"/tmp/omni_test/payload.json")

if echo "$COMPLETION" | jq -e '.choices[0].message.content' >/dev/null 2>&1; then
    CONTENT=$(echo "$COMPLETION" | jq -r '.choices[0].message.content')
    pass "Omni successfully analyzed both vision and audio. Output: \"$CONTENT\""
else
    fail "Omni completion failed: $COMPLETION"
    exit 1
fi

rm -rf /tmp/omni_test
exit 0
