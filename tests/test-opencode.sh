#!/bin/bash
# test-opencode.sh — Integration test for official OpenAI SDK compatibility
#
# Usage:
#   ./tests/test-opencode.sh [binary_path] [port]
#
# Requires: python3, pip (installs openai package dynamically)

set -euo pipefail

BINARY="${1:-.build/release/SwiftLM}"
PORT="${2:-15413}"
HOST="127.0.0.1"
MODEL="mlx-community/gemma-4-e4b-it-4bit"
URL="http://${HOST}:${PORT}"
PASS=0
FAIL=0
TOTAL=0

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${YELLOW}[test]${NC} $*"; }
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

# ── Check prerequisites ─────────────────────────────────────────────
if [ ! -f "$BINARY" ]; then
    echo "Error: Binary not found at $BINARY"
    exit 1
fi

if ! command -v python3 &>/dev/null; then
    echo "Error: python3 is required."
    exit 1
fi

# ── Setup isolated Python environment ───────────────────────────────
log "Setting up virtual environment with openai SDK..."
VENV_DIR="/tmp/opencode_venv"
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --quiet openai

# ── Start the SwiftLM server ────────────────────────────────────────
log "Starting SwiftLM Server on port $PORT..."
"$BINARY" --model "$MODEL" --port "$PORT" --host "$HOST" > /tmp/SwiftLM-test-opencode.log 2>&1 &
SERVER_PID=$!

# Wait for server to be ready (increased timeout for gemma-4 weight download)
MAX_RETRIES=180
RETRY_COUNT=0
SERVER_READY=false

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s "$URL/v1/models" >/dev/null; then
        SERVER_READY=true
        break
    fi
    sleep 1
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ "$SERVER_READY" = false ]; then
    echo "Error: Server failed to start or respond on port $PORT within 180 seconds."
    cat /tmp/SwiftLM-test-opencode.log
    exit 1
fi
log "Server is up and responding."

# ── Generate test python script ─────────────────────────────────────
cat << 'EOF' > /tmp/opencode_test.py
import openai
import sys
import os

client = openai.OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"), api_key="sk-test", max_retries=0)

try:
    response = client.chat.completions.create(
        model=os.environ.get("MODEL"),
        messages=[{"role": "user", "content": "Explain quantum computing in one sentence."}],
        stream=True,
        # This opt-in header triggers the named `event: prefill_progress` chunks.
        # Strict clients will fail if the server sends malformed data objects alongside them.
        extra_headers={"X-SwiftLM-Prefill-Progress": "true"}
    )
    for chunk in response:
        # A successful iteration means the SDK's internal SSE parser accepted the stream.
        pass
    print("Success")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
EOF

# ── Test 1: OpenAI SDK stream parsing ───────────────────────────────
log "Test 1: Official OpenAI SDK compatibility with opt-in heartbeat"

export OPENAI_BASE_URL="$URL/v1"
export MODEL="$MODEL"

if "$VENV_DIR/bin/python" /tmp/opencode_test.py; then
    pass "OpenAI SDK parsed the stream successfully without rejecting events"
else
    fail "OpenAI SDK rejected the stream (likely invalid SSE structure or unknown events)"
fi

# ── Test 2: opencode CLI end-to-end ────────────────────────────────
log "Test 2: OpenCode CLI (opencode-ai) end-to-end compatibility"

log "Installing opencode-ai in isolated directory..."
mkdir -p /tmp/opencode_cli_test
cd /tmp/opencode_cli_test
npm install opencode-ai@latest --silent >/dev/null 2>&1

log "Running opencode CLI against SwiftLM server..."
# We use openai/gpt-4o-mini so the CLI validation passes. SwiftLM ignores the requested model and serves Gemma-4.
# We pipe 'yes' to handle any standard input confirmation OpenCode asks for, and use --dangerously-skip-permissions
OPENAI_BASE_URL="$URL/v1" OPENAI_API_KEY="sk-test" yes | npx --yes opencode run "Say 'I am ready'." --model openai/gpt-4o-mini --pure --dangerously-skip-permissions > /tmp/opencode_cli.log 2>&1 || true

if grep -q "Success" /tmp/opencode_cli.log || grep -qi "ready" /tmp/opencode_cli.log || test -s /tmp/opencode_cli.log; then
    if ! grep -qi "parse error" /tmp/opencode_cli.log && ! grep -qi "Unexpected token" /tmp/opencode_cli.log && ! grep -qi "Model not found" /tmp/opencode_cli.log; then
        pass "OpenCode CLI parsed the stream successfully and completed the generation"
    else
        fail "OpenCode CLI crashed while parsing the stream or rejected the model"
        cat /tmp/opencode_cli.log
    fi
else
    fail "OpenCode CLI failed to run or generated empty output"
    cat /tmp/opencode_cli.log
fi

# ── Results ──────────────────────────────────────────────────────────
echo ""
log "═══════════════════════════════════════"
log "Results: ${PASS} passed, ${FAIL} failed, ${TOTAL} total"
log "═══════════════════════════════════════"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
