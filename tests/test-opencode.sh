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

# ── Test 2: opencode-shaped agent request ──────────────────────────
# Previously this installed opencode-ai from npm and ran the CLI, which the CI runner
# OOM-kills (`Killed: 9`) after the server has already answered correctly — a red job
# that says nothing about SwiftLM. What is worth testing is the request shape opencode
# sends: a system prompt, a tool catalogue, and a streamed response whose tool-call
# deltas a strict client can reassemble. That is exercised directly here, with no npm.
log "Test 2: opencode-shaped agent request (streaming + tools)"

cat << 'PYEOF' > /tmp/opencode_agent_test.py
import json, os, sys
import openai

client = openai.OpenAI(base_url=os.environ["OPENAI_BASE_URL"], api_key="sk-test", max_retries=0)

# The shape opencode sends: a coding-agent system prompt plus its tool catalogue.
TOOLS = [
    {"type": "function", "function": {
        "name": "bash",
        "description": "Execute a shell command",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read",
        "description": "Read a file from disk",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
]
MESSAGES = [
    {"role": "system", "content": "You are a coding agent. Use the provided tools when helpful."},
    {"role": "user", "content": "List the files in the current directory."},
]

try:
    stream = client.chat.completions.create(
        model=os.environ["MODEL"], messages=MESSAGES, tools=TOOLS,
        stream=True, max_tokens=200, temperature=0,
        # opencode's networking layer injects this into every streaming request. It makes
        # SwiftLM append a terminal chunk with an empty `choices` array, which is exactly
        # the shape a strict SSE client can trip on — and which no SDK-based test covered.
        stream_options={"include_usage": True},
    )
except Exception as e:
    print(f"Error: request rejected: {e}")
    sys.exit(1)

# Reassemble exactly as a strict client does: tool-call deltas are keyed by index.
calls, chunks, finish = {}, 0, None
try:
    for chunk in stream:
        chunks += 1
        for choice in chunk.choices:
            if choice.finish_reason:
                finish = choice.finish_reason
            for tc in (choice.delta.tool_calls or []):
                if tc.index is None:
                    print("Error: tool_call delta has no index — cannot be reassembled")
                    sys.exit(1)
                slot = calls.setdefault(tc.index, {"name": "", "args": ""})
                if tc.function and tc.function.name:
                    slot["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    slot["args"] += tc.function.arguments
except Exception as e:
    print(f"Error: SSE stream failed to parse: {e}")
    sys.exit(1)

if chunks == 0:
    print("Error: stream produced no chunks")
    sys.exit(1)
if finish is None:
    print("Error: stream never reported a finish_reason")
    sys.exit(1)

for index, call in calls.items():
    if not call["name"]:
        print(f"Error: tool call {index} has no function name")
        sys.exit(1)
    try:
        json.loads(call["args"] or "{}")
    except json.JSONDecodeError as e:
        print(f"Error: tool call {call['name']} arguments are not valid JSON: {e}")
        sys.exit(1)

if not calls:
    # Not a failure: whether the model chooses a tool is its business, and hard-failing
    # would make this job flaky. But say so loudly, otherwise the test can quietly go
    # vacuous — green forever while the reassembly path stops being exercised.
    print("WARNING: no tool call emitted — the tool-call reassembly path was NOT exercised")

print(f"Success: {chunks} chunks, finish_reason={finish}, tool_calls={len(calls)}")
PYEOF

set +e
AGENT_OUT=$("$VENV_DIR/bin/python" /tmp/opencode_agent_test.py 2>&1)
AGENT_EXIT=$?
set -e

if [ $AGENT_EXIT -eq 0 ]; then
    pass "opencode-shaped request handled — $AGENT_OUT"
else
    fail "opencode-shaped request failed: $AGENT_OUT"
fi

# ── Results ──────────────────────────────────────────────────────────
echo ""
log "═══════════════════════════════════════"
log "Results: ${PASS} passed, ${FAIL} failed, ${TOTAL} total"
log "═══════════════════════════════════════"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
