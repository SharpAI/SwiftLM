#!/usr/bin/env bash
# test-contract.sh — OpenAI-compatibility contract tests.
#
# These assert behaviour the unit suite structurally cannot see: what the client actually
# receives over the wire, once real tokenisation decides where chunk boundaries fall.
# Every defect this file covers was found by review or by a user, never by a unit test
# (see issue #128).
#
# Usage: tests/test-contract.sh [binary] [port]

set -euo pipefail

BINARY="${1:-.build/release/SwiftLM}"
PORT="${2:-15414}"
HOST="127.0.0.1"
URL="http://$HOST:$PORT"
MODEL="${SWIFTLM_TEST_MODEL:-mlx-community/Qwen2.5-0.5B-Instruct-4bit}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
PASS=0; FAIL=0; SKIP=0; TOTAL=0
log()  { echo -e "${YELLOW}[contract]${NC} $*"; }
pass() { PASS=$((PASS + 1)); TOTAL=$((TOTAL + 1)); echo -e "  ${GREEN}✅ PASS${NC}: $*"; }
fail() { FAIL=$((FAIL + 1)); TOTAL=$((TOTAL + 1)); echo -e "  ${RED}❌ FAIL${NC}: $*"; }
# Some assertions depend on what the model is capable of, not on what the server
# guarantees. A tiny CI model cannot reliably emit tool calls or well-formed JSON, and a
# suite that goes red for that reason trains people to ignore it — so those report SKIP.
skip() { SKIP=$((SKIP + 1)); TOTAL=$((TOTAL + 1)); echo -e "  ⏭️  SKIP: $*"; }

SERVER_PID=""
cleanup() { [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

log "Starting server: $BINARY --model $MODEL --port $PORT --thinking"
"$BINARY" --model "$MODEL" --port "$PORT" --host "$HOST" --thinking > /tmp/SwiftLM-test-contract.log 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 120); do
    curl -sf "$URL/health" >/dev/null 2>&1 && break
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "Server died:"; tail -20 /tmp/SwiftLM-test-contract.log; exit 1; }
    sleep 2
done
log "Server ready"

chat() { curl -sf "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$1"; }

# ── 1. A stop sequence must never appear in the response ─────────────────────
# Issue #126: the check only fired on the chunk that completed the stop, so a stop
# spanning two token chunks had its first part streamed before the match was seen.
log "Test 1: stop sequence is not echoed (non-streaming)"
BODY=$(chat '{"model":"x","messages":[{"role":"user","content":"Count: one two three four five"}],"max_tokens":80,"temperature":0,"stop":["three"]}')
CONTENT=$(echo "$BODY" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"].get("content") or "")')
if echo "$CONTENT" | grep -q "three"; then
    fail "stop sequence 'three' leaked into content: $(echo "$CONTENT" | head -c 80)"
else
    pass "stop sequence absent from non-streaming content"
fi

log "Test 2: stop sequence is not echoed (streaming, may split across chunks)"
STREAM=$(curl -sf -N "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"x","messages":[{"role":"user","content":"Count: one two three four five"}],"max_tokens":80,"temperature":0,"stop":["three"],"stream":true}')
ASSEMBLED=$(echo "$STREAM" | python3 -c '
import json,sys
out=""
for line in sys.stdin:
    line=line.strip()
    if not line.startswith("data: ") or line=="data: [DONE]": continue
    try: d=json.loads(line[6:])
    except Exception: continue
    for ch in d.get("choices",[]):
        out += ch.get("delta",{}).get("content") or ""
print(out)')
if echo "$ASSEMBLED" | grep -q "three"; then
    fail "stop sequence leaked into the stream (known bug #126): $(echo "$ASSEMBLED" | head -c 80)"
else
    pass "stop sequence absent from assembled stream"
fi

# ── 2b. A stop sequence that cannot be a single token ────────────────────────
# "own fox" spans a token boundary by construction, so this exercises the straddling
# path specifically: before #126 was fixed, the client received "The quick brown".
log "Test 2b: stop sequence spanning a token boundary is not leaked"
PROMPT="Repeat this exactly: The quick brown fox jumps over the lazy dog"
STRADDLE=$(curl -sf -N "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":40,\"temperature\":0,\"stream\":true,\"stop\":[\"own fox\"]}" \
    | python3 -c '
import json,sys
out=""
for line in sys.stdin:
    line=line.strip()
    if not line.startswith("data: ") or line=="data: [DONE]": continue
    try: d=json.loads(line[6:])
    except Exception: continue
    for ch in d.get("choices",[]):
        out += ch.get("delta",{}).get("content") or ""
print(out)')
if echo "$STRADDLE" | grep -q "own"; then
    fail "a prefix of the stop sequence leaked: $(echo "$STRADDLE" | head -c 60)"
else
    pass "no prefix of a token-straddling stop sequence reached the client"
fi

# ── 3. Reasoning must not leak into content ──────────────────────────────────
# Issue #108: a template that pre-opens <think> left the whole reasoning block in
# `content`, with a stray closing tag inside it.
log "Test 3: no reasoning tags leak into content"
BODY=$(chat '{"model":"x","messages":[{"role":"user","content":"Name the capital of France. One word."}],"max_tokens":300,"temperature":0}')
CONTENT=$(echo "$BODY" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"].get("content") or "")')
if echo "$CONTENT" | grep -qE '</?think>|</?thinking>|<channel\|>'; then
    fail "reasoning tag leaked into content: $(echo "$CONTENT" | head -c 80)"
else
    pass "content carries no reasoning tags"
fi

# ── 4. Streaming and non-streaming must agree ────────────────────────────────
log "Test 4: streamed deltas reassemble to the non-streaming answer"
Q='{"model":"x","messages":[{"role":"user","content":"Reply with exactly: hello world"}],"max_tokens":20,"temperature":0'
NONSTREAM=$(chat "$Q}" | python3 -c 'import json,sys; print((json.load(sys.stdin)["choices"][0]["message"].get("content") or "").strip())')
STREAMED=$(curl -sf -N "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$Q,\"stream\":true}" | python3 -c '
import json,sys
out=""
for line in sys.stdin:
    line=line.strip()
    if not line.startswith("data: ") or line=="data: [DONE]": continue
    try: d=json.loads(line[6:])
    except Exception: continue
    for ch in d.get("choices",[]):
        out += ch.get("delta",{}).get("content") or ""
print(out.strip())')
if [ "$NONSTREAM" = "$STREAMED" ]; then
    pass "streaming and non-streaming agree ('$NONSTREAM')"
else
    fail "divergence — non-streaming '$NONSTREAM' vs streamed '$STREAMED'"
fi

# ── 5. Tool-call round trip ──────────────────────────────────────────────────
# Issue #108's second half: the assistant turn replayed into history must be clean
# enough for the model to use the tool result.
log "Test 5: two-round tool call"
TOOLS='[{"type":"function","function":{"name":"get_weather","description":"Get weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]'
R1=$(chat "{\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Paris? Use the tool.\"}],\"tools\":$TOOLS,\"max_tokens\":400,\"temperature\":0}")
TC=$(echo "$R1" | python3 -c 'import json,sys; m=json.load(sys.stdin)["choices"][0]["message"]; print((m.get("tool_calls") or [{}])[0].get("function",{}).get("name",""))')
CONTENT1=$(echo "$R1" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"].get("content") or "")')
if [ "$TC" = "get_weather" ]; then
    pass "round 1 emitted a get_weather tool call"
else
    skip "model did not emit a tool call (capability, not contract) — tag assertions below still apply"
fi
if echo "$CONTENT1" | grep -qE '</?think>'; then
    fail "round 1 content contains a reasoning tag"
else
    pass "round 1 content is free of reasoning tags"
fi

# ── 6. JSON mode ─────────────────────────────────────────────────────────────
log "Test 6: json_object mode returns parseable JSON"
BODY=$(chat '{"model":"x","messages":[{"role":"user","content":"Return a JSON object with key city set to Paris."}],"max_tokens":120,"temperature":0,"response_format":{"type":"json_object"}}')
CONTENT=$(echo "$BODY" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"].get("content") or "")')
# The server's guarantee is that it strips markdown fences; whether the model closes its
# braces within max_tokens is a model property.
if echo "$CONTENT" | grep -q '```'; then
    fail "json_object content still carries a markdown fence: $(echo "$CONTENT" | head -c 60)"
else
    pass "json_object content has no markdown fence"
fi
if echo "$CONTENT" | python3 -c 'import json,sys; json.loads(sys.stdin.read())' 2>/dev/null; then
    pass "json_object content parses as JSON"
else
    skip "model did not produce complete JSON (capability, not contract)"
fi

log "═══════════════════════════════════════"
log "Results: $PASS passed, $FAIL failed, $SKIP skipped, $TOTAL total"
log "═══════════════════════════════════════"
[ "$FAIL" -eq 0 ]
