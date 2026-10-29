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
# Absence alone would also pass if the fix over-withheld and dropped the tail, so assert
# the expected prefix actually arrived too.
if echo "$STRADDLE" | grep -q "own"; then
    fail "a prefix of the stop sequence leaked: $(echo "$STRADDLE" | head -c 60)"
elif ! echo "$STRADDLE" | grep -q "The quick br"; then
    fail "text before the stop was dropped — got: $(echo "$STRADDLE" | head -c 60)"
else
    pass "token-straddling stop: prefix withheld, preceding text delivered"
fi

# ── 2c. Same guarantee on the legacy /v1/completions endpoint ────────────────
log "Test 2c: /v1/completions does not leak a token-straddling stop sequence"
COMPLETION=$(curl -sf -N "$URL/v1/completions" -H 'Content-Type: application/json' \
    -d '{"model":"x","prompt":"Repeat exactly: The quick brown fox jumps over the lazy dog\n","max_tokens":40,"temperature":0,"stream":true,"stop":["own fox"]}' \
    | python3 -c '
import json,sys
out=""
for line in sys.stdin:
    line=line.strip()
    if not line.startswith("data: ") or line=="data: [DONE]": continue
    try: d=json.loads(line[6:])
    except Exception: continue
    for ch in d.get("choices",[]):
        out += ch.get("text") or ""
print(out)')
if echo "$COMPLETION" | grep -q "own"; then
    fail "/v1/completions leaked a stop prefix: $(echo "$COMPLETION" | head -c 60)"
elif ! echo "$COMPLETION" | grep -q "The quick br"; then
    fail "/v1/completions dropped text before the stop: $(echo "$COMPLETION" | head -c 60)"
else
    pass "/v1/completions withholds the prefix and delivers preceding text"
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

# ── 7. A second, independent SSE parser ──────────────────────────────────────
# Everything above goes through Python httpx. SwiftLM frames SSE with \r\n\r\n rather
# than the spec's \n\n; httpx tolerates that, and whether a different parser does is
# what this establishes (#134). Node is preinstalled on the runner and this needs no
# npm install, so it costs nothing — unlike the opencode CLI it partially replaces.
log "Test 7: SSE stream parsed by a non-Python client (Node)"
if command -v node >/dev/null 2>&1; then
    if node "$(dirname "$0")/test-sse-node.js" "$URL" "x"; then
        pass "Node client parsed the stream"
    else
        fail "Node client rejected the stream — see output above"
    fi
else
    skip "node not available on this machine"
fi

# ── 8. A prompt long enough to be prefilled in chunks ────────────────────────
# Below prefillStepSize (512 tokens) the generator's prepare() returns the prompt
# tokens without ever forwarding them, so a whole code path — chunked prefill —
# goes unexercised. Every other prompt in this repo's tests is under 80 characters,
# which is how a dual-model MTP crash on any real-sized prompt reached a merge
# queue with green CI (SharpAI/mlx-swift-lm#46). This closes the gap for the
# ordinary generate path only — CI runs no --mtp job, so the speculative variant
# of the same path stays uncovered until one exists (#128).
log "Test 8: prompt exceeding the prefill chunk size"
LONG_PROMPT=$(python3 -c '
# ~3000 tokens: comfortably past 512 even with an efficient tokenizer.
print(("The quick brown fox jumps over the lazy dog near the river bank. " * 190) + "Reply with the single word: done.")')
LONG_BODY=$(python3 -c '
import json, sys
print(json.dumps({"messages": [{"role": "user", "content": sys.argv[1]}],
                  "max_tokens": 16, "stream": False}))' "$LONG_PROMPT")
LONG_RESP=$(curl -sf --max-time 300 "$URL/v1/chat/completions" \
    -H 'Content-Type: application/json' -d "$LONG_BODY" 2>/dev/null || true)
if [ -z "$LONG_RESP" ]; then
    # A crash in the prefill path kills the connection rather than returning an error
    # body, so an empty response is the signal we are actually looking for here.
    fail "no response to a chunk-prefilled prompt — server may have died; see /tmp/SwiftLM-test-contract.log"
elif echo "$LONG_RESP" | python3 -c '
import json, sys
d = json.load(sys.stdin)
sys.exit(0 if d["choices"][0]["message"]["content"].strip() else 1)' 2>/dev/null; then
    pass "chunk-prefilled prompt produced content"
else
    fail "chunk-prefilled prompt returned no content: $(echo "$LONG_RESP" | head -c 120)"
fi

# ── 9. `seed` makes sampling reproducible, and a negative one is harmless ────
# Issue #219: the request seed only reached MLX's global generator, which the samplers
# don't draw from, so the same seed sampled differently every time; and UInt64(seed)
# trapped on a negative value, taking the whole server down with one request.
log "Test 9: the same seed samples the same answer"
SEEDED='{"model":"x","messages":[{"role":"user","content":"Pick a random animal and say only its name."}],"max_tokens":12,"temperature":1.0,"seed":4242}'
# The first call also fills the prompt cache; comparing two cache-hit calls keeps the
# assertion about the sampler rather than about prefill numerics.
chat "$SEEDED" >/dev/null
seeded_message() { chat "$SEEDED" | python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["choices"][0]["message"], sort_keys=True))'; }
S1=$(seeded_message); S2=$(seeded_message)
if [ -n "$S1" ] && [ "$S1" = "$S2" ]; then
    pass "seeded sampling is reproducible ($(echo "$S1" | head -c 60))"
else
    fail "same seed gave different output: $(echo "$S1" | head -c 60) vs $(echo "$S2" | head -c 60)"
fi

log "Test 9b: a negative seed is accepted and the server survives it"
NEG_CODE=$(curl -s -o /dev/null -w '%{http_code}' "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"max_tokens":3,"seed":-1}')
if [ "$NEG_CODE" = "200" ] && curl -sf "$URL/health" >/dev/null 2>&1; then
    pass "negative seed answered 200 and /health still responds"
else
    fail "negative seed: HTTP $NEG_CODE (000 = connection dropped), server alive: $(curl -sf "$URL/health" >/dev/null 2>&1 && echo yes || echo no)"
fi

# ── 10. Request shapes the OpenAI API allows ──────────────────────────────────
# `stop` may be a bare string; both request types decoded only the array form, and the
# decode failure surfaced as a 500 `server_error`, which the OpenAI SDKs retry.
log "Test 10: stop as a plain string"
BODY=$(chat '{"model":"x","messages":[{"role":"user","content":"Count: one two three four five"}],"max_tokens":80,"temperature":0,"stop":"three"}' || true)
CONTENT=$(echo "$BODY" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"].get("content") or "")' 2>/dev/null || echo "__no_body__")
if [ "$CONTENT" = "__no_body__" ]; then
    fail "stop as a string was rejected: $(echo "$BODY" | head -c 120)"
elif echo "$CONTENT" | grep -q "three"; then
    fail "stop string 'three' leaked into content: $(echo "$CONTENT" | head -c 80)"
else
    pass "a bare stop string is accepted and honoured"
fi

log "Test 10b: a request that does not decode is a 400 invalid_request_error"
CODE=$(curl -s -o /tmp/SwiftLM-contract-400.json -w '%{http_code}' "$URL/v1/chat/completions" \
    -H 'Content-Type: application/json' -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"stop":42}')
ERR_TYPE=$(python3 -c 'import json; e=json.load(open("/tmp/SwiftLM-contract-400.json"))["error"]; print(e.get("type"), e.get("param"))' 2>/dev/null || echo "unparseable")
if [ "$CODE" = "400" ] && [ "$ERR_TYPE" = "invalid_request_error stop" ]; then
    pass "400 invalid_request_error naming param 'stop'"
else
    fail "got HTTP $CODE with error '$ERR_TYPE' (expected 400 invalid_request_error stop)"
fi

# ── 11. finish_reason says when the model was cut off ─────────────────────────
# A client cannot tell a truncated completion from a finished one except through
# `finish_reason`, so each response path must report "length" when max_tokens ended it.
# Found by review: the non-streaming /v1/completions path hardcoded "stop" (#227).
SSE_FINISH='
import json,sys
reason=None
for line in sys.stdin:
    line=line.strip()
    if not line.startswith("data: ") or line=="data: [DONE]": continue
    try: d=json.loads(line[6:])
    except Exception: continue
    for ch in d.get("choices",[]):
        if ch.get("finish_reason"): reason=ch["finish_reason"]
print(reason)'
LONG_ASK="Write a very long story about a dragon, at least five hundred words."

log "Test 11: finish_reason is \"length\" when max_tokens ends a chat completion"
R=$(chat "{\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":\"$LONG_ASK\"}],\"max_tokens\":5,\"temperature\":0}" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["finish_reason"])')
[ "$R" = "length" ] && pass "chat, non-streaming: finish_reason=length" || fail "chat, non-streaming: finish_reason=$R (expected length)"

R=$(curl -sf -N "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":\"$LONG_ASK\"}],\"max_tokens\":5,\"temperature\":0,\"stream\":true}" | python3 -c "$SSE_FINISH")
[ "$R" = "length" ] && pass "chat, streaming: finish_reason=length" || fail "chat, streaming: finish_reason=$R (expected length)"

R=$(curl -sf -N "$URL/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"x\",\"prompt\":\"$LONG_ASK\",\"max_tokens\":5,\"temperature\":0,\"stream\":true}" | python3 -c "$SSE_FINISH")
[ "$R" = "length" ] && pass "completions, streaming: finish_reason=length" || fail "completions, streaming: finish_reason=$R (expected length)"

R=$(curl -sf "$URL/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"x\",\"prompt\":\"$LONG_ASK\",\"max_tokens\":5,\"temperature\":0}" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["finish_reason"])')
[ "$R" = "length" ] && pass "completions, non-streaming: finish_reason=length" || fail "completions, non-streaming: finish_reason=$R (expected length; #227)"

# ── 12. A stop sequence ends generation, and usage only counts what was generated ──
# The streaming paths stop at the match. The non-streaming ones used to keep generating
# to max_tokens or EOS and cut the text afterwards, so the same request billed far more
# completion_tokens (and spent the compute) when it was not streamed.
log "Test 12: completion_tokens after a stop sequence agree between streaming and non-streaming"
STOP_ASK="Write the numbers one to twenty as English words separated by commas, nothing else."
STOP_WORD="Five"
BASE_REQ="\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":\"$STOP_ASK\"}],\"max_tokens\":120,\"temperature\":0"
# The assertion is only meaningful if the stop sequence would actually fire, so check
# the unconstrained reply contains it first; that is a model capability, not a contract.
BASE_TEXT=$(chat "{$BASE_REQ}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"].get("content") or "")')
if ! echo "$BASE_TEXT" | grep -q "$STOP_WORD"; then
    skip "model's reply never contained '$STOP_WORD', so the stop sequence cannot fire"
else
    NS_TOKENS=$(chat "{$BASE_REQ,\"stop\":[\"$STOP_WORD\"]}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["usage"]["completion_tokens"])')
    ST_TOKENS=$(curl -sf -N "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{$BASE_REQ,\"stop\":[\"$STOP_WORD\"],\"stream\":true,\"stream_options\":{\"include_usage\":true}}" | python3 -c '
import json,sys
n=None
for line in sys.stdin:
    line=line.strip()
    if not line.startswith("data: ") or line=="data: [DONE]": continue
    try: d=json.loads(line[6:])
    except Exception: continue
    if d.get("usage"): n=d["usage"]["completion_tokens"]
print(n)')
    # The streamed count is a chunk count and can trail the real one by a few buffered tokens.
    if [ -n "$NS_TOKENS" ] && [ "$ST_TOKENS" != "None" ] && [ "$NS_TOKENS" -le $((ST_TOKENS + 4)) ]; then
        pass "stop at '$STOP_WORD': non-streaming $NS_TOKENS vs streaming $ST_TOKENS completion tokens"
    else
        fail "stop at '$STOP_WORD': non-streaming reports $NS_TOKENS completion tokens, streaming $ST_TOKENS — the non-streaming path kept generating past the stop sequence"
    fi
fi

log "Test 12b: /v1/completions stops generating at a stop sequence (non-streaming)"
COMP_PROMPT="Numbers in words: One, Two, Three, Four, Five, Six, Seven, Eight, Nine, Ten, Eleven, Twelve. Again:"
comp_req() { curl -sf "$URL/v1/completions" -H 'Content-Type: application/json' -d "$1"; }
COMP_BASE=$(comp_req "{\"model\":\"x\",\"prompt\":\"$COMP_PROMPT\",\"max_tokens\":60,\"temperature\":0}")
COMP_BASE_TEXT=$(echo "$COMP_BASE" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["text"])')
if ! echo "$COMP_BASE_TEXT" | grep -q "Five"; then
    skip "the completion never contained 'Five', so the stop sequence cannot fire"
else
    COMP_STOP=$(comp_req "{\"model\":\"x\",\"prompt\":\"$COMP_PROMPT\",\"max_tokens\":60,\"temperature\":0,\"stop\":[\"Five\"]}")
    COMP_BASE_N=$(echo "$COMP_BASE" | python3 -c 'import json,sys; print(json.load(sys.stdin)["usage"]["completion_tokens"])')
    COMP_STOP_N=$(echo "$COMP_STOP" | python3 -c 'import json,sys; print(json.load(sys.stdin)["usage"]["completion_tokens"])')
    COMP_STOP_TEXT=$(echo "$COMP_STOP" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["text"])')
    if [ "$COMP_STOP_N" -lt "$COMP_BASE_N" ] && ! echo "$COMP_STOP_TEXT" | grep -q "Five"; then
        pass "completions stop at 'Five': $COMP_STOP_N tokens vs $COMP_BASE_N without the stop"
    else
        fail "completions stop at 'Five': $COMP_STOP_N tokens vs $COMP_BASE_N without the stop (text: $(echo "$COMP_STOP_TEXT" | head -c 60))"
    fi
fi

# ── 13. JSON mode streams its reply ─────────────────────────────────────────
# A JSON-mode reply is buffered so a hallucinated prefix can be dropped, and the buffer
# was never flushed when the stream ended. The generator holds a `{`-led reply as a
# possible tool call and delivers it as one chunk, so a JSON-mode stream sent nothing
# while the non-streaming twin was fine (#233).
log "Test 13: json_object mode streams the content it returns non-streaming"
JSON_REQ='"model":"x","messages":[{"role":"user","content":"Return exactly this JSON and nothing else: {\"ok\": true}"}],"max_tokens":60,"temperature":0,"response_format":{"type":"json_object"}'
NS_JSON=$(chat "{$JSON_REQ}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"].get("content") or "")')
ST_JSON=$(curl -sf -N "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{$JSON_REQ,\"stream\":true}" | python3 -c '
import json,sys
out=[]
for line in sys.stdin:
    line=line.strip()
    if not line.startswith("data: ") or line=="data: [DONE]": continue
    try: d=json.loads(line[6:])
    except Exception: continue
    for c in d.get("choices",[]):
        out.append(c.get("delta",{}).get("content") or "")
print("".join(out))')
if [ -z "$ST_JSON" ]; then
    fail "json_object stream carried no content (non-streaming returned: $(echo "$NS_JSON" | head -c 60))"
elif [ "$ST_JSON" = "$NS_JSON" ]; then
    pass "json_object stream matches the non-streaming content: $(echo "$ST_JSON" | head -c 60)"
else
    fail "json_object stream differs from non-streaming: '$(echo "$ST_JSON" | head -c 60)' vs '$(echo "$NS_JSON" | head -c 60)'"
fi

# ── 14. An abandoned non-streaming request stops generating ───────────────────
# The SSE handlers cancel generation when the client goes away (their next write fails).
# A non-streaming request writes nothing until the end, so the server never noticed the
# client had left: the answer ran to max_tokens for nobody and held the slot, and with
# the default --parallel 1 every later request queued behind it. Clients that time out
# and retry (the OpenAI SDKs' default) stacked up several such generations.
log "Test 14: a non-streaming request the client abandons stops generating"
metric() { curl -sf "$URL/metrics" | awk -v m="$1" '$1 == m { print $2 }'; }
TOKENS_BEFORE=$(metric swiftlm_tokens_generated_total)
# curl closes the socket after half a second; left alone, the request would generate 1500 tokens.
curl -s --max-time 0.5 "$URL/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":\"$LONG_ASK\"}],\"max_tokens\":1500,\"temperature\":0.9}" >/dev/null 2>&1 || true
# Once the server has noticed, the request is no longer active and its token count is the
# few chunks it managed before the cancel, not a full answer.
ACTIVE=1
for _ in $(seq 1 20); do
    ACTIVE=$(metric swiftlm_requests_active)
    [ "$ACTIVE" = "0" ] && break
    sleep 0.25
done
TOKENS_AFTER=$(metric swiftlm_tokens_generated_total)
ABANDONED_TOKENS=$((TOKENS_AFTER - TOKENS_BEFORE))
if [ "$ACTIVE" = "0" ] && [ "$ABANDONED_TOKENS" -lt 400 ]; then
    pass "the abandoned request stopped after $ABANDONED_TOKENS tokens and released its slot"
else
    fail "the abandoned request generated $ABANDONED_TOKENS tokens; requests_active=$ACTIVE 5 s after the client left"
fi

# ── 15. A tool-call-only turn saves the prompt cache ─────────────────────────
# #245: the save at `.info` waited for a text chunk, and a turn the model answers with
# nothing but a tool call never produces one. The next turn of an agent loop (tool
# result, follow-up) then re-prefilled the whole conversation. The server log shows the
# hit: the follow-up must reuse at least the whole prompt of the tool-call turn.
log "Test 15: a tool-call-only turn saves the prompt cache"
SERVER_LOG=/tmp/SwiftLM-test-contract.log
hit_count() { grep -c 'Prompt cache HIT' "$SERVER_LOG" || true; }
last_hit_reused() { grep -o 'Prompt cache HIT[^:]*: [0-9]*' "$SERVER_LOG" | tail -1 | grep -o '[0-9]*$'; }
# Reads a non-streaming response or an SSE stream (with include_usage) and prints
# "<first tool name> <prompt_tokens>".
tool_turn_summary() {
    python3 -c '
import json, sys
raw = sys.stdin.read()
if raw.lstrip().startswith("{"):
    r = json.loads(raw); m = r["choices"][0]["message"]
    name = ((m.get("tool_calls") or [{}])[0].get("function") or {}).get("name", "")
    print(name, r.get("usage", {}).get("prompt_tokens", 0)); sys.exit()
name, prompt = "", 0
for line in raw.splitlines():
    if not line.startswith("data: {"): continue
    ev = json.loads(line[6:])
    for c in ev.get("choices") or []:
        for tc in (c.get("delta") or {}).get("tool_calls") or []:
            name = name or (tc.get("function") or {}).get("name", "")
    prompt = (ev.get("usage") or {}).get("prompt_tokens") or prompt
print(name, prompt)'
}
# $1: label, $2: a system prompt no other test uses, $3: extra fields for the tool-call turn.
tool_turn_then_follow_up() {
    local LABEL="$1" SYS="$2" EXTRA="$3"
    local HISTORY="{\"role\":\"system\",\"content\":\"$SYS\"},{\"role\":\"user\",\"content\":\"What is the weather in Paris? Use the tool.\"}"
    local SUMMARY NAME PROMPT_TOKENS HITS_BEFORE REUSED
    SUMMARY=$(chat "{\"model\":\"x\",\"messages\":[$HISTORY],\"tools\":$TOOLS,\"max_tokens\":200,\"temperature\":0$EXTRA}" | tool_turn_summary)
    NAME=${SUMMARY%% *}; PROMPT_TOKENS=${SUMMARY##* }
    if [ "$NAME" != "get_weather" ]; then
        skip "$LABEL: model did not answer with a tool call (capability, not contract)"
        return
    fi
    HITS_BEFORE=$(hit_count)
    # The follow-up replays the tool call and adds its result; its prompt starts with the
    # whole prompt of the tool-call turn, which must now be in the cache.
    chat "{\"model\":\"x\",\"messages\":[$HISTORY,{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"id\":\"call_15\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\": \\\"Paris\\\"}\"}}]},{\"role\":\"tool\",\"tool_call_id\":\"call_15\",\"content\":\"18 C, cloudy\"}],\"tools\":$TOOLS,\"max_tokens\":40,\"temperature\":0}" >/dev/null
    for _ in $(seq 1 20); do
        [ "$(hit_count)" -gt "$HITS_BEFORE" ] && break
        sleep 0.25
    done
    if [ "$(hit_count)" -le "$HITS_BEFORE" ]; then
        fail "$LABEL: the follow-up after a tool-call turn got no prompt cache hit at all"
        return
    fi
    REUSED=$(last_hit_reused)
    if [ "${REUSED:-0}" -ge "$PROMPT_TOKENS" ]; then
        pass "$LABEL: the follow-up reused $REUSED cached tokens, the tool-call turn's prompt was $PROMPT_TOKENS"
    else
        fail "$LABEL: the follow-up reused only $REUSED cached tokens; the tool-call turn's prompt was $PROMPT_TOKENS"
    fi
}
tool_turn_then_follow_up "non-streaming" "You are a weather desk. Call get_weather for any weather question and say nothing else." ""
tool_turn_then_follow_up "streaming" "You are a weather kiosk. Call get_weather for any weather question and say nothing else." ',"stream":true,"stream_options":{"include_usage":true}'

log "═══════════════════════════════════════"
log "Results: $PASS passed, $FAIL failed, $SKIP skipped, $TOTAL total"
log "═══════════════════════════════════════"
[ "$FAIL" -eq 0 ]
