#!/usr/bin/env bash
# test-fixtures.sh — load every synthetic checkpoint shape and generate a token.
#
# These fixtures are a few hundred kilobytes each and live in the repository, so this
# needs no download, no model cache, and no network. See scripts/make-test-fixtures.py
# for why they exist and what they do and do not cover.
#
# Each shape corresponds to a defect that reached users:
#   dense               baseline — nothing special, catches gross breakage
#   stray-shard         #118: a .safetensors beside the index but absent from it
#   kv-shared-absent    #120: gemma-4-e4b shape, shared layers ship no k/v
#   kv-shared-present   b674:  gemma-4-e2b shape, shared layers ship k/v anyway
#   moe-nested          #112: expert count nested under text_config only, with a
#                       decoy count under audio_config; plus the fused
#                       experts.gate_up_proj split that sanitize performs
#
# The output is gibberish by construction — the weights are random. A fixture passes
# when the server loads it and produces *a* token, which is what exercises config
# parsing, weight-key resolution, sanitisation and layer materialisation.
#
# Usage: ./tests/test-fixtures.sh [binary] [port]

set -uo pipefail

BINARY="${1:-.build/release/SwiftLM}"
PORT="${2:-15460}"
HOST="127.0.0.1"
FIXTURE_DIR="$(cd "$(dirname "$0")" && pwd)/fixtures"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
PASS=0; FAIL=0

log()  { echo -e "${YELLOW}[fixtures]${NC} $*"; }
pass() { PASS=$((PASS + 1)); echo -e "  ${GREEN}✅ PASS${NC}: $*"; }
fail() { FAIL=$((FAIL + 1)); echo -e "  ${RED}❌ FAIL${NC}: $*"; }

SERVER_PID=""
# `kill` only asks. Without waiting for the process to go, the next fixture binds the
# same port while the previous server may still hold it — see the readiness loop below
# for why that is worse than a flake.
cleanup() {
    if [ -n "$SERVER_PID" ]; then
        kill "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null
    fi
    SERVER_PID=""
}
trap cleanup EXIT

run_fixture() {
    local name="$1"
    local dir="$FIXTURE_DIR/$name"
    local url="http://$HOST:$PORT"
    local logfile="/tmp/SwiftLM-test-fixture-$name.log"

    if [ ! -d "$dir" ]; then
        fail "$name: fixture directory missing — run scripts/make-test-fixtures.py"
        return
    fi

    "$BINARY" --model "$dir" --port "$PORT" --host "$HOST" > "$logfile" 2>&1 &
    SERVER_PID=$!

    # Liveness is checked *before* the health probe on purpose. The other order lets a
    # server that failed to bind (because the previous one still held the port) pass as
    # ready on a probe answered by that previous server — the assertions then run
    # against the wrong checkpoint and report a false pass rather than a failure.
    local ready=0
    for _ in $(seq 1 60); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
        if curl -sf "$url/health" >/dev/null 2>&1; then ready=1; break; fi
        sleep 1
    done

    if [ "$ready" -ne 1 ]; then
        fail "$name: server did not start — $(grep -m1 -E '^Error|Fatal' "$logfile" || echo 'see '"$logfile")"
        cleanup
        return
    fi

    local body
    body=$(curl -sf --max-time 60 "$url/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d '{"messages":[{"role":"user","content":"ping"}],"max_tokens":4,"temperature":0,"stream":false}' 2>/dev/null)

    # What is being proved is that the checkpoint loaded and a forward pass ran, so the
    # assertion is on prefill and a well-formed response — not on how many tokens came
    # back. Random weights sample randomly: an early EOS yields zero completion tokens,
    # which made an earlier version of this check pass or fail run to run. temperature 0
    # removes the sampling variance; prompt_tokens is what actually evidences the prefill.
    if [ -n "$body" ] && echo "$body" | python3 -c '
import json, sys
d = json.load(sys.stdin)
ok = (d["usage"]["prompt_tokens"] >= 1
      and isinstance(d["choices"][0]["message"]["content"], str)
      and d["choices"][0].get("finish_reason"))
sys.exit(0 if ok else 1)' 2>/dev/null; then
        pass "$name loaded, prefilled and returned a well-formed completion"
    else
        fail "$name: no completion — $(echo "$body" | head -c 120)"
    fi

    cleanup
    sleep 1
}

log "Binary: $BINARY"
for name in dense stray-shard kv-shared-absent kv-shared-present moe-nested; do
    log "Shape: $name"
    run_fixture "$name"
done

# The MoE fixture again with --stream-experts, to assert the *config-level* gate.
#
# #112: expert counts are spelled and nested differently per family, and a model
# misread as dense had --stream-experts silently dropped and was then materialised
# whole until the OS killed it. The check is that the config gate does not reject —
# not that streaming actually engages. Those are two separate gates: gemma4 passes
# detection but has no StreamableMoE conformance, so the second one legitimately
# declines. Asserting on the first is what tracks #112.
#
# This is a live check rather than a decorative one: `gemma4` contains no "moe", so
# the model_type fallback in modelTypeImpliesMoE cannot rescue it, and the count
# exists only inside text_config. Reverting detection to a top-level single-key form
# makes this fail.
log "Shape: moe-nested (nested expert count is detected)"
MOE_LOG="/tmp/SwiftLM-test-fixture-moe-stream.log"
"$BINARY" --model "$FIXTURE_DIR/moe-nested" --port "$PORT" --host "$HOST" \
    --stream-experts > "$MOE_LOG" 2>&1 &
SERVER_PID=$!
ready=0
for _ in $(seq 1 60); do
    kill -0 "$SERVER_PID" 2>/dev/null || break
    curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1 && { ready=1; break; }
    sleep 1
done
if [ "$ready" -ne 1 ]; then
    fail "moe-nested did not start with --stream-experts"
elif grep -q "is not MoE" "$MOE_LOG"; then
    fail "nested expert count missed: $(grep 'is not MoE' "$MOE_LOG" | head -1)"
else
    pass "moe-nested: expert count found under text_config, decoy container ignored"
fi
cleanup

log "═══════════════════════════════════════"
log "Results: $PASS passed, $FAIL failed"
log "═══════════════════════════════════════"
[ "$FAIL" -eq 0 ]
