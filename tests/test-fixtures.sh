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
cleanup() { [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null; SERVER_PID=""; }
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

    local ready=0
    for _ in $(seq 1 60); do
        if curl -sf "$url/health" >/dev/null 2>&1; then ready=1; break; fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
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
        -d '{"messages":[{"role":"user","content":"ping"}],"max_tokens":4,"stream":false}' 2>/dev/null)

    # Random weights make the text meaningless, so assert on the token count instead.
    if [ -n "$body" ] && echo "$body" | python3 -c '
import json, sys
d = json.load(sys.stdin)
sys.exit(0 if d["usage"]["completion_tokens"] >= 1 else 1)' 2>/dev/null; then
        pass "$name loaded and generated"
    else
        fail "$name: no completion — $(echo "$body" | head -c 120)"
    fi

    cleanup
    sleep 1
}

log "Binary: $BINARY"
for name in dense stray-shard kv-shared-absent kv-shared-present; do
    log "Shape: $name"
    run_fixture "$name"
done

log "═══════════════════════════════════════"
log "Results: $PASS passed, $FAIL failed"
log "═══════════════════════════════════════"
[ "$FAIL" -eq 0 ]
