#!/bin/bash
# test-hybrid-cache.sh — Hybrid prompt-cache checkpoint E2E
#
# Usage:
#   ./tests/test-hybrid-cache.sh [binary_path] [port]
#
# Requires: curl, jq.
#
# Why this exists
# ---------------
# Hybrid models (recurrent Mamba/GatedDeltaNet layers plus attention) cannot rewind
# recurrent state, so a prompt that diverges from a cached one used to re-prefill from
# zero. `hybridCheckpointPositions` / `recurrentSnapshot` / `restoreExactPrefix` now
# snapshot the recurrent state at turn boundaries and resume from the newest checkpoint
# before the divergence.
#
# The unit suite covers that bookkeeping against synthetic caches. Nothing covered the
# end-to-end claim, on a real hybrid model, that a checkpoint resume computes the same
# thing as a full prefill — which is exactly what can break silently. Generic prompt-cache
# tests (tests/test-server.sh Test 26) only exercise identical-prefix reuse, which never
# enters the checkpoint branch.
#
# Two assertions, and both matter:
#   - Checkpoint branch taken. The log must show a checkpoint restore that supplied a
#     worthwhile prefix. Without this the test would still pass on a silent fallback to a
#     full prefill, i.e. the feature being broken back to a miss.
#   - Resume == full prefill. The same diverged prompt is decoded twice: once on a server
#     warmed with the unedited prompt (checkpoint resume), once on a freshly started server
#     (full prefill). Greedy decoding must agree byte for byte.
#
# The second run needs a server with an empty cache, so this script starts and stops the
# server twice rather than reusing test-server.sh's single-server scaffold.

set -uo pipefail

BINARY="${1:-.build/release/SwiftLM}"
PORT="${2:-15417}"
HOST="127.0.0.1"
# Must be a hybrid (recurrent + attention) checkpoint. Qwen3.5 is GatedDeltaNet + attention
# and the 0.8B build is small enough to prefetch in CI.
MODEL="${SWIFTLM_TEST_MODEL:-mlx-community/Qwen3.5-0.8B-MLX-4bit}"
URL="http://${HOST}:${PORT}"
PAYLOAD_DIR="${TMPDIR:-/tmp}/swiftlm-hybrid-cache"
MAX_WAIT=600  # 10 minutes for a cold model load

PASS=0
FAIL=0
TOTAL=0

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${YELLOW}[hybrid-cache]${NC} $*"; }
pass() { PASS=$((PASS + 1)); TOTAL=$((TOTAL + 1)); echo -e "  ${GREEN}✅ PASS${NC}: $*"; }
fail() { FAIL=$((FAIL + 1)); TOTAL=$((TOTAL + 1)); echo -e "  ${RED}❌ FAIL${NC}: $*"; }

SERVER_PID=""

cleanup() {
    if [ -n "$SERVER_PID" ]; then
        log "Stopping server (PID $SERVER_PID)"
        kill -9 "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ── Prerequisites ────────────────────────────────────────────────────
if [ ! -f "$BINARY" ]; then
    echo "Error: Binary not found at $BINARY"
    echo "Run 'swift build -c release' first."
    exit 1
fi

if ! command -v jq &>/dev/null; then
    echo "Error: jq is required. Install with: brew install jq"
    exit 1
fi

# ── Payload helpers ──────────────────────────────────────────────────
# Three ~1500-token user turns. The checkpoint spacing rule (`hybridCheckpointPositions`,
# minGap 2048) snapshots the first turn start after the system prompt plus any later turn
# start at least 2048 tokens on. A two-turn conversation leaves only the anchor, which lands
# immediately before the divergence and makes the restore near worthless (observed: 48 of
# 3148 tokens reused). The third turn is what places a checkpoint far enough ahead of the
# edit to skip a real span.
mkdir -p "$PAYLOAD_DIR"

build_payloads() {
    python3 - "$PAYLOAD_DIR" <<'PY'
import json, os, random, sys

out = sys.argv[1]
random.seed(11)
VOCAB = ("the quarterly report indicates that revenue rose in all regions while operating "
         "costs stayed flat management expects moderate growth next period though supply "
         "chain and currency risks remain we advise caution on capital allocation and a "
         "further review at the next meeting efficiency gains continued and headcount was "
         "unchanged throughout the quarter").split()
def filler(n):
    return " ".join(random.choice(VOCAB) for _ in range(n))

system = "You are a precise assistant. Reply with one short sentence only. " + filler(30)
turn1 = "Turn one report.\n" + filler(1500) + "\nEnd of turn one."
turn2 = "Turn two report.\n" + filler(1500) + "\nEnd of turn two."
# The divergence sits at the very start of the last turn, so the shared prefix runs to the
# end of turn two and the checkpoint taken at turn two's start is well before it.
turn3_base = "Mark this turn as M3.\n" + filler(1500) + "\nEnd of turn three."
turn3_edit = "Mark this turn as EDITED3.\n" + filler(1500) + "\nEnd of turn three."

base = [
    {"role": "system", "content": system},
    {"role": "user", "content": turn1},
    {"role": "assistant", "content": "OK1"},
    {"role": "user", "content": turn2},
    {"role": "assistant", "content": "OK2"},
    {"role": "user", "content": turn3_base},
]
diverged = base[:-1] + [{"role": "user", "content": turn3_edit}]

def write(name, messages):
    body = {"model": "x", "messages": messages, "temperature": 0.0,
            "max_tokens": 16, "stream": False}
    with open(os.path.join(out, name), "w") as f:
        json.dump(body, f)

write("base.json", base)
write("diverged.json", diverged)
PY
}

# ── Server lifecycle ─────────────────────────────────────────────────
# Each call needs a server with a state fresh enough to be controlled, so this returns the
# PID and the caller stops it before the next phase.
start_server() {
    local log_file="$1"
    : > "$log_file"
    "$BINARY" --model "$MODEL" --port "$PORT" --host "$HOST" --no-vision \
        > "$log_file" 2>&1 &
    SERVER_PID=$!

    for _ in $(seq 1 "$MAX_WAIT"); do
        if curl -sf "$URL/health" >/dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Error: server process died. Log tail:"
            tail -25 "$log_file"
            return 1
        fi
        sleep 1
    done
    echo "Error: server not ready after ${MAX_WAIT}s"
    tail -25 "$log_file"
    return 1
}

stop_server() {
    if [ -n "$SERVER_PID" ]; then
        kill -9 "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
}

# POST a payload and echo the assistant content, or fail loudly.
send() {
    local payload="$1" out="$2"
    if ! curl -sf --max-time 900 -X POST "$URL/v1/chat/completions" \
        -H 'Content-Type: application/json' -d @"$payload" -o "$out"; then
        echo "Error: request to $URL failed" >&2
        return 1
    fi
    jq -r '.choices[0].message.content // empty' "$out"
}

# ── Run ──────────────────────────────────────────────────────────────
log "Building payloads in $PAYLOAD_DIR"
build_payloads

WARM_LOG="$PAYLOAD_DIR/warm.log"
FRESH_LOG="$PAYLOAD_DIR/fresh.log"

# Phase 1 — warm the cache with the unedited prompt, then send the diverged prompt. The
# second request is the one that must resume from a checkpoint.
log "Phase 1: warm cache, then diverged prompt (checkpoint resume expected)"
if ! start_server "$WARM_LOG"; then
    fail "Phase 1: server did not start"
    echo "Summary: $PASS/$TOTAL passed"
    exit 1
fi

if ! send "$PAYLOAD_DIR/base.json" "$PAYLOAD_DIR/warm_base.json" >/dev/null; then
    fail "Phase 1: priming request failed"
    stop_server
    echo "Summary: $PASS/$TOTAL passed"
    exit 1
fi

RESUMED_CONTENT=""
if RESUMED_CONTENT=$(send "$PAYLOAD_DIR/diverged.json" "$PAYLOAD_DIR/warm_diverged.json"); then
    pass "Phase 1: diverged prompt completed"
else
    fail "Phase 1: diverged prompt request failed"
fi
stop_server

CHECKPOINT_LINE=$(grep -a "checkpoint of" "$WARM_LOG" | tail -1 || true)
if [ -n "$CHECKPOINT_LINE" ]; then
    pass "Checkpoint branch was taken: $(echo "$CHECKPOINT_LINE" | sed 's/^.*Prompt cache/Prompt cache/')"

    # "N/M tokens reused (checkpoint of K)". A one-token restore would technically satisfy
    # the line's existence without the feature doing anything useful, so require that a
    # real span was skipped.
    REUSED=$(echo "$CHECKPOINT_LINE" | sed -n 's/.*HIT (hybrid): \([0-9]*\)\/.*/\1/p')
    if [ -n "$REUSED" ] && [ "$REUSED" -ge 512 ]; then
        pass "Checkpoint supplied a worthwhile prefix ($REUSED tokens >= 512)"
    else
        fail "Checkpoint supplied only ${REUSED:-unknown} tokens (< 512); the prompt may be too short for a checkpoint before the divergence"
    fi
else
    fail "No checkpoint restore in the log — the diverged prompt fell back to a full prefill. Log tail:"
    tail -20 "$WARM_LOG"
fi

# Phase 2 — same prompt, freshly started server, empty cache: full prefill.
log "Phase 2: fresh server, same diverged prompt (full prefill)"
if ! start_server "$FRESH_LOG"; then
    fail "Phase 2: server did not start"
    echo "Summary: $PASS/$TOTAL passed"
    exit 1
fi

FULL_CONTENT=""
if FULL_CONTENT=$(send "$PAYLOAD_DIR/diverged.json" "$PAYLOAD_DIR/fresh_diverged.json"); then
    pass "Phase 2: full prefill completed"
else
    fail "Phase 2: full prefill request failed"
fi
stop_server

if grep -aq "checkpoint of" "$FRESH_LOG"; then
    fail "Phase 2 server hit a checkpoint despite an empty cache — the comparison is not a full prefill"
else
    pass "Phase 2 ran without any cache hit, as required for the comparison"
fi

# ── The assertion this whole script exists for ───────────────────────
if [ -z "$RESUMED_CONTENT" ] || [ -z "$FULL_CONTENT" ]; then
    fail "Could not compare outputs (one side was empty)"
elif [ "$RESUMED_CONTENT" = "$FULL_CONTENT" ]; then
    pass "Checkpoint resume == full prefill (byte-identical greedy output: $(echo "$RESUMED_CONTENT" | head -c 80))"
else
    fail "Checkpoint resume differs from full prefill"
    echo "    resumed: $(echo "$RESUMED_CONTENT" | head -c 200)"
    echo "    fresh  : $(echo "$FULL_CONTENT" | head -c 200)"
fi

echo
echo "──────────────────────────────────────────"
if [ "$FAIL" -eq 0 ]; then
    echo -e "${GREEN}All $TOTAL hybrid-cache tests passed${NC}"
    exit 0
else
    echo -e "${RED}$FAIL of $TOTAL hybrid-cache tests failed${NC}"
    exit 1
fi
