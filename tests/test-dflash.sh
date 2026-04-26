#!/bin/bash
# test-speculative.sh — Speculative decoding E2E verification
#
# Uses a small draft model (Qwen3.5-0.8B) to accelerate a larger main model
# (Qwen3.5-4B) via speculative decoding. Verifies:
#   1. Dual-model loading (draft + main)
#   2. Speculative decoding path activation
#   3. Correct token generation
#   4. Server stability under dual-model memory pressure
#
# Usage:
#   ./tests/test-speculative.sh [binary_path] [port]
#
# Requirements:
#   - ~4 GB RAM (0.8B draft ~1 GB + 4B main ~3 GB)
#   - macos-15 (7 GB) on GitHub Actions is sufficient
#   - curl, jq

set -euo pipefail

BINARY="${1:-.build/release/SwiftLM}"
PORT="${2:-15414}"
HOST="127.0.0.1"
MAIN_MODEL="${MAIN_MODEL:-mlx-community/Qwen3.5-4B-4bit}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3.5-4B-DFlash}"
NUM_DRAFT_TOKENS=16
URL="http://${HOST}:${PORT}"
PASS=0
FAIL=0
TOTAL=0
LOG_FILE="/tmp/SwiftLM-test-dflash.log"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${YELLOW}[dflash-test]${NC} $*"; }
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
    echo "Run 'swift build -c release' first."
    exit 1
fi

if ! command -v jq &>/dev/null; then
    echo "Error: jq is required. Install with: brew install jq"
    exit 1
fi

# ── Memory check ────────────────────────────────────────────────────
TOTAL_RAM_GB=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f", $1 / 1073741824}')
log "System RAM: ${TOTAL_RAM_GB} GB"

# On low-RAM machines (< 12 GB), the combined main + draft model weights
# (~6 GB) exceed available memory after OS reservation.  Without SSD
# streaming, all weights must be GPU-resident or swapped via macOS VM,
# which causes Metal command buffers to exceed Apple's 5-second GPU
# Watchdog timeout → Abort trap: 6.
#
# Fix: enable --stream-experts on low-RAM machines.  This uses mmap-based
# weight loading (pread from SSD via the OS page cache) so the GPU never
# stalls waiting for swap.  Draft tokens are auto-capped to 1 server-side
# to minimise SSD I/O fan-out during the verify pass.
EXTRA_FLAGS=""
if [ "$TOTAL_RAM_GB" -lt 12 ] 2>/dev/null; then
    log "⚠️  ${TOTAL_RAM_GB} GB RAM: enabling --stream-experts for SSD-backed weight paging"
    log "   Combined model weights (~6 GB) exceed available RAM. SSD streaming prevents"
    log "   Metal GPU Watchdog timeouts during DFlash verify passes."
    EXTRA_FLAGS="--stream-experts"
    NUM_DRAFT_TOKENS=1  # auto-capped server-side too, but be explicit
fi

# ══════════════════════════════════════════════════════════════════════
echo -e "\n${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  SwiftLM DFlash Speculative Decoding E2E Test           ║${NC}"
echo -e "${CYAN}║  Draft: Qwen3.5-4B-DFlash → Main: Qwen3.5-4B-4bit       ║${NC}"
echo -e "${CYAN}║  Draft tokens per round: ${NUM_DRAFT_TOKENS}                             ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}\n"

# ── Start server with dual models ───────────────────────────────────
log "Starting server with DFlash speculative decoding..."
log "  Main model:  $MAIN_MODEL"
log "  Draft model: $DRAFT_MODEL"
log "  Draft tokens per round: $NUM_DRAFT_TOKENS"
if [ -n "$EXTRA_FLAGS" ]; then
    log "  Extra flags: $EXTRA_FLAGS"
fi

"$BINARY" --model "$MAIN_MODEL" --port "$PORT" --host "$HOST" \
    --draft-model "$DRAFT_MODEL" \
    --num-draft-tokens "$NUM_DRAFT_TOKENS" \
    --dflash $EXTRA_FLAGS \
    > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

# Wait for server to be ready (both models need to download + load)
log "Waiting for server to load both models (this may take a while on first run)..."
MAX_WAIT=900  # 15 minutes for two model downloads
for i in $(seq 1 "$MAX_WAIT"); do
    if curl -sf "$URL/health" >/dev/null 2>&1; then
        log "Server ready after ${i}s"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Error: Server process died. Server Log:"
        cat "$LOG_FILE"
        exit 1
    fi
    # Print progress every 30 seconds
    if [ $((i % 30)) -eq 0 ]; then
        log "  Still waiting... (${i}s elapsed)"
    fi
    sleep 1
done

if ! curl -sf "$URL/health" >/dev/null 2>&1; then
    echo "Error: Server did not become ready in ${MAX_WAIT}s"
    echo "Server Log:"
    cat "$LOG_FILE"
    exit 1
fi

# ── Test 1: Verify server loaded both models ────────────────────────
log "Test 1: Verify dual-model loading"

# Check server log for draft model loading confirmation
if grep -q "Draft model loaded successfully" "$LOG_FILE"; then
    pass "Draft model loaded successfully"
else
    fail "Draft model loading not confirmed in server logs"
fi

if grep -q "speculative decoding" "$LOG_FILE"; then
    pass "Speculative decoding mode detected in server logs"
else
    fail "Speculative decoding not mentioned in server logs"
fi

# ── Test 2: Health endpoint works with dual models ──────────────────
log "Test 2: Health endpoint"

HEALTH=$(curl -sf "$URL/health")
if echo "$HEALTH" | jq -e '.status == "ok"' >/dev/null 2>&1; then
    pass "Health endpoint returns status=ok"
else
    fail "Health endpoint: $HEALTH"
fi

# ── Test 3: Streaming speculative generation ────────────────────────
log "Test 3: Streaming speculative generation"

STREAM_OUTPUT=$(curl -sf -N --max-time 120 -X POST "$URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MAIN_MODEL\",\"stream\":true,\"max_tokens\":30,\"messages\":[{\"role\":\"user\",\"content\":\"Name three fruits.\"}]}" \
    2>/dev/null || true)

if echo "$STREAM_OUTPUT" | grep -q "data: \[DONE\]"; then
    pass "Streaming speculative: received [DONE] sentinel"
else
    fail "Streaming speculative: missing [DONE] sentinel"
fi

CHUNK_COUNT=$(echo "$STREAM_OUTPUT" | grep -c "^data: {" || true)
if [ "$CHUNK_COUNT" -gt 0 ]; then
    pass "Streaming speculative: received $CHUNK_COUNT data chunks"
else
    fail "Streaming speculative: no data chunks received"
fi

# Check server log for speculative decoding activation
if grep -q "Using speculative decoding" "$LOG_FILE"; then
    pass "Speculative decoding path activated during generation"
else
    fail "Speculative decoding path not activated (missing log line)"
fi

# ── Test 5: Multiple sequential requests (stability) ────────────────
log "Test 5: Sequential request stability (3 requests)"

SEQ_PASS=true
for i in 1 2 3; do
    SEQ_RESP=$(curl -sf --max-time 120 -X POST "$URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MAIN_MODEL\",\"max_tokens\":10,\"messages\":[{\"role\":\"user\",\"content\":\"Say the number $i.\"}]}" 2>/dev/null || echo "")

    SEQ_CONTENT=$(echo "$SEQ_RESP" | jq -r '.choices[0].message.content // empty' 2>/dev/null || echo "")

    if [ -z "$SEQ_CONTENT" ]; then
        SEQ_PASS=false
        fail "Sequential request $i: empty response"
        break
    fi
done

if [ "$SEQ_PASS" = true ]; then
    pass "Sequential stability: 3/3 speculative requests completed successfully"
fi

# ── Test 6: Memory stability check ─────────────────────────────────
log "Test 6: Memory stability"

HEALTH_FINAL=$(curl -sf "$URL/health")
MEM_ACTIVE=$(echo "$HEALTH_FINAL" | jq -r '.memory.active_mb // 0')
MEM_PEAK=$(echo "$HEALTH_FINAL" | jq -r '.memory.peak_mb // 0')

if [ "$MEM_ACTIVE" -gt 0 ] 2>/dev/null; then
    pass "Memory: active=${MEM_ACTIVE} MB, peak=${MEM_PEAK} MB"
else
    fail "Memory: could not read memory stats"
fi

# Verify server is still responsive after all tests
if curl -sf "$URL/health" >/dev/null 2>&1; then
    pass "Server still responsive after all speculative decoding tests"
else
    fail "Server became unresponsive"
fi

# ── Results ──────────────────────────────────────────────────────────
echo ""
log "═══════════════════════════════════════"
log "Speculative Decoding Test Results"
log "  Draft:  $DRAFT_MODEL"
log "  Main:   $MAIN_MODEL"
log "  Tokens/round: $NUM_DRAFT_TOKENS"
log "  Results: ${PASS} passed, ${FAIL} failed, ${TOTAL} total"
log "═══════════════════════════════════════"

if [ "$FAIL" -gt 0 ]; then
    echo ""
    log "Server completely failed. Full Log:"
    cat "$LOG_FILE"
    exit 1
fi

echo ""
log "Server log tail (last 50 lines):"
tail -50 "$LOG_FILE"
exit 0
