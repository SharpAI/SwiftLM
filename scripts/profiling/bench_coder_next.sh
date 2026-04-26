#!/usr/bin/env bash
# SwiftLM Benchmark — Qwen3-Coder-Next-4bit
# Tests 4 configs: baseline, SSD, SSD+DFlash, DFlash-only
set -uo pipefail

MAX_TOKENS=512
MODEL="mlx-community/Qwen3-Coder-Next-4bit"
DRAFT="z-lab/Qwen3-Coder-Next-DFlash"
PORT=5413
RUNS=3
LOG_DIR="/tmp/swiftlm_bench_logs"
mkdir -p "$LOG_DIR"
export LOG_DIR

# Build request JSON with python to avoid bash escaping
export MODEL
python3 << 'PYEOF'
import json, os
prompt = "Write a Python function that computes the nth Fibonacci number using memoization. Include type hints and a docstring. Add a main block that prints the first 20 Fibonacci numbers."
body = {
    "model": os.environ["MODEL"],
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 512,
    "stream": False
}
with open(os.environ["LOG_DIR"] + "/bench_coder_next.json", "w") as f:
    json.dump(body, f)
PYEOF

REQ_FILE="$LOG_DIR/bench_coder_next.json"

# ── Helpers ──────────────────────────────────────────────────────────────────

wait_for_server() {
    for i in $(seq 1 3600); do
        if curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1; then
            echo "  ✅ Ready (${i}s)"
            return 0
        fi
        sleep 1
    done
    echo "  ❌ Failed"
    return 1
}

stop_server() {
    pkill -f "SwiftLM" 2>/dev/null || true
    sleep 4
    pkill -9 -f "SwiftLM" 2>/dev/null || true
    sleep 2
}

# ── Main ─────────────────────────────────────────────────────────────────────

cd "$(git rev-parse --show-toplevel)"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║      SwiftLM Benchmark — Qwen3-Coder-Next-4bit             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Max tokens: $MAX_TOKENS | Runs: $RUNS"
echo ""

declare -a LABELS=()
declare -a SPEEDS=()
declare -a MEMS=()

test_config() {
    local label="$1"
    shift
    local args=("$@")

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $label"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    stop_server
    echo "  Starting server..."
    (cd .build/release && ./SwiftLM "${args[@]}") >"$LOG_DIR/cn_${label// /_}.log" 2>&1 &
    if ! wait_for_server; then
        LABELS+=("$label")
        SPEEDS+=("FAILED")
        MEMS+=("N/A")
        echo ""
        return
    fi

    # Warmup with different prompt
    echo "  🔥 Warmup..."
    curl -sf --max-time 120 http://127.0.0.1:$PORT/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"Say hi in one word."}],"max_tokens":16,"stream":false}' >/dev/null 2>&1
    sleep 2

    # Benchmark runs
    local all_tps=""
    for run in $(seq 1 $RUNS); do
        echo "  🏃 Run $run/$RUNS..."
        local resp
        resp=$(curl -sf --max-time 600 http://127.0.0.1:$PORT/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d @"$REQ_FILE" 2>/dev/null) || resp=""

        if [ -z "$resp" ]; then
            echo "    → FAILED (empty response)"
            continue
        fi

        local tps tokens
        tps=$(echo "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f\"{d['timings']['predicted_per_second']:.1f}\")" 2>/dev/null) || tps="0.0"
        tokens=$(echo "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['usage']['completion_tokens'])" 2>/dev/null) || tokens="0"
        echo "    → ${tps} tok/s (${tokens} tokens)"

        if [ -n "$all_tps" ]; then
            all_tps="${all_tps}, ${tps}"
        else
            all_tps="${tps}"
        fi
    done

    # Average
    local avg="0.0"
    if [ -n "$all_tps" ]; then
        avg=$(python3 -c "vals=[${all_tps}]; print(f'{sum(vals)/len(vals):.1f}')" 2>/dev/null) || avg="0.0"
    fi
    echo "  📊 Avg: ${avg} tok/s"

    # Peak RAM from server log
    local rss
    rss=$(grep "OS_RAM" "$LOG_DIR/cn_${label// /_}.log" | tail -1 | sed 's/.*OS_RAM=\([0-9.]*\).*/\1/')
    echo "  💾 RAM: ${rss} GB"

    LABELS+=("$label")
    SPEEDS+=("$avg")
    MEMS+=("$rss")

    stop_server
    echo ""
}

# ── Run all configs ──────────────────────────────────────────────────────────

test_config "Baseline" --model "$MODEL" --port $PORT

test_config "SSD Streaming" --model "$MODEL" --port $PORT --stream-experts

test_config "SSD + DFlash" --model "$MODEL" --port $PORT --stream-experts --dflash --draft-model "$DRAFT"

test_config "DFlash only" --model "$MODEL" --port $PORT --dflash --draft-model "$DRAFT"

# ── Summary ──────────────────────────────────────────────────────────────────

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                      RESULTS                              ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Config              Speed (tok/s)      RAM (GB)          ║"
echo "╠══════════════════════════════════════════════════════════════╣"
for i in "${!LABELS[@]}"; do
    printf "║  %-20s %-18s %-18s║\n" "${LABELS[$i]}" "${SPEEDS[$i]}" "${MEMS[$i]}"
done
echo "╚══════════════════════════════════════════════════════════════╝"
