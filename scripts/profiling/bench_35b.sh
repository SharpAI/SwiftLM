#!/usr/bin/env bash
# SwiftLM Benchmark — Qwen3.6-35B-A3B-4bit
# Tests 4 configs: baseline, SSD, SSD+DFlash, DFlash-only
# Outputs bench_results.json for use with generate_demo_video.py
set -uo pipefail

MAX_TOKENS=512
MODEL="mlx-community/Qwen3.6-35B-A3B-4bit"
DRAFT="z-lab/Qwen3.6-35B-A3B-DFlash"
PORT=5413
RUNS=3
LOG_DIR="/tmp/swiftlm_bench_logs"
RESULTS_FILE="$LOG_DIR/bench_results.json"
mkdir -p "$LOG_DIR"
export LOG_DIR

# Build request JSON with python to avoid bash escaping hell
export MODEL
python3 << 'PYEOF'
import json, os
prompt = "The function $f$ satisfies the functional equation \\[ f(x) + f(y) = f(x + y) - xy - 1 \\] for all real numbers $x$ and $y$. If $f(1) = 1$, then find all integers $n$ such that $f(n) = n$. Enter all such integers, separated by commas. Please reason step by step, and put your final answer within \\boxed{}."
body = {
    "model": os.environ["MODEL"],
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 512,
    "stream": False
}
with open(os.environ["LOG_DIR"] + "/bench_request.json", "w") as f:
    json.dump(body, f)
PYEOF

REQ_FILE="$LOG_DIR/bench_request.json"

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
echo "║          SwiftLM Benchmark — Qwen3.6-35B-A3B-4bit         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Max tokens: $MAX_TOKENS | Runs: $RUNS"
echo "  Results → $RESULTS_FILE"
echo ""

declare -a LABELS=()
declare -a SPEEDS=()
declare -a MEMS=()

test_config() {
    local label="$1"
    shift
    local args=("$@")
    local slug="${label// /_}"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $label"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    stop_server
    echo "  Starting server..."
    (cd .build/release && ./SwiftLM "${args[@]}") >"$LOG_DIR/server_${slug}.log" 2>&1 &
    if ! wait_for_server; then
        LABELS+=("$label")
        SPEEDS+=("FAILED")
        MEMS+=("N/A")
        return
    fi

    # Warmup with a different prompt (avoid polluting prompt cache)
    echo "  🔥 Warmup..."
    curl -sf --max-time 60 http://127.0.0.1:$PORT/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"What is the capital of France? Answer briefly."}],"max_tokens":32,"stream":false}' >/dev/null 2>&1
    sleep 2

    # Benchmark runs — save each raw response for JSON extraction later
    local all_tps=""
    for run in $(seq 1 $RUNS); do
        echo "  🏃 Run $run/$RUNS..."
        local resp
        resp=$(curl -sf --max-time 600 http://127.0.0.1:$PORT/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d @"$REQ_FILE" 2>/dev/null) || resp=""

        if [ -z "$resp" ]; then
            echo "    → FAILED"
            continue
        fi

        # Save raw response JSON for later extraction
        echo "$resp" > "$LOG_DIR/resp_${slug}_run${run}.json"

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
    rss=$(grep "OS_RAM" "$LOG_DIR/server_${slug}.log" | tail -1 | sed 's/.*OS_RAM=\([0-9.]*\).*/\1/')
    echo "  💾 RAM: ${rss} GB"

    LABELS+=("$label")
    SPEEDS+=("$avg")
    MEMS+=("$rss")

    stop_server
    echo ""
}

# ── Run all configs ───────────────────────────────────────────────────────────

test_config "Baseline"     --model "$MODEL" --port $PORT

test_config "SSD Streaming" --model "$MODEL" --port $PORT --stream-experts

test_config "SSD + DFlash" --model "$MODEL" --port $PORT --stream-experts --dflash --draft-model "$DRAFT"

test_config "DFlash only"  --model "$MODEL" --port $PORT --dflash --draft-model "$DRAFT"

# ── Summary table ─────────────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                      RESULTS                              ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Config              Speed (tok/s)      RAM (GB)          ║"
echo "╠══════════════════════════════════════════════════════════════╣"
for i in "${!LABELS[@]}"; do
    printf "║  %-20s %-18s %-18s║\n" "${LABELS[$i]}" "${SPEEDS[$i]}" "${MEMS[$i]}"
done
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Extract rich JSON for demo video ─────────────────────────────────────────

echo "📦 Extracting results to $RESULTS_FILE ..."

python3 << 'PYEOF'
import json, os, re, time, platform

log_dir = os.environ["LOG_DIR"]
results_file = log_dir + "/bench_results.json"

try:
    chip = "Apple M4 Max"  # could call system_profiler, but keep it simple
    ram  = "64 GB"
    machine = f"{chip} · {ram}"
except Exception:
    machine = "Apple Silicon"

results = {
    "timestamp": int(time.time()),
    "model":   "mlx-community/Qwen3.6-35B-A3B-4bit",
    "machine": machine,
    "configs": [],
}

labels = ["Baseline", "SSD Streaming", "SSD + DFlash", "DFlash only"]

for label in labels:
    slug = label.replace(" ", "_")
    server_log_path = f"{log_dir}/server_{slug}.log"

    if not os.path.exists(server_log_path):
        print(f"  ⚠️  No log for {label}, skipping")
        continue

    with open(server_log_path) as f:
        server_log = f.read()

    # Per-run responses
    run_tps    = []
    run_tokens = []
    response_text = ""

    for run in range(1, 4):
        resp_path = f"{log_dir}/resp_{slug}_run{run}.json"
        if not os.path.exists(resp_path):
            continue
        try:
            with open(resp_path) as f:
                resp = json.load(f)
            tps    = resp["timings"]["predicted_per_second"]
            tokens = resp["usage"]["completion_tokens"]
            run_tps.append(round(tps, 1))
            run_tokens.append(tokens)
            # Use first successful run's response text
            if not response_text:
                response_text = resp["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"    ⚠️  Could not parse {resp_path}: {e}")

    if not run_tps:
        print(f"  ⚠️  No successful runs for {label}")
        continue

    avg_tps    = round(sum(run_tps) / len(run_tps), 1)
    avg_tokens = round(sum(run_tokens) / len(run_tokens)) if run_tokens else 512

    # TTFT: first "prefill done" line for the actual bench prompt (n_tokens=104)
    ttft = None
    for line in server_log.split("\n"):
        m = re.search(r"prefill done \| n_tokens=104.*?t=([0-9.]+)s", line)
        if m:
            ttft = float(m.group(1))
            break

    # Prefill tok/s from same line
    prefill_tps = None
    for line in server_log.split("\n"):
        m = re.search(r"prefill done \| n_tokens=104.*?,\s*([0-9.]+)t/s", line)
        if m:
            prefill_tps = float(m.group(1))
            break

    # Peak GPU mem
    gpu_gb = None
    for line in reversed(server_log.split("\n")):
        m = re.search(r"GPU_MEM=([0-9.]+)GB", line)
        if m:
            gpu_gb = float(m.group(1))
            break

    # Peak OS RAM
    ram_gb = None
    for line in reversed(server_log.split("\n")):
        m = re.search(r"OS_RAM=([0-9.]+)GB", line)
        if m:
            ram_gb = float(m.group(1))
            break

    # DFlash acceptance (last occurrence = most recent run)
    dflash_accept = None
    for line in reversed(server_log.split("\n")):
        m = re.search(r"DFlash summary.*?acceptance=([0-9.]+)%", line)
        if m:
            dflash_accept = round(float(m.group(1)), 1)
            break

    # chars/token from real response
    chars_per_token = (
        round(len(response_text) / avg_tokens, 3)
        if avg_tokens > 0 and response_text
        else 3.5
    )

    entry = {
        "label":           label,
        "speed":           avg_tps,
        "runs":            run_tps,
        "ram_gb":          ram_gb,
        "gpu_gb":          gpu_gb,
        "ttft_s":          ttft,
        "prefill_tps":     prefill_tps,
        "tokens":          avg_tokens,
        "dflash_accept":   dflash_accept,
        "chars_per_token": chars_per_token,
        "response_text":   response_text,
    }
    results["configs"].append(entry)
    print(f"  ✅ {label:<20}  {avg_tps:.1f} tok/s  RAM {ram_gb}G  "
          f"TTFT {ttft}s  chars/tok {chars_per_token:.2f}")

with open(results_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n  📄 Saved: {results_file}")
print(f"  Generate video: python generate_demo_video.py --results {results_file}")
PYEOF
