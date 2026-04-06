#!/bin/bash

# Ensure we execute from the project root
cd "$(dirname "$0")"

echo "=============================================="
echo "    Aegis-AI MLX Profiling Benchmark Suite    "
echo "=============================================="
echo ""

echo "Select Benchmark Suite:"
echo "1) Test 1: Automated Context & Memory Profile (TPS & RAM matrix)"
echo "2) Test 2: Prompt Cache & Sliding Window Regression Test"
echo "3) Quit"
read -p "Option (1-3): " suite_opt

if [ "$suite_opt" == "3" ] || [ -z "$suite_opt" ]; then
    echo "Exiting."
    exit 0
fi

# Quick sanity check
if [ -f ".build/arm64-apple-macosx/release/SwiftLM" ]; then
    BIN=".build/arm64-apple-macosx/release/SwiftLM"
elif [ -f ".build/release/SwiftLM" ]; then
    BIN=".build/release/SwiftLM"
else
    echo "⚠️  SwiftLM release binary not found! Please compile the project by running ./build.sh first."
    exit 1
fi

if [ "$suite_opt" == "2" ]; then
    echo ""
    echo "=> Starting Prompt Cache Regression Test"
    echo "Generating /tmp/big_prompt.json (approx 5K tokens)..."
    python3 -c 'import json; open("/tmp/big_prompt.json", "w").write(json.dumps({"messages": [{"role": "user", "content": "apple "*4500}], "max_tokens": 30}))'
    
    echo "Starting Server in background..."
    killall SwiftLM 2>/dev/null
    mkdir -p tmp
    $BIN --model mlx-community/gemma-4-26b-a4b-it-4bit --port 5431 --turbo-kv --stream-experts --ctx-size 16384 > ./tmp/regression_server.log 2>&1 &
    SERVER_PID=$!
    
    echo "Waiting for server to be ready on port 5431 (this may take a minute if downloading)..."
    for i in {1..300}; do
        if curl -s http://127.0.0.1:5431/health > /dev/null; then break; fi
        sleep 1
    done
    
    echo ""
    echo "Server is up! Running 4-request sliding window validation..."
    
    echo "=== Req 1 (Big 5537t) ===" && curl -sS --max-time 120 http://127.0.0.1:5431/v1/chat/completions -H "Content-Type: application/json" -d @/tmp/big_prompt.json 2>&1 | python3 -c "import sys,json;d=json.load(sys.stdin);print('OK:',d['choices'][0]['message']['content'])" && \
    echo "=== Req 2 (Short 18t) ===" && curl -sS --max-time 60 http://127.0.0.1:5431/v1/chat/completions -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"What is today?"}],"max_tokens":30}' 2>&1 | python3 -c "import sys,json;d=json.load(sys.stdin);print('OK:',d['choices'][0]['message']['content'])" && \
    echo "=== Req 3 (Big 5537t) ===" && curl -sS --max-time 120 http://127.0.0.1:5431/v1/chat/completions -H "Content-Type: application/json" -d @/tmp/big_prompt.json 2>&1 | python3 -c "import sys,json;d=json.load(sys.stdin);print('OK:',d['choices'][0]['message']['content'])" && \
    echo "=== Req 4 (Big Full Cache Hit) ===" && curl -sS --max-time 120 http://127.0.0.1:5431/v1/chat/completions -H "Content-Type: application/json" -d @/tmp/big_prompt.json 2>&1 | python3 -c "import sys,json;d=json.load(sys.stdin);print('OK:',d['choices'][0]['message']['content'])" && \
    echo "=== ALL 4 PASSED ==="
    
    echo ""
    echo "Cleaning up..."
    killall SwiftLM
    wait $SERVER_PID 2>/dev/null
    exit 0
fi

# Fallback to Test 1 for anything else
echo ""
PS3="Select a model to profile (1-7): "
options=(
    "gemma-4-26b-a4b-it-4bit"
    "gemma-4-2b-a4b-it-4bit"
    "Qwen3.5-7B-Instruct-4bit"
    "Qwen3.5-14B-Instruct-4bit"
    "phi-4-mlx-4bit"
    "Custom (Enter your own Hub ID)"
    "Quit"
)

select opt in "${options[@]}"
do
    case $opt in
        "Custom (Enter your own Hub ID)")
            read -p "Enter HuggingFace ID (e.g., Llama-3.2-3B-Instruct-4bit): " custom_model
            MODEL=$custom_model
            break
            ;;
        "Quit")
            echo "Exiting."
            exit 0
            ;;
        *) 
            if [[ -n "$opt" ]]; then
                MODEL=$opt
                break
            else
                echo "Invalid option $REPLY"
            fi
            ;;
    esac
done

echo ""
read -p "Enter context lengths to test [default: 512,40000,100000]: " CONTEXTS
CONTEXTS=${CONTEXTS:-"512,40000,100000"}

echo ""
echo "=> Starting benchmark for $MODEL with contexts: $CONTEXTS"
echo ""

python3 -u scripts/profiling/profile_runner.py \
  --model "$MODEL" \
  --contexts "$CONTEXTS" \
  --out "./profiling_results_$(hostname -s).md"

echo ""
echo "✅ Benchmark finished! Results saved to ./profiling_results_$(hostname -s).md"
