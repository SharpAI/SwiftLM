---
description: Run the extreme context benchmark suite for SwiftLM profiling
---

# Run Extreme Context Benchmark

This workflow runs the full 12-test profiling matrix (4 configurations × 3 context depths) for a given model and captures performance + memory metrics.

## Prerequisites

// turbo
1. Ensure the SwiftLM binary is built:
```bash
swift build -c release
```

## Run the Benchmark

2. Kill any existing SwiftLM server:
```bash
killall SwiftLM 2>/dev/null; sleep 2
```

// turbo
3. Run the profiling suite with the target model:
```bash
python3 -u scripts/profiling/profile_runner.py \
  --model gemma-4-26b-a4b-it-4bit \
  --contexts "512,40000,100000" \
  --out ./profiling_results_$(hostname -s).md
```

The profiler will:
- Start SwiftLM with each configuration (`Dense/Vanilla`, `SSD Stream`, `TurboQuant`, `SSD + TurboQuant`)
- Send requests at each context depth (512, 40K, 100K tokens)
- Measure TTFT, TPS, Active RAM, and GPU Alloc (via `ioreg AGXAccelerator`)
- Save results to the output markdown file

## Expected Runtime

| Context Size | Approximate Time per Config |
|---|---|
| 512 | ~10 seconds |
| 40,000 | ~40 seconds |
| 100,000 | ~120 seconds |

**Total**: ~12 minutes for the full 12-test matrix (4 configs × 3 contexts)

## Customizing

- **Different model**: Replace `gemma-4-26b-a4b-it-4bit` with any MLX model ID  
- **Different contexts**: Change `--contexts` (comma-separated list of token counts)
- **Output file**: Change `--out` path

## Expert Top-K Tuning for MoE Models

For Mixture of Expert (MoE) models (like `Qwen3.5-122B-A10B-4bit`), you can override the number of dynamically routed experts per token using the `SWIFTLM_TOP_K` environment variable. By default, SwiftLM evaluates the maximum number of experts defined by the model architecture. Reducing this trades marginal quality for extreme memory compression and streaming speed gains.

Provide the parameter securely when running the profiler:
```bash
SWIFTLM_TOP_K=6 python3 -u scripts/profiling/profile_runner.py ...
```

### Reference Pipeline (M1 Ultra 64GB, Qwen3.5-122B-A10B-4bit)

| Configuration | tok/s | vs. Original | Notes |
|---|---|---|---|
| Original `--stream-experts` | 0.58 | baseline | Sequential pread, 1 NVMe queue |
| `SWIFTLM_TOP_K=8` | 4.95 | 8.5× | All 8 experts evaluated (Full quality) |
| `SWIFTLM_TOP_K=6` | 5.20 | 9.0× | Recommended default |
| `SWIFTLM_TOP_K=4` | 5.91 | 10.2× | Best quality/speed tradeoff (Speed mode) |
| `SWIFTLM_TOP_K=2` | 6.52 | 11.2× | Still coherent output (Turbo mode) |

## After the Benchmark

4. Review the generated markdown file and check for any `FAILED / OOM` entries.

5. If contributing results back to the project, append the device section to `profiling_results.md`:
   - Add a `## <Device Name> — <RAM> GB Unified Memory` section
   - Include chip, RAM, macOS version, SwiftLM version
   - Submit a PR with title: `bench: <device-name> results for <model-id>`

## Troubleshooting

- **Server fails to start**: Check that the model exists at `~/.aegis-ai/models/mlx_models/mlx-community/<model-id>` or use a full path
- **Python crashes silently**: Run with `python3 -u` for unbuffered output
- **100K tests OOM on <32 GB**: Use `--contexts "512,40000"` to skip 100K
- **Stale results**: The profiler kills any existing SwiftLM process before each config run
