// Copyright 2026 SwiftLM Contributors
// MIT License — see LICENSE file
//
// Micro-benchmark for DFlash Metal kernels.
// Run under Metal System Trace:
//   xcrun xctrace record --template "Metal System Trace" \
//     --launch .build/release/DFlashKernelBench -- [flags]
//
// Flags:
//   --iterations N   kernel calls per benchmark (default: 200)
//   --warmup N       warmup calls before timing (default: 20)
//   --kernels list   comma-separated subset: tape,gdelta,sdpa,variants,ops (default: tape,gdelta,sdpa)
//   --long-ctx       include long-context SDPA sizes (nKV 16k, 32k)

import Foundation
import MLX
import MLXNN
import DFlash
import os.log

// MARK: - Signpost log

private let log = OSLog(subsystem: "com.swiftlm.dflash", category: "kernels")

// MARK: - Helpers

/// Fill an array with uniform random values in bf16.
private func rand(_ shape: [Int], dtype: DType = .bfloat16) -> MLXArray {
    uniform(low: -0.1, high: 0.1, shape, dtype: dtype)
}

/// Wall-clock time in seconds for one synchronised MLX eval.
private func timeEval(_ body: () -> MLXArray) -> Double {
    let arr = body()
    let t0 = clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW)
    MLX.eval(arr)
    let t1 = clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW)
    return Double(t1 - t0) * 1e-9
}

/// Run `iterations` timed calls, return (median_s, min_s, max_s).
private func measure(label: String, iterations: Int, body: () -> MLXArray) -> (median: Double, min: Double, max: Double) {
    var samples = [Double]()
    samples.reserveCapacity(iterations)

    let signpostID = OSSignpostID(log: log)
    for _ in 0 ..< iterations {
        os_signpost(.begin, log: log, name: "kernel", signpostID: signpostID, "%{public}s", label)
        let t = timeEval(body)
        os_signpost(.end, log: log, name: "kernel", signpostID: signpostID, "%{public}s", label)
        samples.append(t)
    }

    samples.sort()
    let med = samples[samples.count / 2]
    return (med, samples.first!, samples.last!)
}

private func printResult(label: String, r: (median: Double, min: Double, max: Double), extraInfo: String = "") {
    let medUs = r.median * 1e6
    let minUs = r.min * 1e6
    let maxUs = r.max * 1e6
    let extra = extraInfo.isEmpty ? "" : "  \(extraInfo)"
    let pad = label.padding(toLength: 42, withPad: " ", startingAt: 0)
    print(String(format: "  %@  med %7.1f µs  min %7.1f µs  max %7.1f µs%@",
                 pad, medUs, minUs, maxUs, extra))
}

/// Theoretical memory bandwidth figure (GB/s) for a kernel that touches `bytes` bytes.
private func bwStr(bytes: Int, seconds: Double) -> String {
    let gb = Double(bytes) / 1e9 / seconds
    return String(format: "%.1f GB/s", gb)
}

// MARK: - Argument parsing

struct Args {
    var iterations = 200
    var warmup = 20
    var kernels: Set<String> = ["tape", "gdelta", "sdpa"]
    var longCtx = false

    init() {
        let argv = CommandLine.arguments
        func intArg(_ flag: String, default d: Int) -> Int {
            guard let i = argv.firstIndex(of: flag), i + 1 < argv.count else { return d }
            return Int(argv[i + 1]) ?? d
        }
        iterations = intArg("--iterations", default: 200)
        warmup     = intArg("--warmup",     default: 20)
        if let i = argv.firstIndex(of: "--kernels"), i + 1 < argv.count {
            kernels = Set(argv[i + 1].split(separator: ",").map(String.init))
        }
        longCtx = argv.contains("--long-ctx")
    }
}

// MARK: - Tape Replay benchmarks

/// Shapes matching Qwen3.5 GDN layers:
///   Hk=8, Hv=16, Dk=128, Dv=128, T=blockSize=16, B=1
private func benchTapeReplay(args: Args) {
    print("\n── Tape Replay ──────────────────────────────────────────────────────────")

    let B = 1; let T = 16; let Hk = 8; let Hv = 16; let Dk = 128; let Dv = 128

    let tape     = rand([B, T, Hv, Dv])
    let k        = rand([B, T, Hk, Dk])
    let gScalar  = rand([B, T, Hv])          // scalar gate
    let gVec     = rand([B, T, Hv, Dk])      // vectorised gate
    let state    = rand([B, Hv, Dv, Dk])
    let mask     = (uniform(low: 0, high: 1, [B, T]) .>= MLXArray(0.5)).asType(DType.bfloat16)

    // warm up
    for _ in 0 ..< args.warmup {
        MLX.eval(DFlashKernels.tapeReplayKernel(tape: tape, k: k, g: gScalar, state: state))
    }

    let stateBytes = B * Hv * Dv * Dk * 2  // bfloat16 = 2 bytes

    let r1 = measure(label: "tape_replay scalar-g", iterations: args.iterations) {
        DFlashKernels.tapeReplayKernel(tape: tape, k: k, g: gScalar, state: state)
    }
    printResult(label: "scalar-g, no mask", r: r1, extraInfo: bwStr(bytes: stateBytes * 2, seconds: r1.median))

    let r2 = measure(label: "tape_replay scalar-g masked", iterations: args.iterations) {
        DFlashKernels.tapeReplayKernel(tape: tape, k: k, g: gScalar, state: state, mask: mask)
    }
    printResult(label: "scalar-g, mask", r: r2, extraInfo: bwStr(bytes: stateBytes * 2, seconds: r2.median))

    let r3 = measure(label: "tape_replay vec-g", iterations: args.iterations) {
        DFlashKernels.tapeReplayKernel(tape: tape, k: k, g: gVec, state: state)
    }
    printResult(label: "vec-g, no mask", r: r3, extraInfo: bwStr(bytes: stateBytes * 2, seconds: r3.median))

    let r4 = measure(label: "tape_replay vec-g masked", iterations: args.iterations) {
        DFlashKernels.tapeReplayKernel(tape: tape, k: k, g: gVec, state: state, mask: mask)
    }
    printResult(label: "vec-g, mask", r: r4, extraInfo: bwStr(bytes: stateBytes * 2, seconds: r4.median))
}

// MARK: - GatedDelta with Tape benchmarks

private func benchGatedDelta(args: Args) {
    print("\n── GatedDelta + Tape ────────────────────────────────────────────────────")

    let B = 1; let T = 16; let Hk = 8; let Hv = 16; let Dk = 128; let Dv = 128

    let q       = rand([B, T, Hk, Dk])
    let k       = rand([B, T, Hk, Dk])
    let v       = rand([B, T, Hv, Dv])
    let gScalar = rand([B, T, Hv])
    let gVec    = rand([B, T, Hv, Dk])
    let beta    = rand([B, T, Hv])
    let state   = rand([B, Hv, Dv, Dk])
    let mask    = (uniform(low: 0, high: 1, [B, T]) .>= MLXArray(0.5)).asType(DType.bfloat16)

    for _ in 0 ..< args.warmup {
        let (y, s, t) = DFlashKernels.gatedDeltaKernelWithTape(q: q, k: k, v: v, g: gScalar, beta: beta, state: state)
        MLX.eval(y, s, t)
    }

    // bytes read+written per call (approximate): q+k+v+state_in+state_out+tape_out
    let callBytes = (B*T*Hk*Dk + B*T*Hk*Dk + B*T*Hv*Dv) * 2   // q,k,v inputs
                  + B*Hv*Dv*Dk * 2 * 2                           // state in+out
                  + B*T*Hv*Dv * 4                                 // tape (f32)

    let r1 = measure(label: "gdelta scalar-g", iterations: args.iterations) {
        let (y, _, _) = DFlashKernels.gatedDeltaKernelWithTape(q: q, k: k, v: v, g: gScalar, beta: beta, state: state)
        return y
    }
    printResult(label: "scalar-g, no mask", r: r1, extraInfo: bwStr(bytes: callBytes, seconds: r1.median))

    let r2 = measure(label: "gdelta scalar-g masked", iterations: args.iterations) {
        let (y, _, _) = DFlashKernels.gatedDeltaKernelWithTape(q: q, k: k, v: v, g: gScalar, beta: beta, state: state, mask: mask)
        return y
    }
    printResult(label: "scalar-g, mask", r: r2, extraInfo: bwStr(bytes: callBytes, seconds: r2.median))

    let r3 = measure(label: "gdelta vec-g", iterations: args.iterations) {
        let (y, _, _) = DFlashKernels.gatedDeltaKernelWithTape(q: q, k: k, v: v, g: gVec, beta: beta, state: state)
        return y
    }
    printResult(label: "vec-g, no mask", r: r3, extraInfo: bwStr(bytes: callBytes, seconds: r3.median))

    let r4 = measure(label: "gdelta vec-g masked", iterations: args.iterations) {
        let (y, _, _) = DFlashKernels.gatedDeltaKernelWithTape(q: q, k: k, v: v, g: gVec, beta: beta, state: state, mask: mask)
        return y
    }
    printResult(label: "vec-g, mask", r: r4, extraInfo: bwStr(bytes: callBytes, seconds: r4.median))
}

// MARK: - Batched SDPA 2-pass benchmarks

private func benchSDPA(args: Args) {
    print("\n── Batched SDPA 2-Pass ──────────────────────────────────────────────────")

    // Shapes: B=1, Hq=32, Hk=8 (GQA 4x), qLen=16, D=128
    // Vary nKV to cover prefill (2k), mid (8k), long (32k)
    let B = 1; let Hq = 32; let Hk = 8; let qLen = 16; let D = 128
    let scale = Float(1.0 / sqrt(Float(D)))

    var kvSizes = [512, 2048, 8192]
    if args.longCtx { kvSizes += [16384, 32768] }

    let q = rand([B, Hq, qLen, D])

    for nKV in kvSizes {
        let k = rand([B, Hk, nKV, D])
        let v = rand([B, Hk, nKV, D])

        // warm up
        for _ in 0 ..< args.warmup {
            if let out = DFlashKernels.batchedSDPA2Pass(queries: q, keys: k, values: v, scale: scale) {
                MLX.eval(out)
            }
        }

        // bytes: read Q + K + V, write output
        let readBytes  = (B*Hq*qLen*D + B*Hk*nKV*D + B*Hk*nKV*D) * 2
        let writeBytes = B*Hq*qLen*D * 2
        let totalBytes = readBytes + writeBytes

        let r = measure(label: "sdpa nKV=\(nKV)", iterations: args.iterations) {
            DFlashKernels.batchedSDPA2Pass(queries: q, keys: k, values: v, scale: scale) ?? q
        }
        printResult(label: "nKV=\(nKV)", r: r, extraInfo: bwStr(bytes: totalBytes, seconds: r.median))

        // Also time the MLXFast fallback for comparison
        let rf = measure(label: "sdpa_fallback nKV=\(nKV)", iterations: args.iterations) {
            DFlashKernels.sdpaFallback(queries: q, keys: k, values: v, scale: scale)
        }
        printResult(label: "nKV=\(nKV) [MLXFast fallback]", r: rf, extraInfo: bwStr(bytes: totalBytes, seconds: rf.median))

        let speedup = rf.median / r.median
        print(String(format: "    → custom vs fallback: %.2fx", speedup))
    }
}

// MARK: - Kernel Variant Comparison (branching vs branchless Metal source)

private func benchKernelVariants(args: Args) {
    print("\n── Kernel Variants: Branching vs Branchless ─────────────────────────────")

    let B = 1; let T = 16; let Hk = 8; let Hv = 16; let Dk = 128; let Dv = 128
    let tape  = rand([B, T, Hv, Dv])
    let k     = rand([B, T, Hk, Dk])
    let g     = rand([B, T, Hv])
    let state = rand([B, Hv, Dv, Dk])
    let mask  = (uniform(low: 0, high: 1, [B, T]) .>= MLXArray(0.5)).asType(DType.bfloat16)
    let q     = rand([B, T, Hk, Dk])
    let v     = rand([B, T, Hv, Dv])
    let beta  = rand([B, T, Hv])

    let inputType = DType.bfloat16

    // ── Tape Replay ──────────────────────────────────────────────────────────

    // Current: if-guard wraps entire inner loop body; two-line state update
    let tapeBranchingSrc = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;
        auto tape_ = tape + b_idx * T * Hv * Dv + hv_idx * Dv;
        auto k_    = k    + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;
        auto i_state = state_in  + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;
        auto g_ = g + b_idx * T * Hv;
        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }
        for (int t = 0; t < T; ++t) {
          if (mask[b_idx * T + t]) {
            auto delta = static_cast<float>(tape_[dv_idx]);
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * g_[hv_idx];
              state[i] = state[i] + k_[s_idx] * delta;
            }
            for (int i = 0; i < n_per_t; ++i) {
              state[i] = static_cast<float>(static_cast<InT>(state[i]));
            }
          }
          tape_ += Hv * Dv;
          k_    += Hk * Dk;
          g_    += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<InT>(state[i]);
        }
    """

    // Corrected: metal::select — no decay when masked, no branch, correct semantics
    let tapeSelectSrc = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;
        auto tape_ = tape + b_idx * T * Hv * Dv + hv_idx * Dv;
        auto k_    = k    + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;
        auto i_state = state_in  + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;
        auto g_ = g + b_idx * T * Hv;
        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i)
            state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
        for (int t = 0; t < T; ++t) {
            bool do_step = static_cast<float>(mask[b_idx * T + t]) > 0.5f;
            float delta = static_cast<float>(tape_[dv_idx]);
            for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                float next = state[i] * g_[hv_idx] + k_[s_idx] * delta;
                next = static_cast<float>(static_cast<InT>(next));
                state[i] = metal::select(state[i], next, do_step);
            }
            tape_ += Hv * Dv;
            k_    += Hk * Dk;
            g_    += Hv;
        }
        for (int i = 0; i < n_per_t; ++i)
            o_state[n_per_t * dk_idx + i] = static_cast<InT>(state[i]);
    """

    let tapeKernelBranching = MLXFast.metalKernel(
        name: "bench_tape_branching_mask",
        inputNames: ["tape", "k", "g", "state_in", "T", "mask"],
        outputNames: ["state_out"],
        source: tapeBranchingSrc
    )

    let tapeKernelSelect = MLXFast.metalKernel(
        name: "bench_tape_select_mask",
        inputNames: ["tape", "k", "g", "state_in", "T", "mask"],
        outputNames: ["state_out"],
        source: tapeSelectSrc
    )

    let steps = T
    func runTape(_ kernel: MLXFast.MLXFastKernel) -> MLXArray {
        kernel(
            [tape, k, g, state, MLXArray(steps), mask],
            template: [("InT", inputType), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
            grid: (32, Dv, B * Hv), threadGroup: (32, 4, 1),
            outputShapes: [state.shape], outputDTypes: [inputType]
        )[0]
    }

    for _ in 0 ..< args.warmup {
        MLX.eval(runTape(tapeKernelBranching))
        MLX.eval(runTape(tapeKernelSelect))
    }

    let stateBytes = B * Hv * Dv * Dk * 2
    let r1 = measure(label: "bench_tape_branching_mask", iterations: args.iterations) {
        runTape(tapeKernelBranching)
    }
    printResult(label: "tape branching (scalar-g, masked)", r: r1,
                extraInfo: bwStr(bytes: stateBytes * 2, seconds: r1.median))

    let r2 = measure(label: "bench_tape_select_mask", iterations: args.iterations) {
        runTape(tapeKernelSelect)
    }
    printResult(label: "tape select    (scalar-g, masked)", r: r2,
                extraInfo: bwStr(bytes: stateBytes * 2, seconds: r2.median))
    print(String(format: "    → select vs branching: %.2fx", r1.median / r2.median))

    // ── GatedDelta + Tape ─────────────────────────────────────────────────────

    // Current: if-guard, separate decay and accumulate assignments
    let gdeltaBranchingSrc = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;
        auto q_    = q    + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_    = k    + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto v_    = v    + b_idx * T * Hv * Dv + hv_idx * Dv;
        y         += b_idx * T * Hv * Dv + hv_idx * Dv;
        auto tape_ = innovation_tape + b_idx * T * Hv * Dv + hv_idx * Dv;
        auto g_    = g    + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;
        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;
        auto i_state = state_in  + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;
        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }
        for (int t = 0; t < T; ++t) {
          float delta = 0.0f;
          if (mask[b_idx * T + t]) {
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * g_[hv_idx];
              kv_mem += state[i] * k_[s_idx];
            }
            kv_mem = simd_sum(kv_mem);
            delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];
            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] + k_[s_idx] * delta;
              out += state[i] * q_[s_idx];
            }
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {
              y[dv_idx] = static_cast<InT>(out);
            }
          }
          if (thread_index_in_simdgroup == 0) {
            tape_[dv_idx] = delta;
          }
          for (int i = 0; i < n_per_t; ++i) {
            state[i] = static_cast<float>(static_cast<InT>(state[i]));
          }
          q_ += Hk * Dk; k_ += Hk * Dk; v_ += Hv * Dv;
          y += Hv * Dv; tape_ += Hv * Dv; g_ += Hv; beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<InT>(state[i]);
        }
    """

    // Corrected: uniform predicate skips simd_sums when masked (no divergence);
    // metal::select restores pre-decay state when !do_step.
    let gdeltaSelectSrc = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;
        auto q_    = q    + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_    = k    + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto v_    = v    + b_idx * T * Hv * Dv + hv_idx * Dv;
        y         += b_idx * T * Hv * Dv + hv_idx * Dv;
        auto tape_ = innovation_tape + b_idx * T * Hv * Dv + hv_idx * Dv;
        auto g_    = g    + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;
        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;
        auto i_state = state_in  + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;
        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i)
            state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
        for (int t = 0; t < T; ++t) {
            bool do_step = static_cast<float>(mask[b_idx * T + t]) > 0.5f;
            float old_state[n_per_t];
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                old_state[i] = state[i];
                state[i]     = state[i] * g_[hv_idx];
                kv_mem      += state[i] * k_[s_idx];
            }
            float delta = 0.0f;
            float out   = 0.0f;
            if (do_step) {
                kv_mem = simd_sum(kv_mem);
                delta  = (static_cast<float>(v_[dv_idx]) - kv_mem)
                         * static_cast<float>(beta_[hv_idx]);
                for (int i = 0; i < n_per_t; ++i) {
                    auto s_idx = n_per_t * dk_idx + i;
                    state[i] += k_[s_idx] * delta;
                    out      += state[i] * static_cast<float>(q_[s_idx]);
                }
                out = simd_sum(out);
            }
            if (thread_index_in_simdgroup == 0) {
                y[dv_idx]     = static_cast<InT>(out);
                tape_[dv_idx] = delta;
            }
            for (int i = 0; i < n_per_t; ++i) {
                float quant_new = static_cast<float>(static_cast<InT>(state[i]));
                state[i] = metal::select(old_state[i], quant_new, do_step);
            }
            q_ += Hk * Dk; k_ += Hk * Dk; v_ += Hv * Dv;
            y += Hv * Dv; tape_ += Hv * Dv; g_ += Hv; beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i)
            o_state[n_per_t * dk_idx + i] = static_cast<InT>(state[i]);
    """

    let gdeltaKernelBranching = MLXFast.metalKernel(
        name: "bench_gdelta_branching_mask",
        inputNames: ["q", "k", "v", "g", "beta", "state_in", "T", "mask"],
        outputNames: ["y", "state_out", "innovation_tape"],
        source: gdeltaBranchingSrc
    )

    let gdeltaKernelSelect = MLXFast.metalKernel(
        name: "bench_gdelta_select_mask",
        inputNames: ["q", "k", "v", "g", "beta", "state_in", "T", "mask"],
        outputNames: ["y", "state_out", "innovation_tape"],
        source: gdeltaSelectSrc
    )

    func runGdelta(_ kernel: MLXFast.MLXFastKernel) -> MLXArray {
        kernel(
            [q, k, v, g, beta, state, MLXArray(steps), mask],
            template: [("InT", inputType), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
            grid: (32, Dv, B * Hv), threadGroup: (32, 4, 1),
            outputShapes: [[B, T, Hv, Dv], state.shape, [B, T, Hv, Dv]],
            outputDTypes: [inputType, inputType, DType.float32]
        )[0]
    }

    for _ in 0 ..< args.warmup {
        MLX.eval(runGdelta(gdeltaKernelBranching))
        MLX.eval(runGdelta(gdeltaKernelSelect))
    }

    let callBytes = (B*T*Hk*Dk + B*T*Hk*Dk + B*T*Hv*Dv) * 2
                  + B*Hv*Dv*Dk * 2 * 2
                  + B*T*Hv*Dv * 4

    let r3 = measure(label: "bench_gdelta_branching_mask", iterations: args.iterations) {
        runGdelta(gdeltaKernelBranching)
    }
    printResult(label: "gdelta branching (scalar-g, masked)", r: r3,
                extraInfo: bwStr(bytes: callBytes, seconds: r3.median))

    let r4 = measure(label: "bench_gdelta_select_mask", iterations: args.iterations) {
        runGdelta(gdeltaKernelSelect)
    }
    printResult(label: "gdelta select    (scalar-g, masked)", r: r4,
                extraInfo: bwStr(bytes: callBytes, seconds: r4.median))
    print(String(format: "    → select vs branching: %.2fx", r3.median / r4.median))
}

// MARK: - Ops Fallback Comparison (MLX.where vs arithmetic masking)

private func benchOpsFallback(args: Args) {
    print("\n── Ops Fallback: MLX.where vs Arithmetic Masking ───────────────────────")

    let B = 1; let T = 16; let Hk = 8; let Hv = 16; let Dk = 128; let Dv = 128
    let tape  = rand([B, T, Hv, Dv])
    let k     = rand([B, T, Hk, Dk])
    let g     = rand([B, T, Hv])
    let state = rand([B, Hv, Dv, Dk])
    let mask  = (uniform(low: 0, high: 1, [B, T]) .>= MLXArray(0.5)).asType(DType.bfloat16)
    let q     = rand([B, T, Hk, Dk])
    let v     = rand([B, T, Hv, Dv])
    let beta  = rand([B, T, Hv])

    // ── Tape Replay Ops ───────────────────────────────────────────────────────

    // Current: MLX.where selects between new state and old state
    func tapeOpsWhere() -> MLXArray {
        let k_ = MLX.repeated(k, count: Hv / Hk, axis: 2)
        var st = state
        for t in 0 ..< T {
            let prev  = st
            let decay = expandedDimensions(g[0..., t, 0...], axes: [2, 3])
            let delta = tape[0..., t, 0..., .newAxis]
            let kT    = expandedDimensions(k_[0..., t, 0...], axis: -2)
            st = st * decay + delta * kT
            let stepMask = mask[0..., t][.newAxis, .newAxis, .newAxis]
            st = MLX.where(stepMask, st, prev)
        }
        return st
    }

    // Optimized: arithmetic gate — next * gate + state * (1 - gate)
    func tapeOpsArith() -> MLXArray {
        let k_ = MLX.repeated(k, count: Hv / Hk, axis: 2)
        var st = state
        for t in 0 ..< T {
            let decay = expandedDimensions(g[0..., t, 0...], axes: [2, 3])
            let delta = tape[0..., t, 0..., .newAxis]
            let kT    = expandedDimensions(k_[0..., t, 0...], axis: -2)
            let next  = st * decay + delta * kT
            let gate  = expandedDimensions(mask[0..., t], axes: [1, 2, 3]).asType(st.dtype)
            st = next * gate + st * (1 - gate)
        }
        return st
    }

    for _ in 0 ..< args.warmup {
        MLX.eval(tapeOpsWhere())
        MLX.eval(tapeOpsArith())
    }

    let r1 = measure(label: "tape_ops_where", iterations: args.iterations) { tapeOpsWhere() }
    printResult(label: "tape ops  MLX.where  (scalar-g, masked)", r: r1)

    let r2 = measure(label: "tape_ops_arith", iterations: args.iterations) { tapeOpsArith() }
    printResult(label: "tape ops  arith gate (scalar-g, masked)", r: r2)
    print(String(format: "    → arith vs where: %.2fx", r1.median / r2.median))

    // ── GatedDelta + Tape Ops ─────────────────────────────────────────────────

    // Current: MLX.where for state and output gating
    func gdeltaOpsWhere() -> MLXArray {
        let rf  = Hv / Hk
        let q_  = MLX.repeated(q, count: rf, axis: 2)
        let k_  = MLX.repeated(k, count: rf, axis: 2)
        var st  = state
        var outs = [MLXArray]()
        outs.reserveCapacity(T)
        for t in 0 ..< T {
            let oldSt   = st
            let decay   = expandedDimensions(g[0..., t, 0...], axes: [2, 3])
            let decayed = st * decay
            let kvMem   = (decayed * expandedDimensions(k_[0..., t, 0...], axis: -2)).sum(axis: -1)
            let delta   = (v[0..., t, 0...] - kvMem) * expandedDimensions(beta[0..., t, 0...], axis: -1)
            let newSt   = decayed + expandedDimensions(k_[0..., t, 0...], axis: -2)
                                  * expandedDimensions(delta, axis: -1)
            let y       = (newSt * expandedDimensions(q_[0..., t, 0...], axis: -2)).sum(axis: -1)
            let sMask   = mask[0..., t][.newAxis, .newAxis, .newAxis]
            let yMask   = mask[0..., t][.newAxis, .newAxis]
            st = MLX.where(sMask, newSt, oldSt)
            outs.append(MLX.where(yMask, y, MLXArray.zeros(y.shape, dtype: y.dtype)))
        }
        return MLX.stacked(outs, axis: 1)
    }

    // Optimized: arithmetic gate — no MLX.where
    func gdeltaOpsArith() -> MLXArray {
        let rf  = Hv / Hk
        let q_  = MLX.repeated(q, count: rf, axis: 2)
        let k_  = MLX.repeated(k, count: rf, axis: 2)
        var st  = state
        var outs = [MLXArray]()
        outs.reserveCapacity(T)
        for t in 0 ..< T {
            let decay   = expandedDimensions(g[0..., t, 0...], axes: [2, 3])
            let decayed = st * decay
            let kvMem   = (decayed * expandedDimensions(k_[0..., t, 0...], axis: -2)).sum(axis: -1)
            let delta   = (v[0..., t, 0...] - kvMem) * expandedDimensions(beta[0..., t, 0...], axis: -1)
            let next    = decayed + expandedDimensions(k_[0..., t, 0...], axis: -2)
                                  * expandedDimensions(delta, axis: -1)
            let y       = (next * expandedDimensions(q_[0..., t, 0...], axis: -2)).sum(axis: -1)
            let sGate   = expandedDimensions(mask[0..., t], axes: [1, 2, 3]).asType(st.dtype)
            let yGate   = expandedDimensions(mask[0..., t], axes: [1, 2]).asType(y.dtype)
            st = next * sGate + st * (1 - sGate)
            outs.append(y * yGate)
        }
        return MLX.stacked(outs, axis: 1)
    }

    for _ in 0 ..< args.warmup {
        MLX.eval(gdeltaOpsWhere())
        MLX.eval(gdeltaOpsArith())
    }

    let r3 = measure(label: "gdelta_ops_where", iterations: args.iterations) { gdeltaOpsWhere() }
    printResult(label: "gdelta ops  MLX.where  (scalar-g, masked)", r: r3)

    let r4 = measure(label: "gdelta_ops_arith", iterations: args.iterations) { gdeltaOpsArith() }
    printResult(label: "gdelta ops  arith gate (scalar-g, masked)", r: r4)
    print(String(format: "    → arith vs where: %.2fx", r3.median / r4.median))
}

// MARK: - Main

let args = Args()

print("DFlash Kernel Micro-Benchmark")
print("═══════════════════════════════════════════════════════════════════════")
print("  Device:     \(Device.defaultDevice().description)")
print("  Iterations: \(args.iterations)  Warmup: \(args.warmup)")
print("  Kernels:    \(args.kernels.sorted().joined(separator: ", "))")
print("  Long-ctx:   \(args.longCtx)")
print("═══════════════════════════════════════════════════════════════════════")

// Force GPU initialisation before any timing
MLX.eval(MLX.zeros([1]))

if args.kernels.contains("tape")     { benchTapeReplay(args: args) }
if args.kernels.contains("gdelta")   { benchGatedDelta(args: args) }
if args.kernels.contains("sdpa")     { benchSDPA(args: args) }
if args.kernels.contains("variants") { benchKernelVariants(args: args) }
if args.kernels.contains("ops")      { benchOpsFallback(args: args) }

print("\nDone.")
