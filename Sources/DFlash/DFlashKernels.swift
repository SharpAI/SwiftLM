// Copyright 2026 SwiftLM Contributors
// MIT License — see LICENSE file
// Based on DFlash (arXiv:2602.06036)

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Metal kernels for DFlash speculative decoding.
///
/// Provides:
/// - **Tape replay kernel**: Replays accepted innovation steps through the
///   GatedDeltaNet recurrent state for efficient rollback.
/// - **GatedDelta kernel with tape**: Modified GatedDelta forward that records
///   the innovation tape alongside the normal output.
/// - **Batched SDPA 2-pass kernel**: Custom attention kernel for long-context
///   verify that stays numerically aligned with stock MLX attention.
public enum DFlashKernels {

    /// Shared instance for use as the global DFlashKernelProvider
    public static let shared = DFlashKernelsInstance()

    // MARK: - Tape Replay Kernel

    private static func makeTapeReplayKernel(
        hasMask: Bool = false,
        vectorized: Bool = false
    ) -> MLXFast.MLXFastKernel? {
        // Branchless + correct semantics via metal::select:
        //   When mask=0 (do_step=false), metal::select returns the OLD state[i],
        //   so state is completely unchanged — no decay, no accumulate.
        //   When mask=1 (do_step=true), the computed next value is used.
        // metal::select is a conditional move with no warp divergence.
        let maskLoad = hasMask
            ? "bool do_step = static_cast<float>(mask[b_idx * T + t]) > 0.5f;"
            : "constexpr bool do_step = true;"
        let gSetup   = vectorized ? "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;"
                                  : "auto g_ = g + b_idx * T * Hv;"
        let gAccess  = vectorized ? "g_[s_idx]" : "g_[hv_idx]"
        let gAdvance = vectorized ? "g_ += Hv * Dk;" : "g_ += Hv;"

        let source = """
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

            \(gSetup)

            float state[n_per_t];
            for (int i = 0; i < n_per_t; ++i)
                state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);

            for (int t = 0; t < T; ++t) {
                \(maskLoad)
                float delta = static_cast<float>(tape_[dv_idx]);
                for (int i = 0; i < n_per_t; ++i) {
                    auto s_idx = n_per_t * dk_idx + i;
                    float next = state[i] * \(gAccess) + k_[s_idx] * delta;
                    next = static_cast<float>(static_cast<InT>(next));
                    // Conditional move: old state when masked, next when accepted.
                    state[i] = metal::select(state[i], next, do_step);
                }
                tape_ += Hv * Dv;
                k_    += Hk * Dk;
                \(gAdvance)
            }

            for (int i = 0; i < n_per_t; ++i)
                o_state[n_per_t * dk_idx + i] = static_cast<InT>(state[i]);
        """

        var inputNames = ["tape", "k", "g", "state_in", "T"]
        if hasMask { inputNames.append("mask") }

        var suffix = ""
        if vectorized { suffix += "_vec" }
        if hasMask { suffix += "_mask" }

        return MLXFast.metalKernel(
            name: "dflash_tape_replay\(suffix)",
            inputNames: inputNames,
            outputNames: ["state_out"],
            source: source
        )
    }

    // MARK: - GatedDelta with Tape Kernel

    private static func makeGatedDeltaTapeKernel(
        hasMask: Bool = false,
        vectorized: Bool = false
    ) -> MLXFast.MLXFastKernel? {
        // Two optimizations over the naive branching version:
        //
        // 1. Uniform simdgroup predicate: mask[b_idx*T+t] is the same scalar for
        //    every thread in the simdgroup (uniform control flow). Wrapping the two
        //    expensive simd_sum calls in `if (do_step)` skips ~50% of them at
        //    typical acceptance rates with zero warp divergence.
        //
        // 2. metal::select for state correctness: state must be completely
        //    unchanged when mask=0 (no decay). We save state before the decay pass,
        //    then use metal::select to restore it when !do_step.
        let maskLoad = hasMask
            ? "bool do_step = static_cast<float>(mask[b_idx * T + t]) > 0.5f;"
            : "constexpr bool do_step = true;"
        let gSetup   = vectorized ? "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;"
                                  : "auto g_ = g + b_idx * T * Hv;"
        let gAccess  = vectorized ? "g_[s_idx]" : "g_[hv_idx]"
        let gAdvance = vectorized ? "g_ += Hv * Dk;" : "g_ += Hv;"

        let source = """
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

            auto dk_idx = thread_position_in_threadgroup.x;
            auto dv_idx = thread_position_in_grid.y;

            auto i_state = state_in  + (n * Dv + dv_idx) * Dk;
            auto o_state = state_out + (n * Dv + dv_idx) * Dk;

            \(gSetup)
            auto beta_ = beta + b_idx * T * Hv;

            float state[n_per_t];
            for (int i = 0; i < n_per_t; ++i)
                state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);

            for (int t = 0; t < T; ++t) {
                \(maskLoad)

                // Save pre-decay state; needed by metal::select to restore when !do_step.
                float old_state[n_per_t];
                float kv_mem = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                    auto s_idx = n_per_t * dk_idx + i;
                    old_state[i] = state[i];
                    state[i]     = state[i] * \(gAccess);
                    kv_mem      += state[i] * k_[s_idx];
                }

                // Uniform predicate: skip two simd_sum calls when !do_step.
                // All threads in the simdgroup read the same mask scalar → no divergence.
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

                // Restore pre-decay state when !do_step; quantize new state when do_step.
                for (int i = 0; i < n_per_t; ++i) {
                    float quant_new = static_cast<float>(static_cast<InT>(state[i]));
                    state[i] = metal::select(old_state[i], quant_new, do_step);
                }

                q_    += Hk * Dk;
                k_    += Hk * Dk;
                v_    += Hv * Dv;
                y     += Hv * Dv;
                tape_ += Hv * Dv;
                \(gAdvance)
                beta_ += Hv;
            }

            for (int i = 0; i < n_per_t; ++i)
                o_state[n_per_t * dk_idx + i] = static_cast<InT>(state[i]);
        """

        var inputNames = ["q", "k", "v", "g", "beta", "state_in", "T"]
        if hasMask { inputNames.append("mask") }

        var suffix = ""
        if vectorized { suffix += "_vec" }
        if hasMask { suffix += "_mask" }

        return MLXFast.metalKernel(
            name: "dflash_gated_delta_tape\(suffix)",
            inputNames: inputNames,
            outputNames: ["y", "state_out", "innovation_tape"],
            source: source
        )
    }

    // MARK: - Lazy Kernel Singleton

    private final class KernelCache {
        static let shared = KernelCache()

        // Layout: [vectorized (0/1)][masked (0/1)]
        let tapeReplay:     [[MLXFast.MLXFastKernel?]]
        let gatedDeltaTape: [[MLXFast.MLXFastKernel?]]

        private init() {
            tapeReplay = [
                [makeTapeReplayKernel(hasMask: false, vectorized: false),
                 makeTapeReplayKernel(hasMask: true,  vectorized: false)],
                [makeTapeReplayKernel(hasMask: false, vectorized: true),
                 makeTapeReplayKernel(hasMask: true,  vectorized: true)],
            ]
            gatedDeltaTape = [
                [makeGatedDeltaTapeKernel(hasMask: false, vectorized: false),
                 makeGatedDeltaTapeKernel(hasMask: true,  vectorized: false)],
                [makeGatedDeltaTapeKernel(hasMask: false, vectorized: true),
                 makeGatedDeltaTapeKernel(hasMask: true,  vectorized: true)],
            ]
        }
    }

    // MARK: - Public API: Tape Replay

    /// Replay the innovation tape through the GatedDeltaNet state.
    ///
    /// - Parameters:
    ///   - tape: Innovation tape [B, T, Hv, Dv]
    ///   - k: Keys [B, T, Hk, Dk]
    ///   - g: Gates (decay) — either [B, T, Hv] or [B, T, Hv, Dk]
    ///   - state: Current recurrent state [B, Hv, Dv, Dk]
    ///   - mask: Optional mask [B, T]
    /// - Returns: Replayed state [B, Hv, Dv, Dk]
    public static func tapeReplayKernel(
        tape: MLXArray,
        k: MLXArray,
        g: MLXArray,
        state: MLXArray,
        mask: MLXArray? = nil
    ) -> MLXArray {
        let forceFallback = ProcessInfo.processInfo.environment["DFLASH_FORCE_OPS"] != nil
        let isCPU = Device.defaultDevice().deviceType == .cpu
        if isCPU || forceFallback { return tapeReplayOps(tape: tape, k: k, g: g, state: state, mask: mask) }

        let B = k.dim(0)
        let steps = k.dim(1)
        let Hk = k.dim(2)
        let Dk = k.dim(3)
        let Hv = tape.dim(2)
        let Dv = tape.dim(3)
        let inputType = state.dtype

        if Dk < 32 || Dk % 32 != 0 {
            return tapeReplayOps(tape: tape, k: k, g: g, state: state, mask: mask)
        }

        let vec = g.ndim == 4 ? 1 : 0
        let msk = mask != nil  ? 1 : 0
        let kernel = KernelCache.shared.tapeReplay[vec][msk]

        guard let kernel else {
            return tapeReplayOps(tape: tape, k: k, g: g, state: state, mask: mask)
        }

        var inputs: [MLXArray] = [tape, k, g, state, MLXArray(steps)]
        if let mask { inputs.append(mask) }

        let outputs = kernel(
            inputs,
            template: [
                ("InT", inputType),
                ("Dk", Dk),
                ("Dv", Dv),
                ("Hk", Hk),
                ("Hv", Hv),
            ],
            grid: (32, Dv, B * Hv),
            threadGroup: (32, 4, 1),
            outputShapes: [state.shape],
            outputDTypes: [inputType]
        )
        return outputs[0]
    }

    // MARK: - Public API: GatedDelta with Tape

    /// Run GatedDelta forward while recording the innovation tape for rollback.
    ///
    /// - Parameters:
    ///   - q: Queries [B, T, Hk, Dk]
    ///   - k: Keys [B, T, Hk, Dk]
    ///   - v: Values [B, T, Hv, Dv]
    ///   - g: Gates (decay) — either [B, T, Hv] or [B, T, Hv, Dk]
    ///   - beta: Beta values [B, T, Hv]
    ///   - state: Recurrent state [B, Hv, Dv, Dk]
    ///   - mask: Optional mask [B, T]
    /// - Returns: Tuple of (output [B, T, Hv, Dv], new state, innovation tape [B, T, Hv, Dv])
    public static func gatedDeltaKernelWithTape(
        q: MLXArray,
        k: MLXArray,
        v: MLXArray,
        g: MLXArray,
        beta: MLXArray,
        state: MLXArray,
        mask: MLXArray? = nil
    ) -> (MLXArray, MLXArray, MLXArray) {
        let forceFallback = ProcessInfo.processInfo.environment["DFLASH_FORCE_OPS"] != nil
        let isCPU = Device.defaultDevice().deviceType == .cpu
        if isCPU || forceFallback { return gatedDeltaOpsWithTape(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask) }

        let B = k.dim(0)
        let T = k.dim(1)
        let Hk = k.dim(2)
        let Dk = k.dim(3)
        let Hv = v.dim(2)
        let Dv = v.dim(3)

        if Dk < 32 || Dk % 32 != 0 {
            return gatedDeltaOpsWithTape(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
        }

        let inputType = q.dtype
        let vec = g.ndim == 4 ? 1 : 0
        let msk = mask != nil  ? 1 : 0
        let kernel = KernelCache.shared.gatedDeltaTape[vec][msk]

        guard let kernel else {
            return gatedDeltaOpsWithTape(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
        }

        var inputs: [MLXArray] = [q, k, v, g, beta, state, MLXArray(T)]
        if let mask { inputs.append(mask) }

        let outputs = kernel(
            inputs,
            template: [
                ("InT", inputType),
                ("Dk", Dk),
                ("Dv", Dv),
                ("Hk", Hk),
                ("Hv", Hv),
            ],
            grid: (32, Dv, B * Hv),
            threadGroup: (32, 4, 1),
            outputShapes: [[B, T, Hv, Dv], state.shape, [B, T, Hv, Dv]],
            outputDTypes: [inputType, inputType, DType.float32]
        )
        return (outputs[0], outputs[1], outputs[2])
    }

    // MARK: - Fallback: Ops-based implementations

    private static func tapeReplayOps(
        tape: MLXArray,
        k: MLXArray,
        g: MLXArray,
        state: MLXArray,
        mask: MLXArray? = nil
    ) -> MLXArray {
        let T = tape.dim(1)
        let Hk = k.dim(2)
        let Hv = tape.dim(2)
        let repeatFactor = Hv / Hk
        var k = k
        if repeatFactor > 1 {
            k = MLX.repeated(k, count: repeatFactor, axis: 2)
        }

        var state = state
        for t in 0 ..< T {
            let prev = state
            let decay: MLXArray
            if g.ndim == 4 {
                decay = g[0..., t, 0..., .newAxis, 0...]
            } else {
                decay = expandedDimensions(g[0..., t, 0...], axes: [2, 3])
            }
            let delta = tape[0..., t, 0..., .newAxis]
            let kT = expandedDimensions(k[0..., t, 0...], axis: -2)
            state = state * decay + delta * kT
            if let mask {
                // MLX.where is faster than arithmetic masking for tape replay ops
                // (benchmark: 382 µs vs 455 µs on M-series, scalar-g masked).
                let stepMask = mask[0..., t][.newAxis, .newAxis, .newAxis]
                state = MLX.where(stepMask, state, prev)
            }
        }
        return state
    }

    private static func gatedDeltaOpsWithTape(
        q: MLXArray,
        k: MLXArray,
        v: MLXArray,
        g: MLXArray,
        beta: MLXArray,
        state: MLXArray,
        mask: MLXArray? = nil
    ) -> (MLXArray, MLXArray, MLXArray) {
        let T = q.dim(1)
        let Hk = q.dim(2)
        let Hv = v.dim(2)
        let repeatFactor = Hv / Hk
        var q = q
        var k = k
        if repeatFactor > 1 {
            q = MLX.repeated(q, count: repeatFactor, axis: 2)
            k = MLX.repeated(k, count: repeatFactor, axis: 2)
        }

        var state = state
        var outputs = [MLXArray]()
        var tapeEntries = [MLXArray]()

        for t in 0 ..< T {
            let decay: MLXArray
            if g.ndim == 4 {
                decay = g[0..., t, 0..., .newAxis, 0...]
            } else {
                decay = expandedDimensions(g[0..., t, 0...], axes: [2, 3])
            }
            let decayedState = state * decay
            let kvMem = (decayedState * expandedDimensions(k[0..., t, 0...], axis: -2)).sum(axis: -1)
            let delta = (v[0..., t, 0...] - kvMem) * expandedDimensions(beta[0..., t, 0...], axis: -1)
            let newState = decayedState + expandedDimensions(k[0..., t, 0...], axis: -2) * expandedDimensions(delta, axis: -1)
            let y = (newState * expandedDimensions(q[0..., t, 0...], axis: -2)).sum(axis: -1)

            if let mask {
                // Arithmetic masking is faster than MLX.where for gdelta ops
                // (benchmark: 816 µs vs 1005 µs on M-series, scalar-g masked).
                let sGate = expandedDimensions(mask[0..., t], axes: [1, 2, 3]).asType(state.dtype)
                let yGate = expandedDimensions(mask[0..., t], axes: [1, 2]).asType(y.dtype)
                state = newState * sGate + state * (1 - sGate)
                outputs.append(y * yGate)
                tapeEntries.append((delta * yGate).asType(DType.float32))
            } else {
                state = newState
                outputs.append(y)
                tapeEntries.append(delta.asType(DType.float32))
            }
        }

        return (
            MLX.stacked(outputs, axis: 1),
            state,
            MLX.stacked(tapeEntries, axis: 1)
        )
    }

    // MARK: - Block Computation for 2-Pass SDPA

    private static func computeSDPA2PassBlocks(gqaFactor: Int, nKV: Int, deviceArch: String? = nil) -> Int {
        let arch = deviceArch ?? Device.defaultDevice().description
        let devc = arch.isEmpty ? "" : String(arch.suffix(1))
        let nSimds = gqaFactor
        let N = nKV

        var blocks: Int
        if devc == "d" {
            blocks = 128
            if nSimds <= 2 && N > 8192 {
                blocks = 256
            } else if nSimds >= 6 {
                if N >= 16384 && N < 65536 {
                    blocks = 512
                } else if N >= 65536 {
                    blocks = 1024
                }
            }
        } else if devc == "s" {
            blocks = 64
            if N > 1024 && nSimds > 4 {
                if N <= 8192 {
                    blocks = 128
                } else if N <= 32768 {
                    blocks = 256
                } else if N <= 65536 {
                    blocks = 512
                } else {
                    blocks = 1024
                }
            }
        } else {
            blocks = nSimds >= 4 ? 64 : 32
        }

        return blocks
    }

    // MARK: - Batched SDPA 2-Pass Kernels

    private final class SDPAKernelCache {
        static let shared = SDPAKernelCache()

        private var _partialsKernel: MLXFast.MLXFastKernel?
        private var _partialsKernelMasked: MLXFast.MLXFastKernel?
        private var _reduceKernel: MLXFast.MLXFastKernel?
        private var _initialized = false
        private let _lock = NSLock()

        var partialsKernel: MLXFast.MLXFastKernel? {
            _lock.lock(); defer { _lock.unlock() }
            if !_initialized { _initAll() }
            return _partialsKernel
        }

        var partialsKernelMasked: MLXFast.MLXFastKernel? {
            _lock.lock(); defer { _lock.unlock() }
            if !_initialized { _initAll() }
            return _partialsKernelMasked
        }

        var reduceKernel: MLXFast.MLXFastKernel? {
            _lock.lock(); defer { _lock.unlock() }
            if !_initialized { _initAll() }
            return _reduceKernel
        }

        private init() {}

        private func _initAll() {
            _partialsKernel = SDPAKernelCache.makePartialsKernel(hasMask: false)
            _partialsKernelMasked = SDPAKernelCache.makePartialsKernel(hasMask: true)
            _reduceKernel = SDPAKernelCache.makeReduceKernel()
            _initialized = true
        }

        private static func makePartialsKernel(hasMask: Bool) -> MLXFast.MLXFastKernel? {
            let maskSetup = hasMask
                ? "auto mask_ = mask + (((b_idx * Hq + q_head_idx) * M_FIXED + q_seq_idx) * N + block_idx);"
                : ""
            let maskUseKey = hasMask
                ? "auto mask_value = static_cast<float>(mask_[0]); use_key = use_key && (mask_value >= Limits<InT>::finite_min);"
                : ""
            let maskScore = hasMask ? "score += static_cast<float>(mask_[0]);" : ""
            let maskAdvance = hasMask ? "mask_ += blocks;" : ""

            var inputs = [
                "queries", "keys", "values", "gqa_factor", "N",
                "k_head_stride", "k_seq_stride", "v_head_stride", "v_seq_stride",
                "scale", "blocks"
            ]
            if hasMask { inputs.append("mask") }

            let source = """
                constexpr int BD = 32;
                constexpr int qk_per_thread = D / BD;
                constexpr int v_per_thread = V / BD;

                auto q_head_idx = threadgroup_position_in_grid.x;
                auto b_idx = threadgroup_position_in_grid.y;
                auto block_idx = threadgroup_position_in_grid.z;
                auto q_seq_idx = thread_position_in_threadgroup.z;
                auto simd_lid = thread_index_in_simdgroup;

                auto Hq = threadgroups_per_grid.x;
                auto hk_idx = q_head_idx / gqa_factor;
                auto q_batch_head_idx = b_idx * Hq + q_head_idx;
                auto o_offset = q_batch_head_idx * M_FIXED + q_seq_idx;

                auto q_ = queries + (o_offset * D) + simd_lid * qk_per_thread;
                auto k_ = keys + ((b_idx * Hk + hk_idx) * k_head_stride) + block_idx * k_seq_stride + simd_lid * qk_per_thread;
                auto v_ = values + ((b_idx * Hk + hk_idx) * v_head_stride) + block_idx * v_seq_stride + simd_lid * v_per_thread;

                partials += (o_offset * blocks + block_idx) * V + simd_lid * v_per_thread;
                sums += o_offset * blocks + block_idx;
                maxs += o_offset * blocks + block_idx;
                \(maskSetup)

                thread float q[qk_per_thread];
                thread float o[v_per_thread];
                threadgroup InT tg_k[BD * qk_per_thread];
                threadgroup InT tg_v[BD * v_per_thread];

                for (int i = 0; i < qk_per_thread; ++i) {
                    q[i] = static_cast<float>(scale) * static_cast<float>(q_[i]);
                }
                for (int i = 0; i < v_per_thread; ++i) {
                    o[i] = 0.0f;
                }

                float max_score = Limits<float>::finite_min;
                float sum_exp_score = 0.0f;

                for (int n = block_idx; n < N; n += blocks) {
                    if (q_seq_idx == 0) {
                        for (int i = 0; i < qk_per_thread; ++i) {
                            tg_k[simd_lid * qk_per_thread + i] = k_[i];
                        }
                        for (int i = 0; i < v_per_thread; ++i) {
                            tg_v[simd_lid * v_per_thread + i] = v_[i];
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    bool use_key = (n <= (N - M_FIXED + q_seq_idx));
                    \(maskUseKey)

                    if (use_key) {
                        float score = 0.0f;
                        for (int i = 0; i < qk_per_thread; ++i) {
                            score += q[i] * static_cast<float>(tg_k[simd_lid * qk_per_thread + i]);
                        }
                        score = simd_sum(score);
                        \(maskScore)

                        float new_max = metal::max(max_score, score);
                        float factor = fast::exp(max_score - new_max);
                        float exp_score = fast::exp(score - new_max);

                        max_score = new_max;
                        sum_exp_score = sum_exp_score * factor + exp_score;
                        for (int i = 0; i < v_per_thread; ++i) {
                            o[i] = o[i] * factor + exp_score * static_cast<float>(tg_v[simd_lid * v_per_thread + i]);
                        }
                    }

                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    k_ += blocks * int(k_seq_stride);
                    v_ += blocks * int(v_seq_stride);
                    \(maskAdvance)
                }

                if (simd_lid == 0) {
                    sums[0] = sum_exp_score;
                    maxs[0] = max_score;
                }
                for (int i = 0; i < v_per_thread; ++i) {
                    partials[i] = static_cast<InT>(o[i]);
                }
            """

            let suffix = hasMask ? "_mask" : ""
            return MLXFast.metalKernel(
                name: "batched_sdpa_2pass_partials\(suffix)",
                inputNames: inputs,
                outputNames: ["partials", "sums", "maxs"],
                source: source
            )
        }

        private static func makeReduceKernel() -> MLXFast.MLXFastKernel? {
            let source = """
                constexpr int BN = 32;
                constexpr int BD = 32;
                constexpr int elem_per_thread = V / BD;

                auto head_idx = threadgroup_position_in_grid.x;
                auto q_seq_idx = threadgroup_position_in_grid.y;
                auto simd_gid = simdgroup_index_in_threadgroup;
                auto simd_lid = thread_index_in_simdgroup;

                auto q_offset = head_idx * M_FIXED + q_seq_idx;
                partials += (q_offset * blocks + simd_gid) * V + simd_lid * elem_per_thread;
                sums += q_offset * blocks;
                maxs += q_offset * blocks;
                out += q_offset * V + simd_gid * elem_per_thread;

                thread float o[elem_per_thread];
                threadgroup float outputs[BN * BD];

                for (int i = 0; i < elem_per_thread; ++i) {
                    o[i] = 0.0f;
                }

                float sum_exp_score = 0.0f;
                float max_score = Limits<float>::finite_min;

                for (int b = 0; b < blocks / BN; ++b) {
                    max_score = metal::max(max_score, maxs[simd_lid + BN * b]);
                }
                max_score = simd_max(max_score);

                for (int b = 0; b < blocks / BN; ++b) {
                    float factor = fast::exp(maxs[simd_lid + BN * b] - max_score);
                    sum_exp_score += factor * sums[simd_lid + BN * b];
                }
                sum_exp_score = simd_sum(sum_exp_score);

                for (int b = 0; b < blocks / BN; ++b) {
                    float factor = fast::exp(maxs[simd_gid] - max_score);
                    for (int i = 0; i < elem_per_thread; ++i) {
                        o[i] += factor * static_cast<float>(partials[i]);
                    }
                    maxs += BN;
                    partials += BN * V;
                }

                for (int i = 0; i < elem_per_thread; ++i) {
                    outputs[simd_lid * BD + simd_gid] = o[i];
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
                    o[i] = sum_exp_score == 0.0f ? o[i] : (o[i] / sum_exp_score);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }

                if (simd_lid == 0) {
                    for (int i = 0; i < elem_per_thread; ++i) {
                        out[i] = static_cast<InT>(o[i]);
                    }
                }
            """

            return MLXFast.metalKernel(
                name: "batched_sdpa_2pass_reduce",
                inputNames: ["partials", "sums", "maxs", "blocks"],
                outputNames: ["out"],
                source: source
            )
        }
    }

    // MARK: - Public API: Batched SDPA

    /// Batched 2-pass SDPA for DFlash verify phase with long context.
    ///
    /// Optimized for: query length 16, bfloat16/float16, head dim 128 or 256.
    /// Returns nil if conditions are not met; callers should fall back to `sdpaFallback`.
    public static func batchedSDPA2Pass(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        scale: Float,
        mask: MLXArray? = nil
    ) -> MLXArray? {
        guard queries.ndim == 4, keys.ndim == 4, values.ndim == 4 else { return nil }

        let B = queries.dim(0)
        let Hq = queries.dim(1)
        let qLen = queries.dim(2)
        let D = queries.dim(3)
        let Hk = keys.dim(1)
        let nKV = keys.dim(2)
        let Vdim = values.dim(3)
        let inputType = queries.dtype

        guard qLen == 16 else { return nil }
        guard inputType == .bfloat16 || inputType == .float16 else { return nil }
        guard (D == 128 || D == 256) && (Vdim == 128 || Vdim == 256) && D == Vdim else { return nil }
        guard Hk > 0 && Hq % Hk == 0 else { return nil }

        let queriesContig = MLX.contiguous(queries)
        let keysContig = MLX.contiguous(keys)
        let valuesContig = MLX.contiguous(values)

        let gqaFactor = Hq / Hk
        let blocks = computeSDPA2PassBlocks(gqaFactor: gqaFactor, nKV: nKV)
        guard blocks > 0 && blocks % 32 == 0 else { return nil }

        let kHeadStride = keys.dim(2) * keys.dim(3)
        let kSeqStride = keys.dim(3)
        let vHeadStride = values.dim(2) * values.dim(3)
        let vSeqStride = values.dim(3)

        let cache = SDPAKernelCache.shared
        var kernel = cache.partialsKernel
        var inputs: [MLXArray] = [
            queriesContig, keysContig, valuesContig,
            MLXArray(gqaFactor), MLXArray(nKV),
            MLXArray(kHeadStride), MLXArray(kSeqStride),
            MLXArray(vHeadStride), MLXArray(vSeqStride),
            MLXArray(scale), MLXArray(blocks)
        ]

        if let mask {
            let maskContig = mask.dtype != inputType ? mask.asType(inputType) : mask
            kernel = cache.partialsKernelMasked
            inputs.append(maskContig)
        }

        guard let partialsKernel = kernel, let reduceKernel = cache.reduceKernel else { return nil }

        let partialShape = [B * Hq, qLen, blocks, Vdim]
        let statsShape = [B * Hq, qLen, blocks]

        let outputs1 = partialsKernel(
            inputs,
            template: [
                ("InT", inputType), ("D", D), ("V", Vdim), ("Hk", Hk), ("M_FIXED", qLen)
            ],
            grid: (Hq * 32, B, blocks * qLen),
            threadGroup: (32, 1, qLen),
            outputShapes: [partialShape, statsShape, statsShape],
            outputDTypes: [inputType, .float32, .float32]
        )

        let outputs2 = reduceKernel(
            [outputs1[0], outputs1[1], outputs1[2], MLXArray(blocks)],
            template: [("InT", inputType), ("V", Vdim), ("M_FIXED", qLen)],
            grid: ((B * Hq) * 1024, qLen, 1),
            threadGroup: (1024, 1, 1),
            outputShapes: [queries.shape],
            outputDTypes: [inputType]
        )

        return outputs2[0]
    }

    /// Fallback SDPA using MLXFast when batched kernel conditions are not met.
    public static func sdpaFallback(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        scale: Float,
        mask: MLXArray? = nil
    ) -> MLXArray {
        MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask
        )
    }
}

/// Concrete DFlashKernelProvider that delegates to DFlashKernels static methods.
public final class DFlashKernelsInstance: DFlashKernelProvider, @unchecked Sendable {
    public func gatedDeltaKernelWithTape(
        q: MLXArray, k: MLXArray, v: MLXArray,
        g: MLXArray, beta: MLXArray,
        state: MLXArray, mask: MLXArray?
    ) -> (MLXArray, MLXArray, MLXArray) {
        DFlashKernels.gatedDeltaKernelWithTape(
            q: q, k: k, v: v, g: g, beta: beta,
            state: state, mask: mask
        )
    }
}
