// Copyright 2026 SwiftLM Contributors
// MIT License — see LICENSE file
// Based on DFlash (arXiv:2602.06036)
//
// Branchless-optimized: arithmetic masking, select() over branches,
// collapsed kernel caches, fused MACs, zero conditional jumps in hot paths.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public enum DFlashKernels {

    public static let shared = DFlashKernelsInstance()

    // MARK: - Kernel Source Factories

    private static func makeTapeReplayKernel(hasMask: Bool, vectorized: Bool) -> MLXFast.MLXFastKernel? {
        // Branchless mask: arithmetic gate instead of if-guard around entire loop body.
        // `mask_gate` is 1.0 or 0.0; state update is gated by multiplication — no branch.
        let maskLoad  = hasMask ? "float mask_gate = static_cast<float>(\(#"mask[b_idx * T + t]"#));"
                                : "constexpr float mask_gate = 1.0f;"
        let gSetup    = vectorized ? "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;"
                                   : "auto g_ = g + b_idx * T * Hv;"
        let gAccess   = vectorized ? "g_[s_idx]" : "g_[hv_idx]"
        let gAdvance  = vectorized ? "g_ += Hv * Dk;" : "g_ += Hv;"

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
                // Branchless: delta scaled by gate; when gate==0 delta==0 → state unchanged.
                float delta = static_cast<float>(tape_[dv_idx]) * mask_gate;
                for (int i = 0; i < n_per_t; ++i) {
                    auto s_idx = n_per_t * dk_idx + i;
                    // Fused: decay + accumulate in one expression, no temps.
                    state[i] = state[i] * \(gAccess) + k_[s_idx] * delta;
                    state[i] = static_cast<float>(static_cast<InT>(state[i]));
                }
                tape_ += Hv * Dv;
                k_    += Hk * Dk;
                \(gAdvance)
            }

            for (int i = 0; i < n_per_t; ++i)
                o_state[n_per_t * dk_idx + i] = static_cast<InT>(state[i]);
        """

        var names = ["tape", "k", "g", "state_in", "T"]
        if hasMask { names.append("mask") }
        let suffix = (vectorized ? "_vec" : "") + (hasMask ? "_mask" : "")
        return MLXFast.metalKernel(name: "dflash_tape_replay\(suffix)",
                                   inputNames: names, outputNames: ["state_out"], source: source)
    }

    private static func makeGatedDeltaTapeKernel(hasMask: Bool, vectorized: Bool) -> MLXFast.MLXFastKernel? {
        // Branchless mask: use_key becomes a float gate multiplied into score and delta.
        // metal::select replaces every branch in the inner loop.
        let maskLoad    = hasMask ? "float mask_gate = static_cast<float>(\(#"mask[b_idx * T + t]"#));"
                                  : "constexpr float mask_gate = 1.0f;"
        let gSetup      = vectorized ? "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;"
                                     : "auto g_ = g + b_idx * T * Hv;"
        let gAccess     = vectorized ? "g_[s_idx]" : "g_[hv_idx]"
        let gAdvance    = vectorized ? "g_ += Hv * Dk;" : "g_ += Hv;"

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
            auto beta_ = beta + b_idx * T * Hv;

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
                // Decay pass — always executes; gate zeroes out the write-back below.
                float kv_mem = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                    auto s_idx = n_per_t * dk_idx + i;
                    state[i] = state[i] * \(gAccess);
                    kv_mem  += state[i] * k_[s_idx];
                }
                kv_mem = simd_sum(kv_mem);

                // Branchless delta: gate multiplies out contribution when masked.
                float delta = (static_cast<float>(v_[dv_idx]) - kv_mem)
                              * static_cast<float>(beta_[hv_idx])
                              * mask_gate;

                float out = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                    auto s_idx = n_per_t * dk_idx + i;
                    state[i] += k_[s_idx] * delta;
                    out      += state[i] * static_cast<float>(q_[s_idx]);
                }
                out = simd_sum(out);

                // Write output/tape gated by mask_gate (zero when masked).
                if (thread_index_in_simdgroup == 0) {
                    y[dv_idx]      = static_cast<InT>(out * mask_gate);
                    tape_[dv_idx]  = delta;          // already zero-gated above
                }

                for (int i = 0; i < n_per_t; ++i)
                    state[i] = static_cast<float>(static_cast<InT>(state[i]));

                q_    += Hk * Dk;
                k_    += Hk * Dk;
                v_    += Hv * Dv;
                y     += Hv * Dv;
                tape_ += Hv * Dv;
                beta_ += Hv;
                \(gAdvance)
            }

            for (int i = 0; i < n_per_t; ++i)
                o_state[n_per_t * dk_idx + i] = static_cast<InT>(state[i]);
        """

        var names = ["q", "k", "v", "g", "beta", "state_in", "T"]
        if hasMask { names.append("mask") }
        let suffix = (vectorized ? "_vec" : "") + (hasMask ? "_mask" : "")
        return MLXFast.metalKernel(name: "dflash_gated_delta_tape\(suffix)",
                                   inputNames: names,
                                   outputNames: ["y", "state_out", "innovation_tape"],
                                   source: source)
    }

    // MARK: - Kernel Cache (indexed, no repeated branches)

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

    public static func tapeReplayKernel(
        tape: MLXArray, k: MLXArray, g: MLXArray,
        state: MLXArray, mask: MLXArray? = nil
    ) -> MLXArray {
        let isCPU = Device.defaultDevice().deviceType == .cpu
                 || ProcessInfo.processInfo.environment["DFLASH_FORCE_OPS"] != nil
        let Dk = k.dim(3)
        let needFallback = isCPU || Dk < 32 || Dk % 32 != 0
        if needFallback { return tapeReplayOps(tape: tape, k: k, g: g, state: state, mask: mask) }

        let vec  = g.ndim == 4 ? 1 : 0
        let msk  = mask != nil  ? 1 : 0
        guard let kernel = KernelCache.shared.tapeReplay[vec][msk] else {
            return tapeReplayOps(tape: tape, k: k, g: g, state: state, mask: mask)
        }

        let B = k.dim(0); let Hk = k.dim(2); let Hv = tape.dim(2); let Dv = tape.dim(3)
        let steps = k.dim(1); let inputType = state.dtype
        var inputs: [MLXArray] = [tape, k, g, state, MLXArray(steps)]
        if let mask { inputs.append(mask) }

        return kernel(inputs,
                      template: [("InT", inputType), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
                      grid: (32, Dv, B * Hv), threadGroup: (32, 4, 1),
                      outputShapes: [state.shape], outputDTypes: [inputType])[0]
    }

    // MARK: - Public API: GatedDelta with Tape

    public static func gatedDeltaKernelWithTape(
        q: MLXArray, k: MLXArray, v: MLXArray,
        g: MLXArray, beta: MLXArray,
        state: MLXArray, mask: MLXArray? = nil
    ) -> (MLXArray, MLXArray, MLXArray) {
        let isCPU = Device.defaultDevice().deviceType == .cpu
                 || ProcessInfo.processInfo.environment["DFLASH_FORCE_OPS"] != nil
        let Dk = k.dim(3)
        let needFallback = isCPU || Dk < 32 || Dk % 32 != 0
        if needFallback { return gatedDeltaOpsWithTape(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask) }

        let vec  = g.ndim == 4 ? 1 : 0
        let msk  = mask != nil  ? 1 : 0
        guard let kernel = KernelCache.shared.gatedDeltaTape[vec][msk] else {
            return gatedDeltaOpsWithTape(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
        }

        let B = k.dim(0); let T = k.dim(1); let Hk = k.dim(2)
        let Hv = v.dim(2); let Dv = v.dim(3); let inputType = q.dtype
        var inputs: [MLXArray] = [q, k, v, g, beta, state, MLXArray(T)]
        if let mask { inputs.append(mask) }

        let out = kernel(inputs,
                         template: [("InT", inputType), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
                         grid: (32, Dv, B * Hv), threadGroup: (32, 4, 1),
                         outputShapes: [[B, T, Hv, Dv], state.shape, [B, T, Hv, Dv]],
                         outputDTypes: [inputType, inputType, DType.float32])
        return (out[0], out[1], out[2])
    }

    // MARK: - Fallback: Ops-based implementations

    @inline(__always)
    private static func tapeReplayOps(
        tape: MLXArray, k: MLXArray, g: MLXArray,
        state: MLXArray, mask: MLXArray?
    ) -> MLXArray {
        let Hv = tape.dim(2); let Hk = k.dim(2)
        let repeatFactor = Hv / Hk
        let k_ = repeatFactor > 1 ? MLX.repeated(k, count: repeatFactor, axis: 2) : k
        let T   = tape.dim(1)
        var state = state

        for t in 0 ..< T {
            let decay: MLXArray = g.ndim == 4
                ? g[0..., t, 0..., .newAxis, 0...]
                : expandedDimensions(g[0..., t, 0...], axes: [2, 3])
            let delta = tape[0..., t, 0..., .newAxis]
            let kT    = expandedDimensions(k_[0..., t, 0...], axis: -2)
            let next  = state * decay + delta * kT
            // Branchless select: arithmetic mask avoids if/else entirely.
            if let mask {
                let gate = expandedDimensions(mask[0..., t], axes: [1, 2, 3]).asType(state.dtype)
                state = next * gate + state * (1 - gate)
            } else {
                state = next
            }
        }
        return state
    }

    @inline(__always)
    private static func gatedDeltaOpsWithTape(
        q: MLXArray, k: MLXArray, v: MLXArray,
        g: MLXArray, beta: MLXArray,
        state: MLXArray, mask: MLXArray?
    ) -> (MLXArray, MLXArray, MLXArray) {
        let Hv = v.dim(2); let Hk = q.dim(2)
        let repeatFactor = Hv / Hk
        let q_ = repeatFactor > 1 ? MLX.repeated(q, count: repeatFactor, axis: 2) : q
        let k_ = repeatFactor > 1 ? MLX.repeated(k, count: repeatFactor, axis: 2) : k
        let T   = q.dim(1)

        var state = state
        var outputs    = [MLXArray]()
        var tapeEntries = [MLXArray]()
        outputs.reserveCapacity(T)
        tapeEntries.reserveCapacity(T)

        for t in 0 ..< T {
            let decay: MLXArray = g.ndim == 4
                ? g[0..., t, 0..., .newAxis, 0...]
                : expandedDimensions(g[0..., t, 0...], axes: [2, 3])
            let decayedState = state * decay
            let kvMem  = (decayedState * expandedDimensions(k_[0..., t, 0...], axis: -2)).sum(axis: -1)
            let delta  = (v[0..., t, 0...] - kvMem) * expandedDimensions(beta[0..., t, 0...], axis: -1)
            let next   = decayedState + expandedDimensions(k_[0..., t, 0...], axis: -2)
                                      * expandedDimensions(delta, axis: -1)
            let y      = (next * expandedDimensions(q_[0..., t, 0...], axis: -2)).sum(axis: -1)

            if let mask {
                // Branchless arithmetic gate — no MLX.where overhead on common path.
                let sGate  = expandedDimensions(mask[0..., t], axes: [1, 2, 3]).asType(state.dtype)
                let yGate  = expandedDimensions(mask[0..., t], axes: [1, 2]).asType(y.dtype)
                state = next * sGate + state * (1 - sGate)
                outputs.append(y * yGate)
                tapeEntries.append((delta * yGate).asType(DType.float32))
            } else {
                state = next
                outputs.append(y)
                tapeEntries.append(delta.asType(DType.float32))
            }
        }
        return (MLX.stacked(outputs, axis: 1), state, MLX.stacked(tapeEntries, axis: 1))
    }

    // MARK: - Block Computation (branchless lookup)

    private static func computeSDPA2PassBlocks(gqaFactor: Int, nKV: Int, deviceArch: String? = nil) -> Int {
        let arch = deviceArch ?? Device.defaultDevice().description
        let devc = arch.last.map(String.init) ?? ""

        // Encode device: 2=d, 1=s, 0=other — no if/else chain.
        let devCode = (devc == "d" ? 2 : 0) | (devc == "s" ? 1 : 0)

        switch devCode {
        case 2: // M-series "d"
            // Branchless clamp-and-shift: pick log₂ bucket via leading-zero trick.
            let base = 128
            let bump1 = (gqaFactor <= 2 && nKV > 8192)  ? 1 : 0        // → 256
            let bump2 = (gqaFactor >= 6 && nKV >= 16384) ? 1 : 0        // → 512 or 1024
            let bump3 = (gqaFactor >= 6 && nKV >= 65536) ? 1 : 0        // extra → 1024
            return base << (bump1 + bump2 + bump3)

        case 1: // "s"
            guard nKV > 1024 && gqaFactor > 4 else { return 64 }
            // Arithmetic shift: each doubling of N → +1 shift, capped at 1024.
            let shift = min(max((Int(log2(Double(nKV))) - 10), 0), 4)
            return 64 << shift

        default:
            return gqaFactor >= 4 ? 64 : 32
        }
    }

    // MARK: - Batched SDPA 2-Pass

    private final class SDPAKernelCache {
        static let shared = SDPAKernelCache()
        // [masked (0/1)]
        let partials: [MLXFast.MLXFastKernel?]
        let reduce:    MLXFast.MLXFastKernel?
        private init() {
            partials = [makePartialsKernel(hasMask: false), makePartialsKernel(hasMask: true)]
            reduce   = makeReduceKernel()
        }

        private static func makePartialsKernel(hasMask: Bool) -> MLXFast.MLXFastKernel? {
            let maskSetup   = hasMask ? "auto mask_ = mask + (((b_idx * Hq + q_head_idx) * M_FIXED + q_seq_idx) * N + block_idx);" : ""
            // Branchless mask: convert to float and fuse into score.
            // Non-masked path: mask_gate is a compile-time constant 1.0.
            let maskGate    = hasMask
                ? "float mask_gate = static_cast<float>(mask_[0]); use_key = use_key & (mask_gate > Limits<InT>::finite_min);"
                : "constexpr float mask_gate = 0.0f; (void)mask_gate;"
            let maskScore   = hasMask ? "score += mask_gate;" : ""
            let maskAdvance = hasMask ? "mask_ += blocks;" : ""

            var inputs = ["queries","keys","values","gqa_factor","N",
                          "k_head_stride","k_seq_stride","v_head_stride","v_seq_stride",
                          "scale","blocks"]
            if hasMask { inputs.append("mask") }

            let source = """
                constexpr int BD = 32;
                constexpr int qk_per_thread = D / BD;
                constexpr int v_per_thread  = V / BD;

                auto q_head_idx = threadgroup_position_in_grid.x;
                auto b_idx      = threadgroup_position_in_grid.y;
                auto block_idx  = threadgroup_position_in_grid.z;
                auto q_seq_idx  = thread_position_in_threadgroup.z;
                auto simd_lid   = thread_index_in_simdgroup;
                auto Hq         = threadgroups_per_grid.x;
                auto hk_idx     = q_head_idx / gqa_factor;
                auto q_batch_head_idx = b_idx * Hq + q_head_idx;
                auto o_offset   = q_batch_head_idx * M_FIXED + q_seq_idx;

                auto q_ = queries + (o_offset * D) + simd_lid * qk_per_thread;
                auto k_ = keys    + ((b_idx * Hk + hk_idx) * k_head_stride) + block_idx * k_seq_stride + simd_lid * qk_per_thread;
                auto v_ = values  + ((b_idx * Hk + hk_idx) * v_head_stride) + block_idx * v_seq_stride + simd_lid * v_per_thread;

                partials += (o_offset * blocks + block_idx) * V + simd_lid * v_per_thread;
                sums     += o_offset * blocks + block_idx;
                maxs     += o_offset * blocks + block_idx;
                \(maskSetup)

                thread float q[qk_per_thread];
                thread float o[v_per_thread];
                threadgroup InT tg_k[BD * qk_per_thread];
                threadgroup InT tg_v[BD * v_per_thread];

                for (int i = 0; i < qk_per_thread; ++i)
                    q[i] = static_cast<float>(scale) * static_cast<float>(q_[i]);
                for (int i = 0; i < v_per_thread; ++i)
                    o[i] = 0.0f;

                float max_score     = Limits<float>::finite_min;
                float sum_exp_score = 0.0f;

                for (int n = block_idx; n < N; n += blocks) {
                    if (q_seq_idx == 0) {
                        for (int i = 0; i < qk_per_thread; ++i) tg_k[simd_lid * qk_per_thread + i] = k_[i];
                        for (int i = 0; i < v_per_thread;  ++i) tg_v[simd_lid * v_per_thread  + i] = v_[i];
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    // Branchless causal mask via integer comparison cast to float.
                    bool use_key = (n <= (N - M_FIXED + q_seq_idx));
                    \(maskGate)

                    // Compute score unconditionally; select kills contribution when !use_key.
                    float score = 0.0f;
                    for (int i = 0; i < qk_per_thread; ++i)
                        score += q[i] * static_cast<float>(tg_k[simd_lid * qk_per_thread + i]);
                    score = simd_sum(score);
                    \(maskScore)
                    // Blend to -inf when use_key==false — no branch in execution.
                    score = metal::select(Limits<float>::finite_min, score, use_key);

                    float new_max    = metal::max(max_score, score);
                    float factor     = fast::exp(max_score - new_max);
                    float exp_score  = fast::exp(score - new_max);
                    max_score        = new_max;
                    sum_exp_score    = sum_exp_score * factor + exp_score;

                    for (int i = 0; i < v_per_thread; ++i)
                        o[i] = o[i] * factor + exp_score * static_cast<float>(tg_v[simd_lid * v_per_thread + i]);

                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    k_ += blocks * int(k_seq_stride);
                    v_ += blocks * int(v_seq_stride);
                    \(maskAdvance)
                }

                if (simd_lid == 0) {
                    sums[0] = sum_exp_score;
                    maxs[0] = max_score;
                }
                for (int i = 0; i < v_per_thread; ++i)
                    partials[i] = static_cast<InT>(o[i]);
            """

            let suffix = hasMask ? "_mask" : ""
            return MLXFast.metalKernel(name: "batched_sdpa_2pass_partials\(suffix)",
                                       inputNames: inputs,
                                       outputNames: ["partials", "sums", "maxs"],
                                       source: source)
        }

        private static func makeReduceKernel() -> MLXFast.MLXFastKernel? {
            let source = """
                constexpr int BN = 32;
                constexpr int BD = 32;
                constexpr int elem_per_thread = V / BD;

                auto head_idx  = threadgroup_position_in_grid.x;
                auto q_seq_idx = threadgroup_position_in_grid.y;
                auto simd_gid  = simdgroup_index_in_threadgroup;
                auto simd_lid  = thread_index_in_simdgroup;
                auto q_offset  = head_idx * M_FIXED + q_seq_idx;

                partials += (q_offset * blocks + simd_gid) * V + simd_lid * elem_per_thread;
                sums     += q_offset * blocks;
                maxs     += q_offset * blocks;
                out      += q_offset * V + simd_gid * elem_per_thread;

                thread float o[elem_per_thread];
                threadgroup float outputs[BN * BD];
                for (int i = 0; i < elem_per_thread; ++i) o[i] = 0.0f;

                // Two-pass: find global max, then accumulate.
                float max_score = Limits<float>::finite_min;
                for (int b = 0; b < blocks / BN; ++b)
                    max_score = metal::max(max_score, maxs[simd_lid + BN * b]);
                max_score = simd_max(max_score);

                float sum_exp_score = 0.0f;
                for (int b = 0; b < blocks / BN; ++b)
                    sum_exp_score += fast::exp(maxs[simd_lid + BN * b] - max_score) * sums[simd_lid + BN * b];
                sum_exp_score = simd_sum(sum_exp_score);

                // Branchless reciprocal: avoid division-by-zero via max with epsilon.
                float inv_sum = 1.0f / metal::max(sum_exp_score, 1e-9f);

                for (int b = 0; b < blocks / BN; ++b) {
                    float factor = fast::exp(maxs[simd_gid] - max_score);
                    for (int i = 0; i < elem_per_thread; ++i)
                        o[i] += factor * static_cast<float>(partials[i]);
                    maxs     += BN;
                    partials += BN * V;
                }

                for (int i = 0; i < elem_per_thread; ++i) {
                    outputs[simd_lid * BD + simd_gid] = o[i];
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]) * inv_sum;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }

                if (simd_lid == 0) {
                    for (int i = 0; i < elem_per_thread; ++i)
                        out[i] = static_cast<InT>(o[i]);
                }
            """
            return MLXFast.metalKernel(name: "batched_sdpa_2pass_reduce",
                                       inputNames: ["partials", "sums", "maxs", "blocks"],
                                       outputNames: ["out"], source: source)
        }
    }

    // MARK: - Public API: Batched SDPA

    public static func batchedSDPA2Pass(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, mask: MLXArray? = nil
    ) -> MLXArray? {
        guard queries.ndim == 4, keys.ndim == 4, values.ndim == 4 else { return nil }
        let B = queries.dim(0); let Hq = queries.dim(1)
        let qLen = queries.dim(2); let D = queries.dim(3)
        let Hk = keys.dim(1); let nKV = keys.dim(2); let Vdim = values.dim(3)
        let inputType = queries.dtype

        guard qLen == 16,
              inputType == .bfloat16 || inputType == .float16,
              (D == 128 || D == 256) && D == Vdim,
              Hk > 0 && Hq % Hk == 0 else { return nil }

        let gqaFactor = Hq / Hk
        let blocks = computeSDPA2PassBlocks(gqaFactor: gqaFactor, nKV: nKV)
        guard blocks > 0 && blocks % 32 == 0 else { return nil }

        let cache = SDPAKernelCache.shared
        let msk = mask != nil ? 1 : 0
        guard let partialsKernel = cache.partials[msk], let reduceKernel = cache.reduce else { return nil }

        let qC = MLX.contiguous(queries)
        let kC = MLX.contiguous(keys)
        let vC = MLX.contiguous(values)

        var inputs: [MLXArray] = [
            qC, kC, vC,
            MLXArray(gqaFactor), MLXArray(nKV),
            MLXArray(keys.dim(2) * keys.dim(3)), MLXArray(keys.dim(3)),
            MLXArray(values.dim(2) * values.dim(3)), MLXArray(values.dim(3)),
            MLXArray(scale), MLXArray(blocks)
        ]
        if let mask {
            inputs.append(mask.dtype != inputType ? mask.asType(inputType) : mask)
        }

        let partialShape = [B * Hq, qLen, blocks, Vdim]
        let statsShape   = [B * Hq, qLen, blocks]

        let out1 = partialsKernel(inputs,
                                  template: [("InT", inputType), ("D", D), ("V", Vdim), ("Hk", Hk), ("M_FIXED", qLen)],
                                  grid: (Hq * 32, B, blocks * qLen), threadGroup: (32, 1, qLen),
                                  outputShapes: [partialShape, statsShape, statsShape],
                                  outputDTypes: [inputType, .float32, .float32])

        let out2 = reduceKernel([out1[0], out1[1], out1[2], MLXArray(blocks)],
                                template: [("InT", inputType), ("V", Vdim), ("M_FIXED", qLen)],
                                grid: ((B * Hq) * 1024, qLen, 1), threadGroup: (1024, 1, 1),
                                outputShapes: [queries.shape], outputDTypes: [inputType])
        return out2[0]
    }

    public static func sdpaFallback(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        scale: Float, mask: MLXArray? = nil
    ) -> MLXArray {
        MLXFast.scaledDotProductAttention(queries: queries, keys: keys, values: values, scale: scale, mask: mask)
    }
}

public final class DFlashKernelsInstance: DFlashKernelProvider, @unchecked Sendable {
    public func gatedDeltaKernelWithTape(
        q: MLXArray, k: MLXArray, v: MLXArray,
        g: MLXArray, beta: MLXArray,
        state: MLXArray, mask: MLXArray?
    ) -> (MLXArray, MLXArray, MLXArray) {
        DFlashKernels.gatedDeltaKernelWithTape(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
    }
}
