// Copyright © 2026 SharpAI
// moe_stream_op.cpp
// Custom MLX Operation that combines GatherMM with SSD Streaming

#include "mlx/core/moe_stream_op.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <atomic>
#include "mlx/primitives.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"

// Static SSD metric trackers for aggregate logging
static std::atomic<size_t>   g_total_bytes_read{0};
static std::atomic<uint64_t> g_total_read_ns{0};
static std::atomic<size_t>   g_read_count{0};
static std::atomic<uint64_t> g_last_log_ns{0};

// Lifetime totals: never reset, read by /metrics endpoint
static std::atomic<uint64_t> g_lifetime_bytes{0};
static std::atomic<uint64_t> g_lifetime_ns{0};
static std::atomic<uint64_t> g_lifetime_chunks{0};
// Last-window throughput (MB/s) — written after each 10s window closes
static std::atomic<double>   g_window_throughput_mbs{0.0};

// TurboKV aggregate counters — incremented from Swift on every compression event.
// Read + reset each 10-second window alongside the SSD stats.
static std::atomic<uint64_t> g_turbo_tokens{0};
static std::atomic<uint64_t> g_turbo_bytes_orig{0};
static std::atomic<uint64_t> g_turbo_bytes_packed{0};

extern "C" void mlx_turbo_kv_record(uint64_t tokens, uint64_t orig_bytes, uint64_t packed_bytes) {
    g_turbo_tokens.fetch_add(tokens, std::memory_order_relaxed);
    g_turbo_bytes_orig.fetch_add(orig_bytes, std::memory_order_relaxed);
    g_turbo_bytes_packed.fetch_add(packed_bytes, std::memory_order_relaxed);
}



namespace mlx::core {

class LoadSSDExpert : public Primitive {
public:
    LoadSSDExpert(
        Stream s,
        uint32_t active_expert,
        std::shared_ptr<fast::SSDStreamer> streamer,
        const std::vector<off_t>& expert_offsets
    ) : Primitive(s), active_expert_(active_expert), streamer_(streamer), expert_offsets_(expert_offsets) {}
    
    void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
        eval_impl(inputs, outputs);
        
        auto& d = metal::device(mlx::core::Device::gpu);
        d.add_temporary(outputs[0], stream().index);
    }
    
    void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
        eval_impl(inputs, outputs);
    }
    
    void eval_impl(const std::vector<array>& inputs, std::vector<array>& outputs) {
        auto& o = outputs[0];
        
        uint32_t active_expert = active_expert_;
        if (active_expert + 1 >= expert_offsets_.size()) {
            throw std::runtime_error("[LoadSSDExpert] Expert index out of bounds.");
        }
        
        off_t block_offset = expert_offsets_[active_expert];
        size_t matrix_bytes = static_cast<size_t>(expert_offsets_[active_expert + 1] - block_offset);

        // We use MLX's allocator to get Metal-accessible (unified) memory.
        o.set_data(allocator::malloc(matrix_bytes));

        auto start_read = std::chrono::high_resolution_clock::now();
        streamer_->load_sync(block_offset, matrix_bytes, o.data<void>());
        auto end_read = std::chrono::high_resolution_clock::now();

        // Accumulate into lifetime totals (never reset)
        g_lifetime_bytes  += matrix_bytes;
        g_lifetime_ns     += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_read - start_read).count());
        g_lifetime_chunks += 1;

        // ─────────────────────────────────────────────────────────────────────
        // AGGREGATE LOGGING — 10-second metric intervals, printed to stderr so
        // the metric lines never interleave with the stdout token stream.
        // ─────────────────────────────────────────────────────────────────────
        g_total_bytes_read += matrix_bytes;
        g_total_read_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end_read - start_read).count();
        g_read_count++;

        auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        uint64_t last = g_last_log_ns.load();

        // 10-second throttle (was 1 s) — keeps metrics visible without flooding
        if (now_ns - last >= 10'000'000'000ULL) {
            if (g_last_log_ns.compare_exchange_strong(last, now_ns)) {
                size_t count  = g_read_count.exchange(0);
                size_t bytes  = g_total_bytes_read.exchange(0);
                uint64_t ns_t = g_total_read_ns.exchange(0);
                if (count > 0 && ns_t > 0) {
                    // True throughput: total bytes / total wall-clock read time
                    double elapsed_s  = ns_t / 1e9;
                    double throughput_mbs = (bytes / (1024.0 * 1024.0)) / elapsed_s;
                    double avg_ms_per_chunk = (ns_t / 1'000'000.0) / count;
                    // Persist for /metrics endpoint (non-resetting)
                    g_window_throughput_mbs.store(throughput_mbs, std::memory_order_relaxed);
                    // Print to stderr — never touches the stdout token stream
                    std::cerr << "[⚡️ SSD Stream] "
                              << std::fixed << std::setprecision(0);
                    std::cerr << throughput_mbs << " MB/s | "
                              << count << " chunks | avg "
                              << std::setprecision(3) << avg_ms_per_chunk << " ms/chunk";

                    // Append TurboKV window stats if active.
                    // Token count is NOT shown — it is multiplied by num_kv_layers
                    // and would be misleading. Ratio and bytes saved are layer-agnostic
                    // because the layer factor cancels in orig/packed.
                    uint64_t tkv_tokens = g_turbo_tokens.exchange(0);  // reset only
                    uint64_t tkv_orig   = g_turbo_bytes_orig.exchange(0);
                    uint64_t tkv_packed = g_turbo_bytes_packed.exchange(0);
                    (void)tkv_tokens;
                    if (tkv_packed > 0 && tkv_orig > tkv_packed) {
                        double ratio    = (double)tkv_orig / tkv_packed;
                        double saved_mb = (tkv_orig - tkv_packed) / 1048576.0;
                        std::cerr << std::fixed << std::setprecision(1)
                                  << " | \U0001f5dc TurboKV " << ratio << "x"
                                  << " (" << std::setprecision(0) << saved_mb << "MB saved)";
                    }
                    std::cerr << std::endl;


                }
            }
        }
    }
    
    std::vector<array> vjp(
        const std::vector<array>& inputs,
        const std::vector<array>& cotangents,
        const std::vector<int>& argnums,
        const std::vector<array>& outputs) override {
        throw std::runtime_error("[LoadSSDExpert] backward pass (VJP) is unsupported.");
    }

    std::vector<array> jvp(
        const std::vector<array>& inputs,
        const std::vector<array>& tangents,
        const std::vector<int>& argnums) override {
        throw std::runtime_error("[LoadSSDExpert] backward pass (JVP) is unsupported.");
    }
    
    bool is_equivalent(const Primitive& other) const override {
        return false;
    }
    
    const char* name() const override {
        return "LoadSSDExpert";
    }
    
private:
    uint32_t active_expert_;
    std::shared_ptr<fast::SSDStreamer> streamer_;
    std::vector<off_t> expert_offsets_;
};

MLX_API array streamed_gather_mm(
    const array& x, // Ignored logic-wise, kept for ABI signature mapping in fast.cpp
    const array& w_shape,
    uint32_t active_expert,
    std::shared_ptr<fast::SSDStreamer> streamer,
    const std::vector<off_t>& expert_offsets,
    StreamOrDevice s
) {
    // Output shape: [1, outputDims, inputDims]
    auto OD = w_shape.shape(1);
    auto ID = w_shape.shape(2);
    
    return array(
        {1, static_cast<int>(OD), static_cast<int>(ID)}, uint32,
        std::make_unique<LoadSSDExpert>(to_stream(s), active_expert, streamer, expert_offsets),
        {x} // MUST pass a dummy input or the node gets eliminated!
    );
}

} // namespace mlx::core

// ── C-ABI metrics accessor (no name mangling — callable from Swift) ───────────
// Define the struct here in the TU that owns it; fast.h has the matching
// typedef so Swift and external C consumers see layout-compatible definitions.
struct MlxSSDMetricsSnapshot {
    double   throughput_mb_per_s;
    uint64_t total_bytes_read;
    uint64_t total_chunks;
    double   avg_chunk_latency_ms;
};

extern "C" void mlx_ssd_metrics_snapshot(struct MlxSSDMetricsSnapshot* out) {
    if (!out) return;
    uint64_t bytes  = g_lifetime_bytes.load(std::memory_order_acquire);
    uint64_t ns     = g_lifetime_ns.load(std::memory_order_acquire);
    uint64_t chunks = g_lifetime_chunks.load(std::memory_order_acquire);
    double   tput   = g_window_throughput_mbs.load(std::memory_order_acquire);

    out->total_bytes_read     = bytes;
    out->total_chunks         = chunks;
    out->throughput_mb_per_s  = tput;
    out->avg_chunk_latency_ms = (ns > 0 && chunks > 0)
        ? (ns / 1'000'000.0) / static_cast<double>(chunks)
        : 0.0;
}
