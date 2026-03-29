// Copyright © 2026 SharpAI
// moe_stream_op.h
// Custom MLX Operation that combines GatherMM with SSD Streaming

#pragma once

#include "mlx/array.h"
#include "mlx/backend/metal/ssd_streamer.h"
#include <memory>
#include "mlx/stream.h"
#include "mlx/utils.h"
#include <memory>
#include <vector>

namespace mlx::core {

/**
 * Perform a Gather Matrix-Multiplication while asynchronously streaming
 * the transposed weights (W1/W3 + W2) directly from an SSD `.safetensors` file.
 * 
 * This graph operation bypasses Apple Unified Memory Page Faults by using 
 * pinned Metal memory paired with `dispatch_io`.
 * 
 * @param inputs Matrix inputs (e.g. x, expert_indices)
 * @param streamer A shared pointer to the SSDStreamer context
 * @param expert_offsets Byte offsets into the safetensor file per expert
 * @param s Stream to run the gathered MM
 */
MLX_API array streamed_gather_mm(
    const array& x,
    const array& w_shape, // We pass shape instead of weights
    uint32_t active_expert,
    std::shared_ptr<fast::SSDStreamer> streamer,
    const std::vector<off_t>& expert_offsets,
    StreamOrDevice s = {}
);

} // namespace mlx::core
