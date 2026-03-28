// Copyright © 2026 SharpAI
// ssd_streamer.h
// Metal-backed zero-copy asynchronous SSD stream architecture for MoE Experts.

#pragma once

#include <string>
#include <memory>
#include <vector>
#include <dispatch/dispatch.h>
#include <Metal/Metal.hpp>

// Forward declaration from MLX allocator
namespace mlx::core::metal {
    class MetalAllocator;
}

namespace mlx::core::fast {

struct StreamBuffer {
    MTL::Buffer* mtl_buffer; // Tied to MTLResourceStorageModeShared
    void* raw_ptr;
    size_t size;
    bool is_busy;
};

class SSDStreamer {
public:
    SSDStreamer(const std::string& file_path, size_t buffer_size_bytes, int num_buffers = 2);
    ~SSDStreamer();

    // Prevent copy/move to maintain descriptor safety
    SSDStreamer(const SSDStreamer&) = delete;
    SSDStreamer& operator=(const SSDStreamer&) = delete;

    /**
     * Dispatch an asynchronous GCD POSIX read from the safetensors file descriptor.
     * @param byte_offset The offset into the .safetensors file.
     * @param length The number of bytes to read (must be <= buffer_size_bytes).
     * @return The StreamBuffer containing the pinned Metal memory, marked as busy.
     */
    StreamBuffer* prefetch_async(off_t byte_offset, size_t length);

    /**
     * Synchronize and wait for a specific buffer's read to complete.
     */
    void wait_until_ready(StreamBuffer* buffer);

    /**
     * Release a buffer back to the streamer pool after the Metal kernel finishes.
     */
    void release_buffer(StreamBuffer* buffer);

private:
    std::string file_path_;
    int fd_;
    dispatch_io_t channel_;
    dispatch_queue_t io_queue_;

    size_t buffer_size_bytes_;
    std::vector<std::unique_ptr<StreamBuffer>> buffers_;

    // To prevent blocking the GPU, we round-robin or track free buffers
    int next_buffer_idx_;
};

} // namespace mlx::core::fast
