// Copyright © 2026 SharpAI
// ssd_streamer.mm

#include "mlx/backend/metal/ssd_streamer.h"
#include "mlx/backend/metal/device.h"
#include <unistd.h>
#include <fcntl.h>
#include <iostream>

namespace mlx::core::fast {

SSDStreamer::SSDStreamer(const std::string& file_path, size_t buffer_size_bytes, int num_buffers)
    : file_path_(file_path), buffer_size_bytes_(buffer_size_bytes), next_buffer_idx_(0) {
    
    // POSIX open file descriptor
    fd_ = open(file_path_.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw std::runtime_error("[SSDStreamer] Failed to open safetensors file: " + file_path_);
    }

    // Grand Central Dispatch IO Queue
    io_queue_ = dispatch_queue_create("com.mlx.ssd_streamer", DISPATCH_QUEUE_CONCURRENT);
    channel_ = dispatch_io_create(DISPATCH_IO_RANDOM, fd_, io_queue_, ^(int error) {
        if (error != 0) {
            std::cerr << "[SSDStreamer] dispatch_io_create failed." << std::endl;
        }
    });
    
    // Allocate Pinned MTLBuffers via MLX backend
    auto& d = metal::device(mlx::core::Device::gpu);
    for (int i = 0; i < num_buffers; i++) {
        auto buffer = std::make_unique<StreamBuffer>();
        // Using MTLResourceStorageModeShared so CPU can write via dispatch_io and GPU can read zero-copy
        buffer->mtl_buffer = d.mtl_device()->newBuffer(buffer_size_bytes, MTL::ResourceStorageModeShared);
        buffer->raw_ptr = buffer->mtl_buffer->contents();
        buffer->size = buffer_size_bytes;
        buffer->is_busy = false;
        buffers_.push_back(std::move(buffer));
    }
}

SSDStreamer::~SSDStreamer() {
    if (channel_) dispatch_io_close(channel_, DISPATCH_IO_STOP);
    if (fd_ >= 0) close(fd_);
    // Release MTLBuffers
    for (auto& buf : buffers_) {
        buf->mtl_buffer->release();
    }
}

StreamBuffer* SSDStreamer::prefetch_async(off_t byte_offset, size_t length) {
    if (length > buffer_size_bytes_) {
        throw std::invalid_argument("[SSDStreamer] Load length exceeds Pinned Buffer capacity.");
    }

    // Simple round-robin for buffer selection (in a robust implementation, use condition variables)
    StreamBuffer* target = buffers_[next_buffer_idx_].get();
    target->is_busy = true;

    // Asynchronously read exactly byte_offset into the MTLBuffer mapped CPU pointer
    dispatch_io_read(channel_, byte_offset, length, io_queue_, ^(bool done, dispatch_data_t data, int error) {
        if (error) {
            std::cerr << "[SSDStreamer] dispatch_io_read failed with error " << error << std::endl;
            return;
        }
        
        if (data) {
            // Copy posix data to the shared explicit metal buffer mapped pointer
            __block size_t offset = 0;
            dispatch_data_apply(data, ^bool(dispatch_data_t region, size_t region_offset, const void *buffer, size_t size) {
                memcpy((uint8_t*)target->raw_ptr + offset, buffer, size);
                offset += size;
                return true;
            });
        }
        
        if (done) {
            target->is_busy = false; // Realistically needs atomics/locks here for thread sync
        }
    });

    next_buffer_idx_ = (next_buffer_idx_ + 1) % buffers_.size();
    return target;
}

void SSDStreamer::wait_until_ready(StreamBuffer* buffer) {
    // Basic polling or semaphore wait to ensure dispatch_io_read hit the 'done' block.
    // In production, we yield or use std::condition_variable to prevent busy-waiting.
    while (buffer->is_busy) {
        usleep(100);
    }
}

void SSDStreamer::release_buffer(StreamBuffer* buffer) {
    buffer->is_busy = false;
}

} // namespace mlx::core::fast
