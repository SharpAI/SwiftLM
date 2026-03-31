// Copyright © 2026 SharpAI
// moe_stream.metal
// Custom Metal compute kernel for Async SSD Streaming MoE GEMM.
// Reads raw 4-bit packed bytes from a pinned MTLBuffer written by GCD dispatch_io.

#include <metal_stdlib>

using namespace metal;

// Kernel for streaming expert matrix multiply.
// W is 4-bit quantized, stored as [K/8, N] uint32 per expert.
// We unpack to float32 and compute output = X * W.T
// Thread grid: [N, M] where M=num_tokens, N=outputDims.
kernel void streamed_moe_gemm(
    const device uint16_t* x     [[buffer(0)]],  // input [M, K] as bf16 raw ints
    const device uint32_t* w     [[buffer(1)]],  // weights 4-bit packed per expert 
    device float* out            [[buffer(2)]],  // output [M, N] as float32
    constant uint& M             [[buffer(3)]],  // num_tokens
    constant uint& K             [[buffer(4)]],  // input dim (hidden_size)  
    constant uint& N             [[buffer(5)]],  // output dim (expert_dim)
    uint2 gid                    [[thread_position_in_grid]])
{
    uint row = gid.y; // token index
    uint col = gid.x; // output feature index

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    
    // Each uint32 in W packs 8 x 4-bit values
    // W shape is [N, K/8] in packed form
    uint packed_K = (K + 7) / 8;
    
    for (uint block = 0; block < packed_K; ++block) {
        // Load packed 4-bit weights for (col, block)
        uint32_t packed = w[col * packed_K + block];
        
        // Unpack 8 x 4-bit values, scale to [-8, 7] / 8.0
        for (uint b = 0; b < 8; ++b) {
            uint ki = block * 8 + b;
            if (ki >= K) break;
            
            // Extract 4-bit nibble, sign extend from 4 bits
            int nibble = (int)((packed >> (b * 4)) & 0xF);
            if (nibble >= 8) nibble -= 16;  // sign extend
            float w_val = (float)nibble / 8.0f;
            
            // X is bf16: reinterpret uint16 as float via bit conversion
            uint16_t x_raw = x[row * K + ki];
            uint32_t x_bits = ((uint32_t)x_raw) << 16;
            float x_val = as_type<float>(x_bits);
            
            sum += x_val * w_val;
        }
    }
    
    out[row * N + col] = sum;
}
