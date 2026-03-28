// Copyright © 2026 SharpAI
// turbo_quant.h
// MLX Core Fast primitive definitions for TurboQuant KV Cache compression.

#pragma once

#include "mlx/array.h"

namespace mlx::core::fast {

/**
 * TurboQuant encoded Key Value Storage structures holding PolarQuant coordinates
 * and QJL residuals natively. 
 */
struct TurboQuantKV {
    // 3-bit PolarQuant encoded arrays
    array polar_keys;
    array polar_values;
    
    // QJL residual correction layers for extreme outlier dots
    array qjl_key_residuals;
    array qjl_value_residuals;

    // Decoding dimensions
    int head_dim;
    int k_dim;
};

/**
 * Encode an FP16/BF16 KV sequence chunk into the TurboQuant format.
 */
TurboQuantKV turbo_encode(
    const array& keys,
    const array& values,
    int k_bits = 3,
    StreamOrDevice s = {}
);

/**
 * Decode TurboQuant directly back into FP16 (primarily for debugging, 
 * since the real decompression happens directly inside the Attention Metal Shader).
 */
std::pair<array, array> turbo_decode(
    const TurboQuantKV& compressed_kv,
    StreamOrDevice s = {}
);

} // namespace mlx::core::fast
