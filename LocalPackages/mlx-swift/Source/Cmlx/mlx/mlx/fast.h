// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <optional>
#include <variant>

#include "mlx/api.h"
#include "mlx/utils.h"

namespace mlx::core::fast {

MLX_API array rms_norm(
    const array& x,
    const std::optional<array>& weight,
    float eps,
    StreamOrDevice s = {});

MLX_API array layer_norm(
    const array& x,
    const std::optional<array>& weight,
    const std::optional<array>& bias,
    float eps,
    StreamOrDevice s = {});

MLX_API array rope(
    const array& x,
    int dims,
    bool traditional,
    std::optional<float> base,
    float scale,
    int offset,
    const std::optional<array>& freqs = std::nullopt,
    StreamOrDevice s = {});

MLX_API array rope(
    const array& x,
    int dims,
    bool traditional,
    std::optional<float> base,
    float scale,
    const array& offset,
    const std::optional<array>& freqs = std::nullopt,
    StreamOrDevice s = {});

/** Computes: O = softmax(Q @ K.T) @ V **/
MLX_API array scaled_dot_product_attention(
    const array& queries,
    const array& keys,
    const array& values,
    const float scale,
    const std::string& mask_mode = "",
    std::optional<array> mask_arr = {},
    const std::optional<array>& sinks = {},
    StreamOrDevice s = {});

using TemplateArg = std::variant<int, bool, Dtype>;
using ScalarArg = std::variant<bool, int, float>;

using CustomKernelFunction = std::function<std::vector<array>(
    const std::vector<array>&,
    const std::vector<Shape>&,
    const std::vector<Dtype>&,
    std::tuple<int, int, int>,
    std::tuple<int, int, int>,
    std::vector<std::pair<std::string, TemplateArg>>,
    std::optional<float>,
    bool,
    StreamOrDevice)>;

MLX_API CustomKernelFunction metal_kernel(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header = "",
    bool ensure_row_contiguous = true,
    bool atomic_outputs = false);

MLX_API CustomKernelFunction cuda_kernel(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header = "",
    bool ensure_row_contiguous = true,
    int shared_memory = 0);

MLX_API std::vector<array> precompiled_cuda_kernel(
    const std::string& name,
    const std::string& compiled_source,
    const std::vector<array>& inputs,
    const std::vector<Shape>& output_shapes,
    const std::vector<Dtype>& output_dtypes,
    const std::vector<ScalarArg>& scalars,
    std::tuple<int, int, int> grid,
    std::tuple<int, int, int> threadgroup,
    int shared_memory = 0,
    std::optional<float> init_value = std::nullopt,
    bool ensure_row_contiguous = false,
    StreamOrDevice s = {});

/**
 * Compress a K-cache tensor to TurboQuant format (3-bit PolarQuant + 1-bit QJL).
 *
 * keys: [batch, heads, seq, 128] — fp16 / bf16 / fp32
 * returns: uint8 array with the same leading dims and last dim = 68
 *          Layout per token: indices[48] | qjl_signs[16] | norm_fp16[2] | rnorm_fp16[2]
 */
MLX_API array turbo_encode_k(const array& keys, StreamOrDevice s = {});

/**
 * Compress a V-cache tensor to TurboQuant format (3-bit PolarQuant only).
 *
 * values: [batch, heads, seq, 128] — fp16 / bf16 / fp32
 * returns: uint8 array with the same leading dims and last dim = 50
 *          Layout per token: indices[48] | norm_fp16[2]
 */
MLX_API array turbo_encode_v(const array& values, StreamOrDevice s = {});

/**
 * Decode TurboKV compressed K-cache back to float32.
 *
 * packed: uint8 with last dim 68 (D=128) or 136 (D=256)
 * returns: float32 array with last dim = head_dim (128 or 256)
 */
MLX_API array turbo_decode_k(const array& packed, StreamOrDevice s = {});

/**
 * Decode TurboKV compressed V-cache back to float32.
 *
 * packed: uint8 with last dim 50 (D=128) or 100 (D=256)
 * returns: float32 array with last dim = head_dim (128 or 256)
 */
MLX_API array turbo_decode_v(const array& packed, StreamOrDevice s = {});

} // namespace mlx::core::fast
