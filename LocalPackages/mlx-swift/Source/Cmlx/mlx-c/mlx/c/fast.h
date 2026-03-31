/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_FAST_H
#define MLX_FAST_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "mlx/c/array.h"
#include "mlx/c/closure.h"
#include "mlx/c/distributed_group.h"
#include "mlx/c/io_types.h"
#include "mlx/c/map.h"
#include "mlx/c/stream.h"
#include "mlx/c/string.h"
#include "mlx/c/vector.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup fast Fast custom operations
 */
/**@{*/

typedef struct mlx_fast_cuda_kernel_config_ {
  void* ctx;
} mlx_fast_cuda_kernel_config;
mlx_fast_cuda_kernel_config mlx_fast_cuda_kernel_config_new(void);
void mlx_fast_cuda_kernel_config_free(mlx_fast_cuda_kernel_config cls);

int mlx_fast_cuda_kernel_config_add_output_arg(
    mlx_fast_cuda_kernel_config cls,
    const int* shape,
    size_t size,
    mlx_dtype dtype);
int mlx_fast_cuda_kernel_config_set_grid(
    mlx_fast_cuda_kernel_config cls,
    int grid1,
    int grid2,
    int grid3);
int mlx_fast_cuda_kernel_config_set_thread_group(
    mlx_fast_cuda_kernel_config cls,
    int thread1,
    int thread2,
    int thread3);
int mlx_fast_cuda_kernel_config_set_init_value(
    mlx_fast_cuda_kernel_config cls,
    float value);
int mlx_fast_cuda_kernel_config_set_verbose(
    mlx_fast_cuda_kernel_config cls,
    bool verbose);
int mlx_fast_cuda_kernel_config_add_template_arg_dtype(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    mlx_dtype dtype);
int mlx_fast_cuda_kernel_config_add_template_arg_int(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    int value);
int mlx_fast_cuda_kernel_config_add_template_arg_bool(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    bool value);

typedef struct mlx_fast_cuda_kernel_ {
  void* ctx;
} mlx_fast_cuda_kernel;

mlx_fast_cuda_kernel mlx_fast_cuda_kernel_new(
    const char* name,
    const mlx_vector_string input_names,
    const mlx_vector_string output_names,
    const char* source,
    const char* header,
    bool ensure_row_contiguous,
    int shared_memory);

void mlx_fast_cuda_kernel_free(mlx_fast_cuda_kernel cls);

int mlx_fast_cuda_kernel_apply(
    mlx_vector_array* outputs,
    mlx_fast_cuda_kernel cls,
    const mlx_vector_array inputs,
    const mlx_fast_cuda_kernel_config config,
    const mlx_stream stream);

int mlx_fast_layer_norm(
    mlx_array* res,
    const mlx_array x,
    const mlx_array weight /* may be null */,
    const mlx_array bias /* may be null */,
    float eps,
    const mlx_stream s);

typedef struct mlx_fast_metal_kernel_config_ {
  void* ctx;
} mlx_fast_metal_kernel_config;
mlx_fast_metal_kernel_config mlx_fast_metal_kernel_config_new(void);
void mlx_fast_metal_kernel_config_free(mlx_fast_metal_kernel_config cls);

int mlx_fast_metal_kernel_config_add_output_arg(
    mlx_fast_metal_kernel_config cls,
    const int* shape,
    size_t size,
    mlx_dtype dtype);
int mlx_fast_metal_kernel_config_set_grid(
    mlx_fast_metal_kernel_config cls,
    int grid1,
    int grid2,
    int grid3);
int mlx_fast_metal_kernel_config_set_thread_group(
    mlx_fast_metal_kernel_config cls,
    int thread1,
    int thread2,
    int thread3);
int mlx_fast_metal_kernel_config_set_init_value(
    mlx_fast_metal_kernel_config cls,
    float value);
int mlx_fast_metal_kernel_config_set_verbose(
    mlx_fast_metal_kernel_config cls,
    bool verbose);
int mlx_fast_metal_kernel_config_add_template_arg_dtype(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    mlx_dtype dtype);
int mlx_fast_metal_kernel_config_add_template_arg_int(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    int value);
int mlx_fast_metal_kernel_config_add_template_arg_bool(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    bool value);

typedef struct mlx_fast_metal_kernel_ {
  void* ctx;
} mlx_fast_metal_kernel;

mlx_fast_metal_kernel mlx_fast_metal_kernel_new(
    const char* name,
    const mlx_vector_string input_names,
    const mlx_vector_string output_names,
    const char* source,
    const char* header,
    bool ensure_row_contiguous,
    bool atomic_outputs);

void mlx_fast_metal_kernel_free(mlx_fast_metal_kernel cls);

int mlx_fast_metal_kernel_apply(
    mlx_vector_array* outputs,
    mlx_fast_metal_kernel cls,
    const mlx_vector_array inputs,
    const mlx_fast_metal_kernel_config config,
    const mlx_stream stream);

int mlx_fast_rms_norm(
    mlx_array* res,
    const mlx_array x,
    const mlx_array weight /* may be null */,
    float eps,
    const mlx_stream s);
int mlx_fast_rope(
    mlx_array* res,
    const mlx_array x,
    int dims,
    bool traditional,
    mlx_optional_float base,
    float scale,
    int offset,
    const mlx_array freqs /* may be null */,
    const mlx_stream s);
int mlx_fast_rope_dynamic(
    mlx_array* res,
    const mlx_array x,
    int dims,
    bool traditional,
    mlx_optional_float base,
    float scale,
    const mlx_array offset,
    const mlx_array freqs /* may be null */,
    const mlx_stream s);
int mlx_fast_scaled_dot_product_attention(
    mlx_array* res,
    const mlx_array queries,
    const mlx_array keys,
    const mlx_array values,
    float scale,
    const char* mask_mode,
    const mlx_array mask_arr /* may be null */,
    const mlx_stream s);

int mlx_fast_streamed_gather_mm(
    mlx_array* res,
    const mlx_array x,
    const mlx_array w_shape,
    const mlx_array expert_indices,
    const char* safetensors_path,
    const char* tensor_name,
    const mlx_stream s);

int mlx_fast_turbo_encode(
    mlx_array* res_polar_k,
    mlx_array* res_polar_v,
    mlx_array* res_residual_k,
    mlx_array* res_residual_v,
    const mlx_array keys,
    const mlx_array values,
    int k_bits,
    const mlx_stream s);

int mlx_fast_turbo_decode_k(
    mlx_array* res,
    const mlx_array packed,
    const mlx_stream s);

int mlx_fast_turbo_decode_v(
    mlx_array* res,
    const mlx_array packed,
    const mlx_stream s);

int mlx_fast_prefault(mlx_array x);


// pread() directly into the already-evaluated MLX array's unified memory buffer.
// This gives full NVMe sequential throughput without OS page-fault overhead.
// The array MUST already be evaluated (concrete pointer exists).
// safetensors_path: full path to .safetensors file
// tensor_name: e.g. "model.layers.0.mlp.experts.gate_proj.weight"
// expert_index: 0-based index of the expert to read
int mlx_fast_pread_into(
    mlx_array dst,
    const char* safetensors_path,
    const char* tensor_name,
    uint32_t expert_index);

/**@}*/

// ── SSD Flash-Stream metrics snapshot ────────────────────────────────────────
// Cumulative NVMe throughput stats since process start.
// Call mlx_ssd_metrics_snapshot() from any thread to read without resetting counters.

typedef struct MlxSSDMetricsSnapshot {
    double   throughput_mb_per_s;  /* 10-s rolling window average (0 before first window) */
    uint64_t total_bytes_read;     /* Lifetime bytes read from SSD */
    uint64_t total_chunks;         /* Lifetime expert chunks loaded */
    double   avg_chunk_latency_ms; /* Lifetime average per-chunk latency (ms) */
} MlxSSDMetricsSnapshot;

void mlx_ssd_metrics_snapshot(MlxSSDMetricsSnapshot* out);

#ifdef __cplusplus
}
#endif

#endif
