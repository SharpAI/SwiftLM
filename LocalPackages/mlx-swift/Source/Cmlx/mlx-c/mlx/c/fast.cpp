/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#include "mlx/c/fast.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"
#include "mlx/array.h"
#include "mlx/fast.h"
#include "mlx/core/moe_stream_op.h"
#include "mlx/fast/turbo_quant.h"

namespace mlx::core {
void prefault(const array& a) {
    if (!a.data_shared_ptr()) return;
    const uint8_t* ptr = static_cast<const uint8_t*>(const_cast<allocator::Buffer&>(a.buffer()).raw_ptr());
    if (!ptr) return;
    
    // volatile to prevent the compiler from optimizing the read away
    volatile uint8_t tmp = 0;
    size_t size = a.buffer_size();
    
    // Read one byte per 16KB page to trigger the OS page fault handler
    // on the CPU, avoiding the 5-second GPU Watchdog timeout.
    for (size_t i = 0; i < size; i += 16384) {
        tmp += ptr[i];
    }
}
}

struct mlx_fast_cuda_kernel_config_cpp_ {
  std::vector<mlx::core::Shape> output_shapes;
  std::vector<mlx::core::Dtype> output_dtypes;
  std::tuple<int, int, int> grid;
  std::tuple<int, int, int> thread_group;
  std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>>
      template_args;
  std::optional<float> init_value;
  bool verbose;
};

inline mlx_fast_cuda_kernel_config mlx_fast_cuda_kernel_config_new_() {
  return mlx_fast_cuda_kernel_config({new mlx_fast_cuda_kernel_config_cpp_()});
}

inline mlx_fast_cuda_kernel_config_cpp_& mlx_fast_cuda_kernel_config_get_(
    mlx_fast_cuda_kernel_config d) {
  if (!d.ctx) {
    throw std::runtime_error(
        "expected a non-empty mlx_fast_cuda_kernel_config");
  }
  return *static_cast<mlx_fast_cuda_kernel_config_cpp_*>(d.ctx);
}

inline void mlx_fast_cuda_kernel_config_free_(mlx_fast_cuda_kernel_config d) {
  if (d.ctx) {
    delete static_cast<mlx_fast_cuda_kernel_config_cpp_*>(d.ctx);
  }
}

extern "C" mlx_fast_cuda_kernel_config mlx_fast_cuda_kernel_config_new(void) {
  try {
    return mlx_fast_cuda_kernel_config_new_();
  } catch (std::exception& e) {
    mlx_error(e.what());
  }
  return {nullptr};
}

extern "C" void mlx_fast_cuda_kernel_config_free(
    mlx_fast_cuda_kernel_config cls) {
  mlx_fast_cuda_kernel_config_free_(cls);
}

extern "C" int mlx_fast_cuda_kernel_config_add_output_arg(
    mlx_fast_cuda_kernel_config cls,
    const int* shape,
    size_t size,
    mlx_dtype dtype) {
  try {
    mlx_fast_cuda_kernel_config_get_(cls).output_shapes.push_back(
        mlx::core::Shape(shape, shape + size));
    mlx_fast_cuda_kernel_config_get_(cls).output_dtypes.push_back(
        mlx_dtype_to_cpp(dtype));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_cuda_kernel_config_set_grid(
    mlx_fast_cuda_kernel_config cls,
    int grid1,
    int grid2,
    int grid3) {
  try {
    mlx_fast_cuda_kernel_config_get_(cls).grid =
        std::make_tuple(grid1, grid2, grid3);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_cuda_kernel_config_set_thread_group(
    mlx_fast_cuda_kernel_config cls,
    int thread1,
    int thread2,
    int thread3) {
  try {
    mlx_fast_cuda_kernel_config_get_(cls).thread_group =
        std::make_tuple(thread1, thread2, thread3);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_cuda_kernel_config_set_init_value(
    mlx_fast_cuda_kernel_config cls,
    float value) {
  try {
    mlx_fast_cuda_kernel_config_get_(cls).init_value = value;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_cuda_kernel_config_set_verbose(
    mlx_fast_cuda_kernel_config cls,
    bool verbose) {
  try {
    mlx_fast_cuda_kernel_config_get_(cls).verbose = verbose;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_cuda_kernel_config_add_template_arg_dtype(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    mlx_dtype dtype) {
  try {
    mlx_fast_cuda_kernel_config_get_(cls).template_args.push_back(
        std::make_pair(std::string(name), mlx_dtype_to_cpp(dtype)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_cuda_kernel_config_add_template_arg_int(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    int value) {
  try {
    mlx_fast_cuda_kernel_config_get_(cls).template_args.push_back(
        std::make_pair(std::string(name), value));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_cuda_kernel_config_add_template_arg_bool(
    mlx_fast_cuda_kernel_config cls,
    const char* name,
    bool value) {
  try {
    mlx_fast_cuda_kernel_config_get_(cls).template_args.push_back(
        std::make_pair(std::string(name), value));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

struct mlx_fast_cuda_kernel_cpp_ {
  mlx::core::fast::CustomKernelFunction mkf;
  mlx_fast_cuda_kernel_cpp_(mlx::core::fast::CustomKernelFunction mkf)
      : mkf(mkf) {};
};

inline mlx_fast_cuda_kernel mlx_fast_cuda_kernel_new_(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header,
    bool ensure_row_contiguous,
    int shared_memory) {
  return mlx_fast_cuda_kernel({new mlx_fast_cuda_kernel_cpp_(
      mlx::core::fast::cuda_kernel(
          name,
          input_names,
          output_names,
          source,
          header,
          ensure_row_contiguous,
          shared_memory))});
}

extern "C" mlx_fast_cuda_kernel mlx_fast_cuda_kernel_new(
    const char* name,
    const mlx_vector_string input_names,
    const mlx_vector_string output_names,
    const char* source,
    const char* header,
    bool ensure_row_contiguous,
    int shared_memory) {
  try {
    return mlx_fast_cuda_kernel_new_(
        name,
        mlx_vector_string_get_(input_names),
        mlx_vector_string_get_(output_names),
        source,
        header,
        ensure_row_contiguous,
        shared_memory);
  } catch (std::exception& e) {
    mlx_error(e.what());
  }
  return {nullptr};
}

inline mlx::core::fast::CustomKernelFunction& mlx_fast_cuda_kernel_get_(
    mlx_fast_cuda_kernel d) {
  if (!d.ctx) {
    throw std::runtime_error("expected a non-empty mlx_fast_cuda_kernel");
  }
  return static_cast<mlx_fast_cuda_kernel_cpp_*>(d.ctx)->mkf;
}

inline void mlx_fast_cuda_kernel_free_(mlx_fast_cuda_kernel d) {
  if (d.ctx) {
    delete static_cast<mlx_fast_cuda_kernel_cpp_*>(d.ctx);
  }
}

extern "C" void mlx_fast_cuda_kernel_free(mlx_fast_cuda_kernel cls) {
  mlx_fast_cuda_kernel_free_(cls);
}

extern "C" int mlx_fast_cuda_kernel_apply(
    mlx_vector_array* outputs,
    mlx_fast_cuda_kernel cls,
    const mlx_vector_array inputs,
    const mlx_fast_cuda_kernel_config config,
    const mlx_stream stream) {
  try {
    auto config_ctx = mlx_fast_cuda_kernel_config_get_(config);
    mlx_vector_array_set_(
        *outputs,
        mlx_fast_cuda_kernel_get_(cls)(
            mlx_vector_array_get_(inputs),
            config_ctx.output_shapes,
            config_ctx.output_dtypes,
            config_ctx.grid,
            config_ctx.thread_group,
            config_ctx.template_args,
            config_ctx.init_value,
            config_ctx.verbose,
            mlx_stream_get_(stream)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_fast_layer_norm(
    mlx_array* res,
    const mlx_array x,
    const mlx_array weight /* may be null */,
    const mlx_array bias /* may be null */,
    float eps,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::fast::layer_norm(
            mlx_array_get_(x),
            (weight.ctx ? std::make_optional(mlx_array_get_(weight))
                        : std::nullopt),
            (bias.ctx ? std::make_optional(mlx_array_get_(bias))
                      : std::nullopt),
            eps,
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

struct mlx_fast_metal_kernel_config_cpp_ {
  std::vector<mlx::core::Shape> output_shapes;
  std::vector<mlx::core::Dtype> output_dtypes;
  std::tuple<int, int, int> grid;
  std::tuple<int, int, int> thread_group;
  std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>>
      template_args;
  std::optional<float> init_value;
  bool verbose;
};

inline mlx_fast_metal_kernel_config mlx_fast_metal_kernel_config_new_() {
  return mlx_fast_metal_kernel_config(
      {new mlx_fast_metal_kernel_config_cpp_()});
}

inline mlx_fast_metal_kernel_config_cpp_& mlx_fast_metal_kernel_config_get_(
    mlx_fast_metal_kernel_config d) {
  if (!d.ctx) {
    throw std::runtime_error(
        "expected a non-empty mlx_fast_metal_kernel_config");
  }
  return *static_cast<mlx_fast_metal_kernel_config_cpp_*>(d.ctx);
}

inline void mlx_fast_metal_kernel_config_free_(mlx_fast_metal_kernel_config d) {
  if (d.ctx) {
    delete static_cast<mlx_fast_metal_kernel_config_cpp_*>(d.ctx);
  }
}

extern "C" mlx_fast_metal_kernel_config mlx_fast_metal_kernel_config_new(void) {
  try {
    return mlx_fast_metal_kernel_config_new_();
  } catch (std::exception& e) {
    mlx_error(e.what());
  }
  return {nullptr};
}

extern "C" void mlx_fast_metal_kernel_config_free(
    mlx_fast_metal_kernel_config cls) {
  mlx_fast_metal_kernel_config_free_(cls);
}

extern "C" int mlx_fast_metal_kernel_config_add_output_arg(
    mlx_fast_metal_kernel_config cls,
    const int* shape,
    size_t size,
    mlx_dtype dtype) {
  try {
    mlx_fast_metal_kernel_config_get_(cls).output_shapes.push_back(
        mlx::core::Shape(shape, shape + size));
    mlx_fast_metal_kernel_config_get_(cls).output_dtypes.push_back(
        mlx_dtype_to_cpp(dtype));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_metal_kernel_config_set_grid(
    mlx_fast_metal_kernel_config cls,
    int grid1,
    int grid2,
    int grid3) {
  try {
    mlx_fast_metal_kernel_config_get_(cls).grid =
        std::make_tuple(grid1, grid2, grid3);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_metal_kernel_config_set_thread_group(
    mlx_fast_metal_kernel_config cls,
    int thread1,
    int thread2,
    int thread3) {
  try {
    mlx_fast_metal_kernel_config_get_(cls).thread_group =
        std::make_tuple(thread1, thread2, thread3);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_metal_kernel_config_set_init_value(
    mlx_fast_metal_kernel_config cls,
    float value) {
  try {
    mlx_fast_metal_kernel_config_get_(cls).init_value = value;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_metal_kernel_config_set_verbose(
    mlx_fast_metal_kernel_config cls,
    bool verbose) {
  try {
    mlx_fast_metal_kernel_config_get_(cls).verbose = verbose;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_metal_kernel_config_add_template_arg_dtype(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    mlx_dtype dtype) {
  try {
    mlx_fast_metal_kernel_config_get_(cls).template_args.push_back(
        std::make_pair(std::string(name), mlx_dtype_to_cpp(dtype)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_metal_kernel_config_add_template_arg_int(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    int value) {
  try {
    mlx_fast_metal_kernel_config_get_(cls).template_args.push_back(
        std::make_pair(std::string(name), value));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_metal_kernel_config_add_template_arg_bool(
    mlx_fast_metal_kernel_config cls,
    const char* name,
    bool value) {
  try {
    mlx_fast_metal_kernel_config_get_(cls).template_args.push_back(
        std::make_pair(std::string(name), value));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

struct mlx_fast_metal_kernel_cpp_ {
  mlx::core::fast::CustomKernelFunction mkf;
  mlx_fast_metal_kernel_cpp_(mlx::core::fast::CustomKernelFunction mkf)
      : mkf(mkf) {};
};

inline mlx_fast_metal_kernel mlx_fast_metal_kernel_new_(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header,
    bool ensure_row_contiguous,
    bool atomic_outputs) {
  return mlx_fast_metal_kernel({new mlx_fast_metal_kernel_cpp_(
      mlx::core::fast::metal_kernel(
          name,
          input_names,
          output_names,
          source,
          header,
          ensure_row_contiguous,
          atomic_outputs))});
}

extern "C" mlx_fast_metal_kernel mlx_fast_metal_kernel_new(
    const char* name,
    const mlx_vector_string input_names,
    const mlx_vector_string output_names,
    const char* source,
    const char* header,
    bool ensure_row_contiguous,
    bool atomic_outputs) {
  try {
    return mlx_fast_metal_kernel_new_(
        name,
        mlx_vector_string_get_(input_names),
        mlx_vector_string_get_(output_names),
        source,
        header,
        ensure_row_contiguous,
        atomic_outputs);
  } catch (std::exception& e) {
    mlx_error(e.what());
  }
  return {nullptr};
}

inline mlx::core::fast::CustomKernelFunction& mlx_fast_metal_kernel_get_(
    mlx_fast_metal_kernel d) {
  if (!d.ctx) {
    throw std::runtime_error("expected a non-empty mlx_fast_metal_kernel");
  }
  return static_cast<mlx_fast_metal_kernel_cpp_*>(d.ctx)->mkf;
}

inline void mlx_fast_metal_kernel_free_(mlx_fast_metal_kernel d) {
  if (d.ctx) {
    delete static_cast<mlx_fast_metal_kernel_cpp_*>(d.ctx);
  }
}

extern "C" void mlx_fast_metal_kernel_free(mlx_fast_metal_kernel cls) {
  mlx_fast_metal_kernel_free_(cls);
}

extern "C" int mlx_fast_metal_kernel_apply(
    mlx_vector_array* outputs,
    mlx_fast_metal_kernel cls,
    const mlx_vector_array inputs,
    const mlx_fast_metal_kernel_config config,
    const mlx_stream stream) {
  try {
    auto config_ctx = mlx_fast_metal_kernel_config_get_(config);
    mlx_vector_array_set_(
        *outputs,
        mlx_fast_metal_kernel_get_(cls)(
            mlx_vector_array_get_(inputs),
            config_ctx.output_shapes,
            config_ctx.output_dtypes,
            config_ctx.grid,
            config_ctx.thread_group,
            config_ctx.template_args,
            config_ctx.init_value,
            config_ctx.verbose,
            mlx_stream_get_(stream)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_fast_rms_norm(
    mlx_array* res,
    const mlx_array x,
    const mlx_array weight /* may be null */,
    float eps,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::fast::rms_norm(
            mlx_array_get_(x),
            (weight.ctx ? std::make_optional(mlx_array_get_(weight))
                        : std::nullopt),
            eps,
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_rope(
    mlx_array* res,
    const mlx_array x,
    int dims,
    bool traditional,
    mlx_optional_float base,
    float scale,
    int offset,
    const mlx_array freqs /* may be null */,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::fast::rope(
            mlx_array_get_(x),
            dims,
            traditional,
            (base.has_value ? std::make_optional<float>(base.value)
                            : std::nullopt),
            scale,
            offset,
            (freqs.ctx ? std::make_optional(mlx_array_get_(freqs))
                       : std::nullopt),
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_rope_dynamic(
    mlx_array* res,
    const mlx_array x,
    int dims,
    bool traditional,
    mlx_optional_float base,
    float scale,
    const mlx_array offset,
    const mlx_array freqs /* may be null */,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::fast::rope(
            mlx_array_get_(x),
            dims,
            traditional,
            (base.has_value ? std::make_optional<float>(base.value)
                            : std::nullopt),
            scale,
            mlx_array_get_(offset),
            (freqs.ctx ? std::make_optional(mlx_array_get_(freqs))
                       : std::nullopt),
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_scaled_dot_product_attention(
    mlx_array* res,
    const mlx_array queries,
    const mlx_array keys,
    const mlx_array values,
    float scale,
    const char* mask_mode,
    const mlx_array mask_arr /* may be null */,
    const mlx_array sinks /* may be null */,
    const mlx_stream s) {
  try {
    mlx_array_set_(
        *res,
        mlx::core::fast::scaled_dot_product_attention(
            mlx_array_get_(queries),
            mlx_array_get_(keys),
            mlx_array_get_(values),
            scale,
            std::string(mask_mode),
            (mask_arr.ctx ? std::make_optional(mlx_array_get_(mask_arr))
                          : std::nullopt),
            (sinks.ctx ? std::make_optional(mlx_array_get_(sinks))
                       : std::nullopt),
            mlx_stream_get_(s)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

#include <json.hpp>
#include <fstream>
#include <mutex>
#include <unordered_map>
#include "mlx/backend/metal/ssd_streamer.h"

struct SSDStreamEntry {
    std::shared_ptr<mlx::core::fast::SSDStreamer> streamer;
    // Maps tensor_name -> (data_start_in_file, bytes_per_expert)
    // Key MUST be tensor_name (not num_experts) because gate_proj/up_proj/down_proj
    // all have E=256 in the same file and would collide if keyed by E.
    std::unordered_map<std::string, std::pair<size_t, size_t>> tensor_offset_map;
};

static std::mutex streamer_cache_mutex;
static std::unordered_map<std::string, SSDStreamEntry> streamer_cache;

extern "C" int mlx_fast_streamed_gather_mm(
    mlx_array* res,
    const mlx_array x,
    const mlx_array w_shape,
    uint32_t active_expert,
    const char* safetensors_path,
    const char* tensor_name,
    const mlx_stream s) {
  try {
    std::string path(safetensors_path);
    std::string target_tensor(tensor_name);
    std::shared_ptr<mlx::core::fast::SSDStreamer> streamer;
    
    // The expert index is provided directly now
    
    // We need the max possible expert index range → use w_shape dim 0
    auto w_shape_arr = mlx_array_get_(w_shape);
    size_t E = w_shape_arr.shape(0); // number of experts (e.g. 256)
    
    size_t data_start = 0;      // absolute byte offset in file to first expert's weight
    size_t bytes_per_expert = 0; // size in bytes of one expert's weight matrix
    
    {
        std::lock_guard<std::mutex> lock(streamer_cache_mutex);
        auto it = streamer_cache.find(path);
        if (it != streamer_cache.end()) {
            streamer = it->second.streamer;
            auto& tmap = it->second.tensor_offset_map;
            auto tit = tmap.find(target_tensor);
            if (tit != tmap.end()) {
                data_start       = tit->second.first;
                bytes_per_expert = tit->second.second;
            }
        }
        
        if (!streamer || bytes_per_expert == 0) {
            bool has_tensor_offsets = (it != streamer_cache.end()) &&
                                      (it->second.tensor_offset_map.count(target_tensor) > 0);
            
            if (has_tensor_offsets) {
                auto& p = it->second.tensor_offset_map.at(target_tensor);
                data_start = p.first;
                bytes_per_expert = p.second;
            } else {
                // Parse safetensors JSON header to find expert tensor layout
                std::ifstream in(path, std::ios::binary);
                if (!in.is_open()) throw std::runtime_error("[SSD] Cannot open: " + path);
                
                uint64_t hlen = 0;
                in.read(reinterpret_cast<char*>(&hlen), 8);
                std::vector<char> hbuf(hlen);
                in.read(hbuf.data(), hlen);
                auto j = nlohmann::json::parse(hbuf.data(), hbuf.data() + hlen);
                in.close();
                
                size_t data_section_start = 8 + hlen;
                
                for (auto& item : j.items()) {
                    if (item.key() == target_tensor) {
                        auto& v = item.value();
                        auto offsets = v.at("data_offsets").get<std::vector<size_t>>();
                        size_t tensor_data_start = data_section_start + offsets[0];
                        size_t tensor_total_bytes = offsets[1] - offsets[0];
                        bytes_per_expert = tensor_total_bytes / E;
                        data_start = tensor_data_start;
                        break;
                    }
                }
                
                if (bytes_per_expert == 0) {
                    throw std::runtime_error("[SSD] Could not find tensor " + target_tensor + " in " + path);
                }
                
                if (!streamer) {
                    streamer = std::make_shared<mlx::core::fast::SSDStreamer>(path, bytes_per_expert);
                    SSDStreamEntry entry;
                    entry.streamer = streamer;
                    entry.tensor_offset_map[target_tensor] = {data_start, bytes_per_expert};
                    streamer_cache[path] = std::move(entry);
                } else {
                    streamer_cache[path].tensor_offset_map[target_tensor] = {data_start, bytes_per_expert};
                }
            }
        }
    }
    
    // Build per-expert absolute file offsets
    std::vector<off_t> eo(E + 1);
    for (size_t i = 0; i <= E; ++i) {
        eo[i] = static_cast<off_t>(data_start + i * bytes_per_expert);
    }

    mlx_array_set_(
        *res,
        mlx::core::streamed_gather_mm(
            mlx_array_get_(x),
            mlx_array_get_(w_shape),
            active_expert,
            streamer,
            eo,
            mlx_stream_get_(s).device
        ));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_fast_turbo_encode(
    mlx_array* res_polar_k,
    mlx_array* res_polar_v,
    mlx_array* res_residual_k,
    mlx_array* res_residual_v,
    const mlx_array keys,
    const mlx_array values,
    int k_bits,
    const mlx_stream s) {
    try {
        // Encode K: 3-bit PolarQuant + 1-bit QJL, packed into [.., 68] uint8
        mlx_array_set_(
            *res_polar_k,
            mlx::core::fast::turbo_encode_k(
                mlx_array_get_(keys),
                mlx_stream_get_(s)));

        // Encode V: 3-bit PolarQuant only, packed into [.., 50] uint8
        mlx_array_set_(
            *res_polar_v,
            mlx::core::fast::turbo_encode_v(
                mlx_array_get_(values),
                mlx_stream_get_(s)));

        // Metadata is packed inline — residual arrays are unused but must be
        // valid (non-null ctx) so the Swift bridge can call mlx_array_free on them.
        *res_residual_k = mlx_array_new();
        *res_residual_v = mlx_array_new();
    } catch (std::exception& e) {
        mlx_error(e.what());
        return 1;
    }
    return 0;
}

extern "C" int mlx_fast_turbo_decode_k(
    mlx_array* res,
    const mlx_array packed,
    const mlx_stream s) {
    try {
        mlx_array_set_(
            *res,
            mlx::core::fast::turbo_decode_k(
                mlx_array_get_(packed),
                mlx_stream_get_(s)));
    } catch (std::exception& e) {
        mlx_error(e.what());
        return 1;
    }
    return 0;
}

extern "C" int mlx_fast_turbo_decode_v(
    mlx_array* res,
    const mlx_array packed,
    const mlx_stream s) {
    try {
        mlx_array_set_(
            *res,
            mlx::core::fast::turbo_decode_v(
                mlx_array_get_(packed),
                mlx_stream_get_(s)));
    } catch (std::exception& e) {
        mlx_error(e.what());
        return 1;
    }
    return 0;
}


extern "C" int mlx_fast_prefault(
    mlx_array x) {

    try {
        mlx::core::prefault(mlx_array_get_(x));
    } catch (std::exception& e) {
        mlx_error(e.what());
        return 1;
    }
    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// mlx_fast_pread_into
// Overwrite the data of an already-evaluated MLX array by pread()-ing
// the matching expert slab directly from a .safetensors file.
// This gives full NVMe sequential throughput (~5 GB/s) while preserving all
// MLX tensor metadata (shape, strides, dtype) on the dst array.
// ─────────────────────────────────────────────────────────────────────────────
struct STPReadEntry {
    int fd = -1;
    size_t data_start = 0;
    size_t bytes_per_expert = 0;
};
static std::mutex st_pread_cache_mutex;
static std::unordered_map<std::string, STPReadEntry> st_pread_cache;

extern "C" int mlx_fast_pread_into(
    mlx_array dst,
    const char* safetensors_path,
    const char* tensor_name,
    uint32_t expert_index) {
    try {
        std::string path(safetensors_path);
        std::string tname(tensor_name);
        std::string key = path + "|" + tname;

        size_t data_start = 0;
        size_t bytes_per_expert = 0;
        int fd = -1;

        {
            std::lock_guard<std::mutex> lock(st_pread_cache_mutex);
            auto it = st_pread_cache.find(key);
            if (it != st_pread_cache.end()) {
                fd = it->second.fd;
                data_start = it->second.data_start;
                bytes_per_expert = it->second.bytes_per_expert;
            } else {
                int new_fd = open(path.c_str(), O_RDONLY);
                if (new_fd < 0) throw std::runtime_error("[pread_into] Cannot open: " + path);

                uint64_t hlen = 0;
                if (pread(new_fd, &hlen, 8, 0) != 8) { close(new_fd); throw std::runtime_error("[pread_into] Cannot read header length"); }
                std::vector<char> hbuf(hlen);
                if ((size_t)pread(new_fd, hbuf.data(), hlen, 8) != hlen) { close(new_fd); throw std::runtime_error("[pread_into] Cannot read header JSON"); }
                auto j = nlohmann::json::parse(hbuf.data(), hbuf.data() + hlen);
                size_t data_section_start = 8 + hlen;

                for (auto& item : j.items()) {
                    if (item.key() == tname) {
                        auto& v = item.value();
                        auto shape = v.at("shape").get<std::vector<size_t>>();
                        auto offsets = v.at("data_offsets").get<std::vector<size_t>>();
                        size_t E = shape[0];
                        bytes_per_expert = (offsets[1] - offsets[0]) / E;
                        data_start = data_section_start + offsets[0];
                        break;
                    }
                }
                if (bytes_per_expert == 0) { close(new_fd); throw std::runtime_error("[pread_into] Tensor not found: " + tname); }

                STPReadEntry entry{ new_fd, data_start, bytes_per_expert };
                st_pread_cache[key] = entry;
                fd = new_fd;
            }
        }

        auto& arr = mlx_array_get_(dst);
        void* buf = const_cast<void*>(static_cast<const void*>(arr.data<uint8_t>()));
        if (!buf) throw std::runtime_error("[pread_into] dst has no data pointer — call eval() first");
        size_t nbytes = arr.nbytes();
        off_t file_offset = static_cast<off_t>(data_start + (size_t)expert_index * bytes_per_expert);
        ssize_t result = pread(fd, buf, nbytes, file_offset);
        if (result < 0 || (size_t)result != nbytes)
            throw std::runtime_error("[pread_into] pread failed: got " + std::to_string(result) + " of " + std::to_string(nbytes));

    } catch (std::exception& e) {
        mlx_error(e.what());
        return 1;
    }
    return 0;
}
