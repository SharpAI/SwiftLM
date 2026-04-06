# TurboQuant Implementation Details

For reference and future maintenance, this file documents the design decisions of the TurboQuant KV Cache compression inside SwiftLM, specifically comparing it against the `turboquant-mlx` reference reproduction.

Our C++ / Metal backend effectively achieves the "V3 Quality at V2 Speed" benchmark described in the reproduction. By baking the Lloyd-Max codebooks directly into the C++ `turbo_quantize` process and evaluating it during the Metal SDPA kernel (`bggml-metal`), we completely bypass the Python orchestration bottlenecks.

- **K Cache**: 4.25 bits/dim. (3-bit Lloyd-Max centroids + 1-bit QJL residual sign projection). Includes L2 Normalization and WHT rotation.
- **V Cache**: 3.125 bits/dim. (3-bit Lloyd-Max centroids). QJL works by estimating inner-product bias, so it is strictly unnecessary for the V cache. 

This gives a total compression of ~3.5x relative to fp16, while the QJL correction on the K-cache preserves the attention distribution nearly losslessly.
