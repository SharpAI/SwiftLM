---
description: How to synchronize Apple MLX ecosystem updates into SharpAI forks and triage SSD-streaming bugs
---

# Upstream MLX Synchronization & SSD Streaming Maintenance

This workflow documents the architecture for maintaining Apple MLX forks within the SharpAI repository ecosystem, executing upstream synchronization, and resolving bugs within the `ssd_streamer` custom extensions.

## 1. Ecosystem Architecture

The `mlx-server` repository now cleanly references the upstream Swift layer `SharpAI/mlx-swift` via Swift Package Manager (`SPM`).

```
mlx-server (SharpAI/SwiftLM)
│
└── SPM Dependency: SharpAI/mlx-swift (The Swift wrapper wrapper)
    ├── .gitmodules
    │   ├── submodules/mlx   -> https://github.com/SharpAI/mlx   (Branch: main)
    │   └── submodules/mlx-c -> https://github.com/SharpAI/mlx-c (Branch: main)
```

**Never bundle C++ source files directly into `mlx-swift`.** All Apple core Engine updates and C-wrapper modifications MUST be executed in the `SharpAI/mlx` and `SharpAI/mlx-c` forks respectively.

## 2. Synchronizing New Apple Upstream Features

When Apple releases new features to `ml-explore/mlx` or `ml-explore/mlx-c` that you want to integrate into the inference engine:

1. **Pull Upstream to SharpAI forks**:
   ```bash
   git clone https://github.com/SharpAI/mlx
   cd mlx
   git remote add upstream https://github.com/ml-explore/mlx
   git fetch upstream
   
   # Rebase or merge Apple's latest main into SharpAI's main
   git rebase upstream/main
   git push origin main
   ```
2. Repeat the exact same process for `SharpAI/mlx-c`.
3. In `SharpAI/mlx-swift`, update the submodule pointers:
   ```bash
   git submodule update --remote --recursive
   git commit -am "chore: sync latest Apple MLX components"
   git push origin main
   ```
4. Finally, your local `mlx-server` will automatically pull the updated `mlx-swift` package upon running `./build.sh` (or `swift package resolve`).

## 3. Triaging SSD-Stream Bugs

The SSD streaming kernels introduce custom memory synchronization routines (`ssd_streamer.h`, `ssd_streamer.mm`) that interact with Apple's core MLX framework (`mlx/core/moe_stream_op.cpp`). 

**Triage Protocol:**
- **Crash in Metal Execution (`fence.air`, `moe_stream.metal`)**: Identify if Apple's upstream Metal API (`mlx/backend/metal/device.h`) changed rendering assumptions. Navigate to `SharpAI/mlx` and patch `mlx/backend/metal/ssd_streamer.mm`.
- **C-API Mapping Errors (`fast.cpp`, `ops.cpp`)**: Swift throws errors linking to underlying kernels. Navigate to `SharpAI/mlx-c` and ensure `mlx/c/ops.cpp` cleanly wraps the updated arguments from `SharpAI/mlx`'s `moe_stream_op.h`.
- **Memory Leaks/High Swap Usage**: Typically arises if the `fence.air` streaming barrier lacks synchronization with the newly upstreamed Apple thread-pool executors.

## 4. Retiring the Fork (When to Drop)

> [!WARNING]
> The ultimate goal is to delete the `SharpAI/mlx` and `SharpAI/mlx-c` forks and point `SharpAI/mlx-swift` directly to `ml-explore/mlx` natively.

**Indications for Dropping the Fork:**
1. Apple officially merges Turbo Quant framework into `ml-explore/mlx/fast/turbo_quant.h` or equivalent upstream PR.
2. Apple natively supports out-of-core SSD context offloading (e.g., streaming inference blocks directly from Non-Volatile Memory to GPU) in `ml-explore/mlx/backend/metal/`.
3. If Apple's `moe_stream_op` native implementations match or exceed the latency speedups provided by your custom `ssd_streamer.mm`.

If any of these conditions are met, simply rewrite `SharpAI/mlx-swift/.gitmodules` back to `https://github.com/ml-explore/mlx` and delete your Github forks!
