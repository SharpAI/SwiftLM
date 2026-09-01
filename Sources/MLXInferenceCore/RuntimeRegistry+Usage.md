# RuntimeRegistry Usage Guide

## Overview

The RuntimeRegistry provides a unified interface for SwiftBuddy to manage different runtime engines (MLX standard, DFlash, speculative decoding, streaming MoE) without external dependencies.

## Architecture

```
RuntimeRegistry (singleton)
├── RuntimeEngine (protocol)
│   ├── MLXRuntimeEngine (.standard)
│   ├── MLXRuntimeEngine (.dflash)
│   ├── MLXRuntimeEngine (.speculative)
│   └── MLXRuntimeEngine (.streamingMoE)
└── RuntimeSelector (strategy)
```

## Basic Usage

### 1. Discover Available Runtimes

```swift
import MLXInferenceCore

let registry = RuntimeRegistry.shared

// List all available runtimes
for runtime in registry.availableRuntimes {
    print("\(runtime.displayName): \(runtime.memoryEfficiency)")
}

// Find runtimes that support specific capabilities
let dflashRuntimes = registry.runtimes { $0.supportsDFlash }
let speculativeRuntimes = registry.runtimes { $0.supportsSpeculative }
```

### 2. Select and Load a Runtime

```swift
// Let the registry select the best runtime for your config
let config = RuntimeConfig(
    temperature: 0.7,
    topP: 0.9,
    maxTokens: 2048,
    enableDFlash: true  // Request DFlash-optimized runtime
)

guard let engine = registry.selectRuntime(
    for: "mlx-community/Qwen3.5-4B-MLX-4bit",
    config: config
) else {
    print("No compatible runtime found")
    return
}

// Load the model
try await engine.load(modelId: "mlx-community/Qwen3.5-4B-MLX-4bit", config: config)
```

### 3. Generate Completions

```swift
let messages = [
    ChatMessage.system("You are a helpful assistant."),
    ChatMessage.user("What is the capital of France?")
]

let stream = engine.generate(messages: messages, config: config)

for try await token in stream {
    print(token.text, terminator: "")
    if token.isThinking {
        // Handle thinking tokens (e.g., dim them in UI)
    }
}
```

### 4. Runtime Switching

```swift
// Switch to a different runtime for the same model
let speculativeConfig = RuntimeConfig(
    temperature: 0.7,
    maxTokens: 2048,
    enableSpeculative: true  // Request speculative decoding
)

// Registry will unload the current engine and activate the new one
if let speculativeEngine = registry.selectRuntime(
    for: modelId,
    config: speculativeConfig
) {
    try await speculativeEngine.load(modelId: modelId, config: speculativeConfig)
}
```

## SwiftBuddy Integration

### In Settings View

```swift
struct RuntimeSettingsView: View {
    @StateObject private var registry = RuntimeRegistry.shared
    @State private var selectedRuntimeId = "mlx.standard"
    
    var body: some View {
        VStack {
            Picker("Runtime Engine", selection: $selectedRuntimeId) {
                ForEach(registry.availableRuntimes, id: \.id) { runtime in
                    Text(runtime.displayName).tag(runtime.id)
                }
            }
            
            if let cap = registry.capability(for: selectedRuntimeId) {
                VStack(alignment: .leading) {
                    Label("Memory: \(cap.memoryEfficiency.rawValue)", 
                          systemImage: "memorychip")
                    if cap.supportsDFlash {
                        Label("DFlash Enabled", systemImage: "bolt.fill")
                    }
                    if cap.supportsSpeculative {
                        Label("Speculative Decoding", systemImage: "forward.fill")
                    }
                }
            }
        }
    }
}
```

### In Chat View

```swift
struct ChatView: View {
    @StateObject private var registry = RuntimeRegistry.shared
    @State private var messages: [ChatMessage] = []
    @State private var currentResponse = ""
    
    func sendMessage(_ text: String) {
        messages.append(.user(text))
        
        Task {
            // Use the active engine
            guard let engine = registry.activeEngine else { return }
            
            let config = RuntimeConfig(
                temperature: 0.7,
                maxTokens: 2048
            )
            
            currentResponse = ""
            let stream = engine.generate(messages: messages, config: config)
            
            for try await token in stream {
                currentResponse += token.text
            }
            
            messages.append(.assistant(currentResponse))
        }
    }
}
```

## Runtime Selection Logic

The `RuntimeSelector` automatically picks the best runtime based on:

1. **DFlash requested** → Use `mlx.dflash` if available
2. **Speculative requested** → Use `mlx.speculative` if available
3. **MoE + streaming** → Use `mlx.streaming_moe` if available
4. **Default** → Use `mlx.standard`

## Runtime Capabilities Matrix

| Runtime | Streaming | DFlash | Speculative | Vision | Audio | Memory |
|---------|-----------|--------|-------------|--------|-------|--------|
| mlx.standard | ✓ | — | — | ✓ | ✓ | Standard |
| mlx.dflash | ✓ | ✓ | — | ✓ | ✓ | Optimized |
| mlx.speculative | ✓ | — | ✓ | ✓ | ✓ | Standard |
| mlx.streaming_moe | ✓ | ✓ | — | — | — | Extreme |

## Next Steps

1. **Extend RuntimeConfig** with additional flags (thinking mode, tool calling, etc.)
2. **Add Runtime Monitoring** (tokens/sec, memory usage, thermal state)
3. **Implement Runtime Presets** (Performance, Balanced, Efficiency)
4. **Add Runtime Benchmarks** (TTFT, throughput comparison)
