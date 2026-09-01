# SwiftBuddy RuntimeRegistry Integration Guide

## Overview

The RuntimeRegistry system is now fully integrated into SwiftLM and ready for SwiftBuddy. This guide shows how to wire it into SwiftBuddy's UI.

## What's Been Implemented

### Core Infrastructure (MLXInferenceCore)

1. **RuntimeEngine.swift** - Protocol defining the runtime interface
   - `RuntimeCapability`: describes runtime features (DFlash, speculative, vision, etc.)
   - `RuntimeConfig`: per-request configuration (temperature, topP, etc.)
   - `RuntimeEngine` protocol: load/unload/generate/state management
   - `RuntimeSelector`: automatic runtime selection logic

2. **RuntimeRegistry.swift** - Central singleton registry
   - Manages all available runtimes
   - Factory-based lazy instantiation
   - Automatic runtime switching
   - Four built-in runtimes: standard, DFlash, speculative, streaming MoE

3. **MLXRuntimeEngine.swift** - Adapter wrapping InferenceEngine
   - Implements RuntimeEngine protocol
   - Exposes `inferenceEngine` for gradual migration
   - Four runtime modes matching different InferenceEngine configurations

### SwiftBuddy Integration Layer

4. **RuntimeService.swift** - SwiftBuddy's bridge to RuntimeRegistry
   - ObservableObject for SwiftUI integration
   - Manages model loading with runtime selection
   - Persists user's preferred runtime
   - Provides `legacyEngine` for gradual migration

5. **RuntimePickerView.swift** - UI component for runtime selection
   - Shows available runtimes with capability badges
   - Displays memory efficiency indicators
   - Real-time capability visualization
   - Drop-in component for SettingsView

## Integration Steps

### Step 1: Add RuntimePickerView to SettingsView

Add the runtime picker to the `engineTab` in [SettingsView.swift](../SwiftBuddy/Views/SettingsView.swift):

```swift
// Around line 410, inside engineTab's ScrollView
private var engineTab: some View {
    ScrollView {
        VStack(spacing: 16) {
            // ── NEW: Runtime Selection ───────────────────────────────
            parameterCard("Runtime Engine") {
                RuntimePickerView()
            }
            
            // ── Existing: Local API Server ──────────────────────────
            parameterCard("Local API Server") {
                // ... existing server config UI ...
            }
            
            // ... rest of engineTab ...
        }
    }
}
```

### Step 2: Wire RuntimeService into App Lifecycle

In [SwiftBuddyApp.swift](../SwiftBuddy/SwiftBuddyApp.swift), add RuntimeService:

```swift
@main
struct SwiftBuddyApp: App {
    @StateObject private var appearance = AppearanceStore()
    @StateObject private var engine = InferenceEngine()
    @StateObject private var runtimeService = RuntimeService()  // NEW
    @StateObject private var server = ServerManager()
    
    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(appearance)
                .environmentObject(engine)
                .environmentObject(runtimeService)  // NEW
                .environmentObject(server)
        }
    }
}
```

### Step 3: Gradual Migration Path

The RuntimeService provides a `legacyEngine` property that returns the underlying InferenceEngine. This allows SwiftBuddy to continue using its existing ChatViewModel while gradually adopting the new RuntimeRegistry system.

**Current State (No Changes Required):**
```swift
// ChatViewModel.swift continues to work as-is
weak var engine: InferenceEngine?

func send(_ userText: String) async {
    guard let engine, !isGenerating else { return }
    // ... existing code using InferenceEngine directly ...
}
```

**Future State (Gradual Migration):**
```swift
// When ready, inject RuntimeService instead
@ObservedObject var runtimeService: RuntimeService

func send(_ userText: String) async {
    guard let engine = runtimeService.activeEngine else { return }
    
    let config = RuntimeConfig(
        temperature: viewModel.config.temperature,
        maxTokens: viewModel.config.maxTokens
    )
    
    let stream = engine.generate(messages: messages, config: config)
    for try await token in stream {
        // Handle token...
    }
}
```

## User Experience Flow

### 1. Runtime Selection

User opens Settings → Engine tab → sees Runtime Engine card:

```
┌─────────────────────────────────────────┐
│ Runtime Engine                          │
│                                         │
│ [MLX + DFlash ▼]              Optimized│
│                                         │
│ • Streaming  • DFlash  • Vision • Audio │
│                                         │
│ Memory-optimized attention (DFlash).    │
│ Best for large context windows with     │
│ lower RAM usage.                        │
└─────────────────────────────────────────┘
```

### 2. Runtime Persistence

User's selection is saved to UserDefaults:
- Key: `swiftbuddy.preferredRuntime`
- Values: `mlx.standard`, `mlx.dflash`, `mlx.speculative`, `mlx.streaming_moe`

### 3. Automatic Runtime Application

When user loads a model:
- RuntimeService uses the preferred runtime
- If incompatible, falls back to `mlx.standard`
- User can see active runtime capabilities in real-time

## Runtime Capabilities Reference

| Runtime | Memory | DFlash | Speculative | Vision | Audio | Best For |
|---------|--------|--------|-------------|--------|-------|----------|
| Standard | Standard | ❌ | ❌ | ✅ | ✅ | General use, vision/audio models |
| DFlash | Optimized | ✅ | ❌ | ✅ | ✅ | Large contexts, RAM-constrained |
| Speculative | Standard | ❌ | ✅ | ✅ | ✅ | Fast inference with draft models |
| Streaming MoE | Extreme | ✅ | ❌ | ❌ | ❌ | MoE models larger than RAM |

## Testing

### Manual Testing

1. **Build and run SwiftBuddy**
   ```bash
   cd /Users/simba/SwiftLM
   swift build
   ```

2. **Open Settings → Engine tab**
   - Verify RuntimePickerView appears
   - Try selecting different runtimes
   - Check that selection persists after restart

3. **Load a model**
   - Verify the selected runtime is used
   - Check capability badges match runtime
   - Confirm generation works

### Unit Testing

```swift
import XCTest
@testable import MLXInferenceCore

class RuntimeRegistryTests: XCTestCase {
    func testBuiltInRuntimesRegistered() {
        let registry = RuntimeRegistry.shared
        XCTAssertEqual(registry.availableRuntimes.count, 4)
    }
    
    func testRuntimeSelection() {
        let registry = RuntimeRegistry.shared
        let config = RuntimeConfig(enableDFlash: true)
        let runtimeId = RuntimeSelector.selectRuntime(
            from: registry.availableRuntimes,
            for: "mlx-community/Qwen3.5-4B-MLX-4bit",
            config: config
        )
        XCTAssertEqual(runtimeId, "mlx.dflash")
    }
}
```

## Next Steps

After RuntimeRegistry stabilizes:

1. **ModelLibrary Module** (from 5-module architecture)
   - Extend ModelCatalog with runtime compatibility checks
   - Add model import/export
   - Integrate with RuntimeRegistry

2. **Runtime Monitoring**
   - Add tokens/sec tracking
   - Expose thermal state in UI
   - Show active context tokens

3. **Runtime Presets**
   - "Performance" (speculative)
   - "Balanced" (standard)
   - "Efficiency" (DFlash + streaming)

4. **Advanced Runtime Config**
   - Per-model runtime preferences
   - Automatic runtime switching based on device state
   - Custom runtime registration via plugins

## Files Modified/Created

### Core (MLXInferenceCore)
- ✅ [RuntimeEngine.swift](../Sources/MLXInferenceCore/RuntimeEngine.swift)
- ✅ [RuntimeRegistry.swift](../Sources/MLXInferenceCore/RuntimeRegistry.swift)
- ✅ [MLXRuntimeEngine.swift](../Sources/MLXInferenceCore/MLXRuntimeEngine.swift)
- ✅ [RuntimeRegistry+Usage.md](../Sources/MLXInferenceCore/RuntimeRegistry+Usage.md)

### SwiftBuddy
- ✅ [RuntimeService.swift](../SwiftBuddy/SwiftBuddy/ViewModels/RuntimeService.swift)
- ✅ [RuntimePickerView.swift](../SwiftBuddy/SwiftBuddy/Views/RuntimePickerView.swift)
- ⏳ [SwiftBuddyApp.swift](../SwiftBuddy/SwiftBuddy/SwiftBuddyApp.swift) - needs RuntimeService injection
- ⏳ [SettingsView.swift](../SwiftBuddy/SwiftBuddy/Views/SettingsView.swift) - needs RuntimePickerView integration

### Documentation
- ✅ This integration guide

## Status

**Runtime Core:** ✅ Complete
**SwiftBuddy UI Components:** ✅ Complete
**Integration:** ⏳ Ready for wiring (Steps 1-2 above)
**Testing:** ⏳ Ready for manual testing
**Production Use:** ⏳ After user testing
