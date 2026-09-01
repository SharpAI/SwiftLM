// RuntimeRegistry.swift — Central registry for all available runtime engines
// Provides discovery, selection, and lifecycle management for runtimes.

import Foundation

// MARK: — Runtime Registry

/// Central registry that manages all available runtime engines.
/// SwiftBuddy queries this to discover and instantiate runtimes.
@MainActor
public final class RuntimeRegistry: ObservableObject {
    
    public static let shared = RuntimeRegistry()
    
    @Published public private(set) var availableRuntimes: [RuntimeCapability] = []
    @Published public private(set) var activeEngine: (any RuntimeEngine)?
    
    private var engineFactories: [String: () -> any RuntimeEngine] = [:]
    
    private init() {
        registerBuiltInRuntimes()
    }
    
    // MARK: — Registration
    
    /// Register a runtime engine factory.
    /// The factory closure is called lazily when the runtime is requested.
    public func register(
        capability: RuntimeCapability,
        factory: @escaping () -> any RuntimeEngine
    ) {
        engineFactories[capability.id] = factory
        if !availableRuntimes.contains(where: { $0.id == capability.id }) {
            availableRuntimes.append(capability)
        }
    }
    
    /// Register the built-in native Swift runtimes.
    private func registerBuiltInRuntimes() {
        // Standard MLX runtime
        register(
            capability: RuntimeCapability(
                id: "mlx.standard",
                displayName: "MLX Standard",
                supportsStreaming: true,
                supportsVision: true,
                supportsAudio: true,
                supportsToolCalling: true,
                memoryEfficiency: .standard
            ),
            factory: { MLXRuntimeEngine(mode: .standard) }
        )
        
        // DFlash-optimized runtime (memory-efficient attention)
        register(
            capability: RuntimeCapability(
                id: "mlx.dflash",
                displayName: "MLX + DFlash",
                supportsStreaming: true,
                supportsDFlash: true,
                supportsVision: true,
                supportsAudio: true,
                supportsToolCalling: true,
                memoryEfficiency: .optimized
            ),
            factory: { MLXRuntimeEngine(mode: .dflash) }
        )
        
        // Speculative decoding runtime (faster inference)
        register(
            capability: RuntimeCapability(
                id: "mlx.speculative",
                displayName: "MLX Speculative",
                supportsStreaming: true,
                supportsSpeculative: true,
                supportsVision: true,
                supportsAudio: true,
                supportsToolCalling: true,
                memoryEfficiency: .standard
            ),
            factory: { MLXRuntimeEngine(mode: .speculative) }
        )
        
        // MoE with SSD expert streaming (extreme memory efficiency)
        register(
            capability: RuntimeCapability(
                id: "mlx.streaming_moe",
                displayName: "MLX Streaming MoE",
                supportsStreaming: true,
                supportsDFlash: true,
                supportsVision: false,  // MoE models typically don't support vision
                supportsAudio: false,
                supportsToolCalling: true,
                memoryEfficiency: .extreme
            ),
            factory: { MLXRuntimeEngine(mode: .streamingMoE) }
        )
    }
    
    // MARK: — Runtime Selection
    
    /// Get or create a runtime engine by ID.
    public func engine(for runtimeId: String) -> (any RuntimeEngine)? {
        // If this is the active engine, return it directly
        if let active = activeEngine, active.id == runtimeId {
            return active
        }
        
        // Otherwise create a new instance
        guard let factory = engineFactories[runtimeId] else {
            return nil
        }
        
        let engine = factory()
        return engine
    }
    
    /// Select and activate the best runtime for the given model and config.
    /// This will deactivate any currently active engine.
    public func selectRuntime(
        for modelId: String,
        config: RuntimeConfig
    ) -> (any RuntimeEngine)? {
        
        // Use the selector to determine the best runtime
        guard let runtimeId = RuntimeSelector.selectRuntime(
            modelId: modelId,
            preferredConfig: config,
            availableRuntimes: availableRuntimes
        ) else {
            return nil
        }
        
        // Unload the current engine if it's different
        if let current = activeEngine, current.id != runtimeId {
            current.unload()
            activeEngine = nil
        }
        
        // Get or create the selected engine
        if activeEngine == nil {
            activeEngine = engine(for: runtimeId)
        }
        
        return activeEngine
    }
    
    /// Get the capability descriptor for a runtime ID.
    public func capability(for runtimeId: String) -> RuntimeCapability? {
        availableRuntimes.first(where: { $0.id == runtimeId })
    }
    
    /// Check if a model is compatible with any registered runtime.
    public func hasCompatibleRuntime(for modelId: String) async -> Bool {
        for (id, factory) in engineFactories {
            let engine = factory()
            if await engine.isCompatible(modelId: modelId) {
                return true
            }
        }
        return false
    }
    
    // MARK: — State Management
    
    /// Unload and deactivate the current runtime engine.
    public func deactivate() {
        activeEngine?.unload()
        activeEngine = nil
    }
    
    /// Get a list of all runtime IDs that support a specific capability.
    public func runtimes(supporting capability: (RuntimeCapability) -> Bool) -> [String] {
        availableRuntimes.filter(capability).map(\.id)
    }
    
    /// Get a user-friendly description of the active runtime.
    public var activeRuntimeDescription: String {
        guard let engine = activeEngine else { return "No runtime active" }
        let cap = capability(for: engine.id)
        let efficiency = cap?.memoryEfficiency.rawValue ?? "standard"
        return "\(engine.displayName) (\(efficiency))"
    }
}
