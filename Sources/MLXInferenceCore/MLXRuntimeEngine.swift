// MLXRuntimeEngine.swift — RuntimeEngine adapter for the existing InferenceEngine
// Wraps InferenceEngine with different runtime modes (standard, DFlash, speculative, etc.)

import Foundation
import MLX

// MARK: — Runtime Mode

public enum MLXRuntimeMode: String, Sendable {
    case standard       // Vanilla MLX inference
    case dflash         // DFlash-optimized attention
    case speculative    // Speculative decoding
    case streamingMoE   // MoE with SSD expert streaming
}

// MARK: — MLX Runtime Engine

/// Adapter that wraps InferenceEngine and exposes it via the RuntimeEngine protocol.
/// Different instances can be configured for different runtime modes.
@MainActor
public final class MLXRuntimeEngine: RuntimeEngine {
    
    public let mode: MLXRuntimeMode
    
    // Exposed for gradual migration: SwiftBuddy can access the underlying InferenceEngine
    // while transitioning to the RuntimeEngine protocol.
    public let inferenceEngine: InferenceEngine
    
    // Store the config used during load so we can pass it to generate()
    private var lastConfig: RuntimeConfig = .default
    
    public init(mode: MLXRuntimeMode) {
        self.mode = mode
        self.inferenceEngine = InferenceEngine()
    }
    
    // MARK: — RuntimeEngine Protocol
    
    public var id: String {
        switch mode {
        case .standard:     return "mlx.standard"
        case .dflash:       return "mlx.dflash"
        case .speculative:  return "mlx.speculative"
        case .streamingMoE: return "mlx.streaming_moe"
        }
    }
    
    public var displayName: String {
        switch mode {
        case .standard:     return "MLX Standard"
        case .dflash:       return "MLX + DFlash"
        case .speculative:  return "MLX Speculative"
        case .streamingMoE: return "MLX Streaming MoE"
        }
    }
    
    public var capabilities: RuntimeCapability {
        switch mode {
        case .standard:
            return RuntimeCapability(
                id: id,
                displayName: displayName,
                supportsStreaming: true,
                supportsVision: true,
                supportsAudio: true,
                supportsToolCalling: true,
                memoryEfficiency: .standard
            )
        case .dflash:
            return RuntimeCapability(
                id: id,
                displayName: displayName,
                supportsStreaming: true,
                supportsDFlash: true,
                supportsVision: true,
                supportsAudio: true,
                supportsToolCalling: true,
                memoryEfficiency: .optimized
            )
        case .speculative:
            return RuntimeCapability(
                id: id,
                displayName: displayName,
                supportsStreaming: true,
                supportsSpeculative: true,
                supportsVision: true,
                supportsAudio: true,
                supportsToolCalling: true,
                memoryEfficiency: .standard
            )
        case .streamingMoE:
            return RuntimeCapability(
                id: id,
                displayName: displayName,
                supportsStreaming: true,
                supportsDFlash: true,
                supportsToolCalling: true,
                memoryEfficiency: .extreme
            )
        }
    }
    
    public var state: ModelState {
        inferenceEngine.state
    }
    
    public var loadedModelId: String? {
        inferenceEngine.loadedModelId
    }
    
    public var activeContextTokens: Int {
        inferenceEngine.activeContextTokens
    }
    
    public var maxContextWindow: Int {
        inferenceEngine.maxContextWindow
    }
    
    public func load(modelId: String, config: RuntimeConfig) async throws {
        // Store config for later use in generate()
        lastConfig = config
        
        // Configure InferenceEngine based on runtime mode and config
        // The actual model loading is delegated to InferenceEngine
        await inferenceEngine.load(modelId: modelId)
        
        // TODO: When InferenceEngine exposes runtime flags (DFlash, speculative, etc.),
        // configure them here based on self.mode and config flags.
        // For now, InferenceEngine handles this internally.
    }
    
    public func unload() {
        inferenceEngine.unload()
    }
    
    public func generate(
        messages: [ChatMessage],
        config: RuntimeConfig
    ) -> AsyncThrowingStream<GenerationToken, Error> {
        // Forward to InferenceEngine's generate method
        // Map RuntimeConfig to GenerationConfig
        
        return AsyncThrowingStream { continuation in
            Task { @MainActor in
                // Build GenerationConfig from RuntimeConfig
                var genConfig = GenerationConfig.load()
                genConfig.maxTokens = config.maxTokens
                genConfig.temperature = config.temperature
                genConfig.topP = config.topP
                
                if let topK = config.topK {
                    genConfig.topK = topK
                }
                if let minP = config.minP {
                    genConfig.minP = minP
                }
                if let repPenalty = config.repetitionPenalty {
                    genConfig.repetitionPenalty = repPenalty
                }
                
                // Apply runtime-specific settings
                genConfig.streamExperts = config.streamExperts
                
                // InferenceEngine.generate returns AsyncStream (non-throwing)
                // Adapt it to AsyncThrowingStream
                let stream = inferenceEngine.generate(
                    messages: messages,
                    config: genConfig
                )
                
                for await token in stream {
                    continuation.yield(token)
                }
                continuation.finish()
            }
        }
    }
    
    public func stopGeneration() {
        inferenceEngine.stopGeneration()
    }
    
    public func isCompatible(modelId: String) async -> Bool {
        // Check model compatibility based on runtime mode
        
        // For streamingMoE, only MoE models are compatible
        if mode == .streamingMoE {
            // Check if model is MoE (common patterns)
            let lowercaseId = modelId.lowercased()
            return lowercaseId.contains("moe") || 
                   lowercaseId.contains("mixtral") ||  // Mixtral is MoE
                   lowercaseId.contains("a3b") || 
                   lowercaseId.contains("a10b") ||
                   lowercaseId.contains("deepseek") ||  // DeepSeek models often MoE
                   lowercaseId.contains("qwen-moe")     // Qwen MoE variants
        }
        
        // For other modes, most MLX models are compatible
        // Vision models need vision support
        let lowercaseId = modelId.lowercased()
        let isVisionModel = lowercaseId.contains("vision") || 
                           lowercaseId.contains("vlm") ||
                           lowercaseId.contains("4vl") ||
                           lowercaseId.contains("pixtral")  // Pixtral is vision
        
        if isVisionModel && !capabilities.supportsVision {
            return false
        }
        
        return true
    }
}


