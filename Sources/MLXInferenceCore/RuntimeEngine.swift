// RuntimeEngine.swift — Protocol for unified runtime abstraction
// Allows SwiftBuddy to select between MLX vanilla, DFlash, speculative, etc.

import Foundation
import MLX

// MARK: — Runtime Capability

/// Describes what a runtime engine supports.
public struct RuntimeCapability: Sendable, Hashable {
    public let id: String
    public let displayName: String
    public let supportsStreaming: Bool
    public let supportsSpeculative: Bool
    public let supportsDFlash: Bool
    public let supportsVision: Bool
    public let supportsAudio: Bool
    public let supportsToolCalling: Bool
    public let memoryEfficiency: MemoryEfficiency
    
    public enum MemoryEfficiency: String, Sendable, Hashable {
        case standard     // Normal MLX inference
        case optimized    // DFlash memory-efficient attention
        case extreme      // SSD streaming, expert offloading
    }
    
    public init(
        id: String,
        displayName: String,
        supportsStreaming: Bool = true,
        supportsSpeculative: Bool = false,
        supportsDFlash: Bool = false,
        supportsVision: Bool = false,
        supportsAudio: Bool = false,
        supportsToolCalling: Bool = false,
        memoryEfficiency: MemoryEfficiency = .standard
    ) {
        self.id = id
        self.displayName = displayName
        self.supportsStreaming = supportsStreaming
        self.supportsSpeculative = supportsSpeculative
        self.supportsDFlash = supportsDFlash
        self.supportsVision = supportsVision
        self.supportsAudio = supportsAudio
        self.supportsToolCalling = supportsToolCalling
        self.memoryEfficiency = memoryEfficiency
    }
}

// MARK: — Runtime Configuration

/// Per-session runtime configuration.
public struct RuntimeConfig: Sendable {
    public var temperature: Float
    public var topP: Float
    public var topK: Int?
    public var minP: Float?
    public var repetitionPenalty: Float?
    public var maxTokens: Int
    public var contextWindow: Int?
    
    // Runtime-specific flags
    public var enableSpeculative: Bool
    public var enableDFlash: Bool
    public var streamExperts: Bool  // MoE SSD streaming
    
    public static let `default` = RuntimeConfig(
        temperature: 0.7,
        topP: 1.0,
        topK: nil,
        minP: nil,
        repetitionPenalty: nil,
        maxTokens: 2048,
        contextWindow: nil,
        enableSpeculative: false,
        enableDFlash: false,
        streamExperts: false
    )
    
    public init(
        temperature: Float = 0.7,
        topP: Float = 1.0,
        topK: Int? = nil,
        minP: Float? = nil,
        repetitionPenalty: Float? = nil,
        maxTokens: Int = 2048,
        contextWindow: Int? = nil,
        enableSpeculative: Bool = false,
        enableDFlash: Bool = false,
        streamExperts: Bool = false
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.maxTokens = maxTokens
        self.contextWindow = contextWindow
        self.enableSpeculative = enableSpeculative
        self.enableDFlash = enableDFlash
        self.streamExperts = streamExperts
    }
}

// MARK: — Runtime Engine Protocol

/// Unified interface for all runtime engines (MLX, DFlash, Speculative, etc.)
@MainActor
public protocol RuntimeEngine: AnyObject {
    /// Unique identifier for this runtime engine.
    var id: String { get }
    
    /// Human-readable name for UI display.
    var displayName: String { get }
    
    /// Capabilities this runtime supports.
    var capabilities: RuntimeCapability { get }
    
    /// Current runtime state.
    var state: ModelState { get }
    
    /// Currently loaded model ID, if any.
    var loadedModelId: String? { get }
    
    /// Active context tokens in the current session.
    var activeContextTokens: Int { get }
    
    /// Maximum context window for the loaded model.
    var maxContextWindow: Int { get }
    
    /// Load a model with the given configuration.
    func load(modelId: String, config: RuntimeConfig) async throws
    
    /// Unload the currently loaded model and free resources.
    func unload()
    
    /// Generate a streaming response for the given messages.
    /// Returns an async sequence of token strings.
    func generate(
        messages: [ChatMessage],
        config: RuntimeConfig
    ) -> AsyncThrowingStream<GenerationToken, Error>
    
    /// Stop any active generation.
    func stopGeneration()
    
    /// Check if a model is compatible with this runtime.
    func isCompatible(modelId: String) async -> Bool
}

// MARK: — Runtime Selection Strategy

/// Determines which runtime to use for a given model and configuration.
public struct RuntimeSelector: Sendable {
    
    /// Select the best runtime for the given model and preferences.
    public static func selectRuntime(
        modelId: String,
        preferredConfig: RuntimeConfig,
        availableRuntimes: [RuntimeCapability]
    ) -> String? {
        
        // Priority order:
        // 1. If DFlash explicitly requested and available → use DFlash runtime
        // 2. If speculative decoding requested → use Speculative runtime
        // 3. If MoE with SSD streaming → use StreamingMoE runtime
        // 4. Default to standard MLX runtime
        
        if preferredConfig.enableDFlash {
            if let dflash = availableRuntimes.first(where: { $0.supportsDFlash }) {
                return dflash.id
            }
        }
        
        if preferredConfig.enableSpeculative {
            if let spec = availableRuntimes.first(where: { $0.supportsSpeculative }) {
                return spec.id
            }
        }
        
        if preferredConfig.streamExperts {
            if let streaming = availableRuntimes.first(where: { 
                $0.memoryEfficiency == .extreme 
            }) {
                return streaming.id
            }
        }
        
        // Fallback to standard MLX
        return availableRuntimes.first(where: { $0.id == "mlx.standard" })?.id
    }
}
