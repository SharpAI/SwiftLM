// RuntimePreset.swift — Curated runtime configuration presets
import Foundation

// MARK: — Runtime Preset

/// Pre-configured, validated parameter combination for a specific runtime engine.
public struct RuntimePreset: Identifiable, Sendable {
    public let id: String
    public let name: String
    public let description: String
    public let config: RuntimeConfig
    public let runtimeId: String
    public let icon: String
    
    public init(
        id: String,
        name: String,
        description: String,
        config: RuntimeConfig,
        runtimeId: String,
        icon: String
    ) {
        self.id = id
        self.name = name
        self.description = description
        self.config = config
        self.runtimeId = runtimeId
        self.icon = icon
    }
}

// MARK: — Preset Library

/// Centralized library of recommended presets for each runtime engine.
public struct RuntimePresetLibrary {
    
    // MARK: MLX Standard Presets
    
    public static let standardBalanced = RuntimePreset(
        id: "mlx.standard.balanced",
        name: "Balanced",
        description: "General-purpose settings. Good balance between creativity and coherence.",
        config: RuntimeConfig(
            temperature: 0.7,
            topP: 0.9,
            topK: nil,
            minP: nil,
            repetitionPenalty: 1.1,
            maxTokens: 2048,
            contextWindow: 8192
        ),
        runtimeId: "mlx.standard",
        icon: "equal.circle.fill"
    )
    
    public static let standardCreative = RuntimePreset(
        id: "mlx.standard.creative",
        name: "Creative",
        description: "Higher temperature for creative writing, brainstorming, storytelling.",
        config: RuntimeConfig(
            temperature: 1.0,
            topP: 0.95,
            topK: nil,
            minP: nil,
            repetitionPenalty: 1.05,
            maxTokens: 2048,
            contextWindow: 8192
        ),
        runtimeId: "mlx.standard",
        icon: "sparkles"
    )
    
    public static let standardPrecise = RuntimePreset(
        id: "mlx.standard.precise",
        name: "Precise",
        description: "Low temperature for factual tasks, code generation, structured output.",
        config: RuntimeConfig(
            temperature: 0.3,
            topP: 0.7,
            topK: 40,
            minP: 0.05,
            repetitionPenalty: 1.15,
            maxTokens: 2048,
            contextWindow: 8192
        ),
        runtimeId: "mlx.standard",
        icon: "target"
    )
    
    public static let standardVision = RuntimePreset(
        id: "mlx.standard.vision",
        name: "Vision (VLM)",
        description: "Optimized for vision-language models (Qwen-VL, LLaVA, Pixtral).",
        config: RuntimeConfig(
            temperature: 0.6,
            topP: 0.9,
            topK: nil,
            minP: nil,
            repetitionPenalty: 1.1,
            maxTokens: 1024,
            contextWindow: 4096
        ),
        runtimeId: "mlx.standard",
        icon: "eye.fill"
    )
    
    // MARK: DFlash Presets
    
    public static let dflashLongContext = RuntimePreset(
        id: "mlx.dflash.long",
        name: "Long Context (40K)",
        description: "Memory-optimized for 40K token context windows. Lower temperature to reduce repetition.",
        config: RuntimeConfig(
            temperature: 0.5,
            topP: 0.85,
            topK: 50,
            minP: 0.05,
            repetitionPenalty: 1.2,
            maxTokens: 4096,
            contextWindow: 40000,
            enableDFlash: true
        ),
        runtimeId: "mlx.dflash",
        icon: "doc.text.fill"
    )
    
    public static let dflashUltraContext = RuntimePreset(
        id: "mlx.dflash.ultra",
        name: "Ultra Context (100K)",
        description: "Extreme memory optimization for 100K token windows. Best with TurboQuant.",
        config: RuntimeConfig(
            temperature: 0.4,
            topP: 0.8,
            topK: 40,
            minP: 0.1,
            repetitionPenalty: 1.25,
            maxTokens: 8192,
            contextWindow: 100000,
            enableDFlash: true
        ),
        runtimeId: "mlx.dflash",
        icon: "doc.on.doc.fill"
    )
    
    // MARK: Speculative Decoding Presets
    
    public static let speculativeSpeed = RuntimePreset(
        id: "mlx.speculative.speed",
        name: "Speed Optimized",
        description: "Fast inference with 9B draft model. Requires draft+target model pair.",
        config: RuntimeConfig(
            temperature: 0.7,
            topP: 0.9,
            topK: nil,
            minP: nil,
            repetitionPenalty: 1.1,
            maxTokens: 2048,
            contextWindow: 8192,
            enableSpeculative: true
        ),
        runtimeId: "mlx.speculative",
        icon: "bolt.fill"
    )
    
    public static let speculativeBalanced = RuntimePreset(
        id: "mlx.speculative.balanced",
        name: "Balanced Speed",
        description: "Moderate speculative settings. Good for general use with draft models.",
        config: RuntimeConfig(
            temperature: 0.6,
            topP: 0.85,
            topK: 50,
            minP: nil,
            repetitionPenalty: 1.15,
            maxTokens: 2048,
            contextWindow: 8192,
            enableSpeculative: true
        ),
        runtimeId: "mlx.speculative",
        icon: "gauge.with.needle.fill"
    )
    
    // MARK: Streaming MoE Presets
    
    public static let streaming122B = RuntimePreset(
        id: "mlx.moe.122b",
        name: "Qwen3.5-122B",
        description: "Optimized for 122B MoE models. ~16GB RAM, expert streaming from SSD.",
        config: RuntimeConfig(
            temperature: 0.7,
            topP: 0.9,
            topK: nil,
            minP: nil,
            repetitionPenalty: 1.1,
            maxTokens: 2048,
            contextWindow: 32768,
            streamExperts: true
        ),
        runtimeId: "mlx.streaming_moe",
        icon: "externaldrive.fill.badge.checkmark"
    )
    
    public static let streaming397B = RuntimePreset(
        id: "mlx.moe.397b",
        name: "Qwen3.5-397B",
        description: "Optimized for ultra-large 397B MoE models. ~22GB RAM, max SSD throughput.",
        config: RuntimeConfig(
            temperature: 0.7,
            topP: 0.9,
            topK: nil,
            minP: nil,
            repetitionPenalty: 1.1,
            maxTokens: 2048,
            contextWindow: 32768,
            streamExperts: true
        ),
        runtimeId: "mlx.streaming_moe",
        icon: "externaldrive.fill.badge.wifi"
    )
    
    public static let streamingDeepSeek = RuntimePreset(
        id: "mlx.moe.deepseek",
        name: "DeepSeek-V3/V4",
        description: "Optimized for DeepSeek MoE models. Aggressive expert streaming.",
        config: RuntimeConfig(
            temperature: 0.6,
            topP: 0.85,
            topK: nil,
            minP: nil,
            repetitionPenalty: 1.15,
            maxTokens: 2048,
            contextWindow: 32768,
            streamExperts: true
        ),
        runtimeId: "mlx.streaming_moe",
        icon: "cpu.fill"
    )
    
    // MARK: Gemma 4 26B MoE Presets (Bootstrap Model)
    
    public static let gemma4_26B_mtpFast = RuntimePreset(
        id: "gemma4.moe.mtp_fast",
        name: "Gemma 4 26B - MTP Acceleration",
        description: "Default. 1.14x speedup with MTP assistant. Tested at 147.2 tok/s.",
        config: RuntimeConfig(
            temperature: 0.7,
            topP: 0.9,
            topK: nil,
            minP: nil,
            repetitionPenalty: 1.1,
            maxTokens: 2048,
            contextWindow: 8192,
            enableSpeculative: true,
            streamExperts: true
        ),
        runtimeId: "mlx.streaming_moe",
        icon: "bolt.circle.fill"
    )
    
    public static let gemma4_26B_baseline = RuntimePreset(
        id: "gemma4.moe.baseline",
        name: "Gemma 4 26B - Baseline",
        description: "Stable 61.5 tok/s baseline without MTP. SSD expert streaming only.",
        config: RuntimeConfig(
            temperature: 0.7,
            topP: 0.9,
            topK: nil,
            minP: nil,
            repetitionPenalty: 1.1,
            maxTokens: 2048,
            contextWindow: 8192,
            streamExperts: true
        ),
        runtimeId: "mlx.streaming_moe",
        icon: "cpu.fill"
    )
    
    public static let gemma4_26B_quality = RuntimePreset(
        id: "gemma4.moe.quality",
        name: "Gemma 4 26B - Quality",
        description: "Lower temperature for coherent, focused outputs. MTP enabled.",
        config: RuntimeConfig(
            temperature: 0.5,
            topP: 0.85,
            topK: 50,
            minP: nil,
            repetitionPenalty: 1.15,
            maxTokens: 4096,
            contextWindow: 8192,
            enableSpeculative: true,
            streamExperts: true
        ),
        runtimeId: "mlx.streaming_moe",
        icon: "doc.text.fill"
    )
    
    public static let gemma4_26B_creative = RuntimePreset(
        id: "gemma4.moe.creative",
        name: "Gemma 4 26B - Creative",
        description: "Higher temperature for diverse, creative outputs. MTP enabled.",
        config: RuntimeConfig(
            temperature: 0.9,
            topP: 0.95,
            topK: nil,
            minP: nil,
            repetitionPenalty: 1.05,
            maxTokens: 2048,
            contextWindow: 8192,
            enableSpeculative: true,
            streamExperts: true
        ),
        runtimeId: "mlx.streaming_moe",
        icon: "sparkles"
    )
    
    // MARK: Preset Groups
    
    /// Get all presets for a specific runtime engine.
    public static func presets(for runtimeId: String) -> [RuntimePreset] {
        switch runtimeId {
        case "mlx.standard":
            return [standardBalanced, standardCreative, standardPrecise, standardVision]
        case "mlx.dflash":
            return [dflashLongContext, dflashUltraContext]
        case "mlx.speculative":
            return [speculativeSpeed, speculativeBalanced]
        case "mlx.streaming_moe":
            return [
                // Gemma 4 26B MoE (Bootstrap Model) - recommended first
                gemma4_26B_mtpFast, gemma4_26B_baseline,
                gemma4_26B_quality, gemma4_26B_creative,
                // Other large MoE models
                streaming122B, streaming397B, streamingDeepSeek
            ]
        default:
            return []
        }
    }
    
    /// Get all available presets.
    public static var allPresets: [RuntimePreset] {
        return [
            // Standard
            standardBalanced, standardCreative, standardPrecise, standardVision,
            // DFlash
            dflashLongContext, dflashUltraContext,
            // Speculative
            speculativeSpeed, speculativeBalanced,
            // Gemma 4 26B MoE
            gemma4_26B_mtpFast, gemma4_26B_baseline,
            gemma4_26B_quality, gemma4_26B_creative,
            // Streaming MoE
            streaming122B, streaming397B, streamingDeepSeek
        ]
    }
    
    /// Find preset by ID.
    public static func preset(id: String) -> RuntimePreset? {
        return allPresets.first { $0.id == id }
    }
}
