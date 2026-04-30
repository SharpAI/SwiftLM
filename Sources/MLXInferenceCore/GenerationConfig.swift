// GenerationConfig.swift — SwiftLM inference parameters
import Foundation

/// Configuration for a single generation request.
///
/// Conforms to `Codable` so settings can be persisted across app launches
/// via `save()` / `load()` using `UserDefaults`.
///
/// ### Notes on removed fields
/// - `streamExperts` was removed: expert streaming is a **load-time** flag
///   automatically derived from `ModelCatalog.isMoE` inside `InferenceEngine.load()`.
///   Exposing it as a per-request toggle had no effect and misled users.
/// - `turboKV` was removed: the PolarQuant+QJL path was never wired into
///   `GenerateParameters` or the mlx-lm call chain. Use `kvBits: 4` or `kvBits: 8`
///   for KV-cache quantisation instead.
public struct GenerationConfig: Sendable, Codable {
    public var maxTokens: Int
    public var temperature: Float
    public var topP: Float
    public var topK: Int
    public var minP: Float
    public var repetitionPenalty: Float

    /// Optional RNG seed for reproducible outputs.
    /// When non-nil, `MLX.seed(UInt32(seed!))` is called before each generation.
    public var seed: UInt64?

    public var enableThinking: Bool

    /// Chunk size for prefill evaluation.
    /// Lower values prevent GPU timeout on large models.
    public var prefillSize: Int

    /// KV-cache quantization bits (nil = no quantization, 4 or 8 typical).
    public var kvBits: Int?

    /// KV-cache quantization group size (default 64).
    public var kvGroupSize: Int

    public init(
        maxTokens: Int = 2048,
        temperature: Float = 0.6,
        topP: Float = 1.0,
        topK: Int = 50,
        minP: Float = 0.0,
        repetitionPenalty: Float = 1.05,
        seed: UInt64? = nil,
        enableThinking: Bool = false,
        prefillSize: Int = 512,
        kvBits: Int? = nil,
        kvGroupSize: Int = 64
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
        self.enableThinking = enableThinking
        self.prefillSize = prefillSize
        self.kvBits = kvBits
        self.kvGroupSize = kvGroupSize
    }

    public static let `default` = GenerationConfig()

    // MARK: — Persistence

    private static let storageKey = "swiftlm.generationConfig"

    /// Persist this config to `UserDefaults`.
    public func save() {
        guard let data = try? JSONEncoder().encode(self) else { return }
        UserDefaults.standard.set(data, forKey: Self.storageKey)
    }

    /// Load previously persisted config, falling back to `.default`.
    public static func load() -> GenerationConfig {
        guard let data = UserDefaults.standard.data(forKey: storageKey),
              let decoded = try? JSONDecoder().decode(GenerationConfig.self, from: data)
        else { return .default }
        return decoded
    }
}
