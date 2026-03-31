// InferenceEngine.swift — Core MLX inference actor for SwiftLM Chat
// Extracted from Server.swift — no HTTP, no CLI, pure Swift concurrency.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// The state of the inference engine.
public enum ModelState: Equatable, Sendable {
    case idle
    case downloading(progress: Double, speed: String)
    case loading
    case ready(modelId: String)
    case generating
    case error(String)
}

/// Token-level output from the generation stream.
public struct GenerationToken: Sendable {
    public let text: String
    public let isThinking: Bool   // true when inside <think>...</think>

    public init(text: String, isThinking: Bool = false) {
        self.text = text
        self.isThinking = isThinking
    }
}

/// Thread-safe MLX inference engine. One instance per app.
/// Uses Swift actor isolation so MLX calls never race.
@MainActor
public final class InferenceEngine: ObservableObject {
    @Published public private(set) var state: ModelState = .idle

    private var container: ModelContainer?
    private var currentModelId: String?
    private var generationTask: Task<Void, Never>?

    public init() {}

    // MARK: — Model Loading

    /// Load a model by HuggingFace ID. Downloads if not cached.
    public func load(modelId: String) async {
        guard state != .ready(modelId: modelId) else { return }

        state = .loading
        currentModelId = modelId

        do {
            // Configure download progress reporting
            let config = ModelConfiguration(id: modelId)
            container = try await LLMModelFactory.shared.loadContainer(
                configuration: config
            ) { [weak self] progress in
                Task { @MainActor in
                    let pct = progress.fractionCompleted
                    let speed = progress.throughput.map { String(format: "%.1f MB/s", $0 / 1_000_000) } ?? ""
                    self?.state = .downloading(progress: pct, speed: speed)
                }
            }
            state = .ready(modelId: modelId)
        } catch {
            state = .error("Failed to load \(modelId): \(error.localizedDescription)")
            container = nil
        }
    }

    /// Unload the current model and free memory.
    public func unload() {
        generationTask?.cancel()
        container = nil
        currentModelId = nil
        state = .idle
        MLX.Memory.clearCache()
    }

    // MARK: — Generation

    /// Generate a response as an AsyncStream of tokens.
    /// Each yielded value is a `GenerationToken` (text + thinking flag).
    public nonisolated func generate(
        messages: [ChatMessage],
        config: GenerationConfig = .default
    ) -> AsyncStream<GenerationToken> {
        AsyncStream { continuation in
            Task { @MainActor in
                guard let container = self.container else {
                    continuation.finish()
                    return
                }

                self.state = .generating

                do {
                    let mlxMessages = messages.map { msg -> [String: String] in
                        ["role": msg.role.rawValue, "content": msg.content]
                    }

                    // Build MLXLMCommon GenerateParameters
                    var params = GenerateParameters(temperature: config.temperature)
                    params.topP = config.topP

                    var thinkingActive = false

                    let userInput = UserInput(messages: mlxMessages)
                    let lmInput = try await container.prepare(input: userInput)
                    let stream: AsyncStream<Generation> = try await container.generate(
                        input: lmInput,
                        parameters: params
                    )

                    var outputText = ""
                    var tokenCount = 0

                    for await generation in stream {
                        switch generation {
                        case .chunk(let text, tokenId: _):
                            outputText += text
                            tokenCount += 1

                            if tokenCount >= config.maxTokens {
                                continuation.finish()
                                break
                            }

                            // Thinking state tracking (<think> tags)
                            if config.enableThinking {
                                if outputText.contains("<think>") && !outputText.contains("</think>") {
                                    thinkingActive = true
                                } else if outputText.contains("</think>") {
                                    thinkingActive = false
                                }
                            }

                            continuation.yield(GenerationToken(text: text, isThinking: thinkingActive))

                        default:
                            break
                        }
                    }
                } catch {
                    // Yield error as a token so the UI can display it
                    continuation.yield(GenerationToken(text: "\n\n[Error: \(error.localizedDescription)]"))
                }

                self.state = self.currentModelId.map { .ready(modelId: $0) } ?? .idle
                continuation.finish()
            }
        }
    }

    /// Cancel any in-progress generation.
    public func stopGeneration() {
        generationTask?.cancel()
        generationTask = nil
        if let id = currentModelId {
            state = .ready(modelId: id)
        }
    }
}
