import XCTest
import Foundation
import Hummingbird
import HTTPTypes
import MLXLMCommon
import NIOCore
@testable import SwiftLM

// MARK: - A tool-call-only turn saves the prompt cache (#245)
//
// Both chat handlers save the prompt cache when the producer's `.info` arrives, and used
// to do so only once a text chunk had been seen. A turn the model answers with nothing
// but a tool call never yields a chunk: the parser holds the whole `<tool_call>` body and
// emits one `.toolCall`. So the save was skipped and the next turn of the agent loop,
// tool result and follow-up, re-prefilled the whole conversation. These drive the
// handlers with a synthetic generation stream, no model needed.
final class ToolCallPromptCacheTests: XCTestCase {

    private final class Counter: @unchecked Sendable {
        private let lock = NSLock()
        private var count = 0
        func increment() { lock.lock(); count += 1; lock.unlock() }
        var value: Int { lock.lock(); defer { lock.unlock() }; return count }
    }

    private final class BodySink: @unchecked Sendable {
        var data = Data()
    }

    private struct CollectingWriter: ResponseBodyWriter {
        let sink: BodySink
        mutating func write(_ buffer: ByteBuffer) async throws {
            sink.data.append(contentsOf: buffer.readableBytesView)
        }
        consuming func finish(_ trailingHeaders: HTTPFields?) async throws {}
    }

    private func toolCall() -> Generation {
        .toolCall(ToolCall(function: .init(name: "get_weather", arguments: ["city": "Paris"])))
    }

    private func info(tokens: Int) -> Generation {
        .info(GenerateCompletionInfo(
            promptTokenCount: 3, generationTokenCount: tokens,
            promptTime: 0, generationTime: 0, stopReason: .stop))
    }

    private func finishedStream(_ events: [Generation]) -> AsyncStream<Generation> {
        let (stream, continuation) = AsyncStream<Generation>.makeStream()
        for event in events { continuation.yield(event) }
        continuation.finish()
        return stream
    }

    private func acquiredSlot() async -> GenerationSlot {
        let semaphore = AsyncSemaphore(limit: 1)
        await semaphore.wait()
        return GenerationSlot(semaphore: semaphore)
    }

    /// Runs `events` through the non-streaming handler and returns how many times the
    /// prompt-cache save ran, plus the tool calls the response carries.
    private func nonStreamingSaves(for events: [Generation]) async throws -> (saves: Int, toolCalls: Int) {
        let saves = Counter()
        let slot = await acquiredSlot()
        let response = try await handleChatNonStreaming(
            stream: finishedStream(events), modelId: "test-model", stopSequences: [],
            promptTokenCount: 3, tokenEcho: false, slot: slot, stats: ServerStats(),
            genStart: Date(), prefillStart: Date(), onPrefillDone: { saves.increment() })
        let sink = BodySink()
        try await response.body.write(CollectingWriter(sink: sink))
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: sink.data) as? [String: Any])
        let choices = try XCTUnwrap(json["choices"] as? [[String: Any]])
        let message = try XCTUnwrap(choices.first?["message"] as? [String: Any])
        let toolCalls = (message["tool_calls"] as? [[String: Any]]) ?? []
        return (saves.value, toolCalls.count)
    }

    /// Same for the streaming handler; the body is drained so the consumer runs to the end.
    private func streamingSaves(for events: [Generation]) async throws -> (saves: Int, toolCalls: Int) {
        let saves = Counter()
        let stream = finishedStream(events)
        let slot = await acquiredSlot()
        let response = handleChatStreaming(
            startGeneration: { (stream, { saves.increment() }) }, modelId: "test-model",
            stopSequences: [], includeUsage: false, promptTokenCount: 3, tokenEcho: false,
            slot: slot, stats: ServerStats(), genStart: Date(), prefillStart: Date(),
            emitPrefillProgress: false)
        let sink = BodySink()
        try await response.body.write(CollectingWriter(sink: sink))
        let body = String(decoding: sink.data, as: UTF8.self)
        var toolCalls = 0
        for raw in body.components(separatedBy: "\n") {
            let line = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard line.hasPrefix("data: {") else { continue }
            let event = try XCTUnwrap(
                JSONSerialization.jsonObject(with: Data(line.dropFirst(6).utf8)) as? [String: Any])
            let choices = (event["choices"] as? [[String: Any]]) ?? []
            if let delta = choices.first?["delta"] as? [String: Any],
               let calls = delta["tool_calls"] as? [[String: Any]] {
                toolCalls += calls.count
            }
        }
        return (saves.value, toolCalls)
    }

    // MARK: The bug itself

    func testNonStreamingToolCallOnlyTurnSavesThePrompt() async throws {
        let result = try await nonStreamingSaves(for: [toolCall(), info(tokens: 12)])
        XCTAssertEqual(result.toolCalls, 1)
        XCTAssertEqual(result.saves, 1, "a turn that is only a tool call must still save its prompt")
    }

    func testStreamingToolCallOnlyTurnSavesThePrompt() async throws {
        let result = try await streamingSaves(for: [toolCall(), info(tokens: 12)])
        XCTAssertEqual(result.toolCalls, 1)
        XCTAssertEqual(result.saves, 1, "a turn that is only a tool call must still save its prompt")
    }

    // MARK: What stays the same

    func testTextTurnsStillSaveOnce() async throws {
        let text: [Generation] = [.chunk("Sunny", tokenId: 1), .chunk(".", tokenId: 2), info(tokens: 2)]
        let nonStreaming = try await nonStreamingSaves(for: text)
        XCTAssertEqual(nonStreaming.saves, 1)
        let streaming = try await streamingSaves(for: text)
        XCTAssertEqual(streaming.saves, 1)
    }

    /// An early exit (stop sequence, disconnect, error) never reaches `.info` and must
    /// keep skipping the save: the cache may still be mid-write.
    func testAStreamThatEndsWithoutInfoDoesNotSave() async throws {
        let cut: [Generation] = [toolCall()]
        let nonStreaming = try await nonStreamingSaves(for: cut)
        XCTAssertEqual(nonStreaming.saves, 0)
        let streaming = try await streamingSaves(for: cut)
        XCTAssertEqual(streaming.saves, 0)
    }
}
