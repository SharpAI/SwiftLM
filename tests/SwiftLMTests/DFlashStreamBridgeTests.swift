import XCTest
import Foundation
@testable import DFlash
import MLXLMCommon
@testable import SwiftLM

// MARK: - The DFlash → Generation bridge (issue #224)
//
// The bridge used to be a bare `Task` that nobody cancelled. When the SSE handler ended
// a response early (a text-level stop sequence, a client disconnect) the handler released
// its slot, but the bridge kept draining the DFlash stream and the DFlash loop kept
// generating up to `max_tokens`; the next request then ran a second generation on the
// same model and MLX aborted. These pin the two halves of the fix that need no model:
// the stop-token set handed to DFlash, and cancellation reaching the DFlash stream.
final class DFlashStreamBridgeTests: XCTestCase {

    // MARK: stopTokenIDs(…) — same union the standard path stops on

    func testStopTokenIDsUnionsConfigurationTokenizerAndExtraTokens() {
        let ids = stopTokenIDs(
            eosTokenIDs: [151645, 151643],
            tokenizerEOS: 151645,
            extraEOSTokens: ["<|endoftext|>", "<|im_end|>", "<|not_in_vocab|>"],
            tokenID: { ["<|endoftext|>": 151643, "<|im_end|>": 151645][$0] })
        XCTAssertEqual(ids, [151643, 151645], "deduplicated; unknown extra tokens are dropped")
    }

    func testStopTokenIDsWithoutTokenizerEOS() {
        XCTAssertEqual(stopTokenIDs(eosTokenIDs: [2], tokenizerEOS: nil, extraEOSTokens: [], tokenID: { _ in nil }), [2])
        XCTAssertEqual(stopTokenIDs(eosTokenIDs: [], tokenizerEOS: nil, extraEOSTokens: [], tokenID: { _ in nil }), [])
    }

    // MARK: dflashGenerationStream(…) — tearing down the consumer cancels the producer

    /// A DFlash-shaped stream that never ends on its own, and reports whether the
    /// consumer side tore it down.
    private final class EndlessEvents: @unchecked Sendable {
        let terminated = XCTestExpectation(description: "DFlash stream terminated")
        private(set) var yielded = 0
        let stream: AsyncStream<DFlashEvent>

        init() {
            let terminated = self.terminated
            var counter = 0
            stream = AsyncStream<DFlashEvent> { continuation in
                let producer = Task {
                    while !Task.isCancelled {
                        counter += 1
                        continuation.yield(.token(tokenID: counter, generatedTokens: counter, acceptanceRatio: 0, cyclesCompleted: counter))
                        try? await Task.sleep(nanoseconds: 5_000_000)
                    }
                    continuation.finish()
                }
                continuation.onTermination = { _ in
                    producer.cancel()
                    terminated.fulfill()
                }
            }
        }
    }

    func testStreamTornDownAfterConsumerStopsEarly() async throws {
        let events = EndlessEvents()
        var received: [String] = []
        do {
            // Scoped like the handler's `for await` over the generation stream: once the
            // consumer breaks out and the stream value goes away, the bridge must be
            // torn down and the DFlash stream with it.
            let generation = dflashGenerationStream(
                events.stream, tokenLimit: 1_000, stopTokenIDs: [], handler: SpellingHandler())
            for await item in generation {
                if case .chunk(let text, _) = item { received.append(text) }
                if received.count == 3 { break }
            }
        }
        XCTAssertEqual(received, ["t1 ", "t2 ", "t3 "])
        await fulfillment(of: [events.terminated], timeout: 2)
    }

    func testBridgeMapsTokensAndSummary() async throws {
        let summary = DFlashSummary(
            elapsedUs: 2_000_000, promptTokenCount: 7, generatedTokenIDs: [1, 2, 3],
            acceptedFromDraft: 1, acceptanceRatio: 0.33, blockTokens: 16, cyclesCompleted: 2,
            phaseTimingsUs: .init(prefill: 500_000, draft: 0, verify: 0, replay: 0))
        let (events, continuation) = AsyncStream<DFlashEvent>.makeStream()
        continuation.yield(.prefill(promptTokenCount: 7, prefillUs: 500_000))
        continuation.yield(.token(tokenID: 1, generatedTokens: 1, acceptanceRatio: 0, cyclesCompleted: 1))
        continuation.yield(.token(tokenID: 2, generatedTokens: 2, acceptanceRatio: 0, cyclesCompleted: 1))
        continuation.yield(.summary(summary))
        continuation.finish()

        var chunks: [String] = []
        var info: GenerateCompletionInfo?
        // A handler may hold text back and release it later, or at the end of the generation.
        let handler = SpellingHandler(holdBack: [1], flushAtEnd: "end ")
        for await item in dflashGenerationStream(events, tokenLimit: 3, stopTokenIDs: [], handler: handler) {
            switch item {
            case .chunk(let text, _): chunks.append(text)
            case .info(let i): info = i
            default: break
            }
        }
        XCTAssertEqual(chunks, ["t2 ", "end "], "held-back text is released by onGenerationEnd before .info")
        XCTAssertEqual(info?.generationTokenCount, 3)
        XCTAssertEqual(info?.promptTokenCount, 7)
        XCTAssertEqual(info?.stopReason, .length, "3 generated against a limit of 3 is a length stop")
    }

    // MARK: the handler sees tokens the way the standard loop shows them to it

    /// DFlash yields the stop token it ends on. The standard loop hands such a token to
    /// `onStopToken`, which a handler ignores unless it asked for stop tokens, so the end
    /// marker never reaches the client as text.
    func testStopTokenGoesToOnStopTokenNotOnToken() async throws {
        let (events, continuation) = AsyncStream<DFlashEvent>.makeStream()
        continuation.yield(.token(tokenID: 5, generatedTokens: 1, acceptanceRatio: 0, cyclesCompleted: 1))
        continuation.yield(.token(tokenID: 99, generatedTokens: 2, acceptanceRatio: 0, cyclesCompleted: 1))
        continuation.yield(.summary(DFlashSummary(
            elapsedUs: 1_000, promptTokenCount: 1, generatedTokenIDs: [5, 99], acceptedFromDraft: 0,
            acceptanceRatio: 0, blockTokens: 16, cyclesCompleted: 1,
            phaseTimingsUs: .init(prefill: 0, draft: 0, verify: 0, replay: 0))))
        continuation.finish()

        let seen = SpellingHandler.Seen()
        var chunks: [String] = []
        for await item in dflashGenerationStream(
            events, tokenLimit: 10, stopTokenIDs: [99], handler: SpellingHandler(seen: seen))
        {
            if case .chunk(let text, _) = item { chunks.append(text) }
        }
        XCTAssertEqual(chunks, ["t5 "])
        XCTAssertEqual(seen.tokens, [5])
        XCTAssertEqual(seen.stopTokens, [99])
    }

    /// A handler that ends the generation (one of the model's stop strings) ends the
    /// stream: the bridge flushes it, reports the counts so far, and tears the DFlash
    /// stream down instead of draining it to max_tokens.
    func testHandlerStopEndsTheStreamAndTearsDownTheProducer() async throws {
        let events = EndlessEvents()
        var chunks: [String] = []
        var info: GenerateCompletionInfo?
        do {
            let generation = dflashGenerationStream(
                events.stream, tokenLimit: 1_000, stopTokenIDs: [],
                handler: SpellingHandler(stopAt: 3, flushAtEnd: "end "))
            for await item in generation {
                switch item {
                case .chunk(let text, _): chunks.append(text)
                case .info(let i): info = i
                default: break
                }
            }
        }
        XCTAssertEqual(chunks, ["t1 ", "t2 ", "end "], "token 3 stops: flushed, not spelled")
        XCTAssertEqual(info?.generationTokenCount, 3)
        XCTAssertEqual(info?.stopReason, .stop)
        await fulfillment(of: [events.terminated], timeout: 2)
    }

    // MARK: the real handler: a tool call comes out as a tool call, not as text

    /// With the bare detokenizer the DFlash path used before, a model's `<tool_call>` block
    /// was streamed to the client as content. Through the standard path's handler it is a
    /// `.toolCall` event, as it is without `--dflash`.
    func testRealHandlerTurnsAToolCallIntoAToolCallEvent() async throws {
        let text = #"Checking.<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>"#
        let tokenizer = ByteTokenizer()
        let (events, continuation) = AsyncStream<DFlashEvent>.makeStream()
        var count = 0
        for id in tokenizer.encode(text: text, addSpecialTokens: false) {
            count += 1
            continuation.yield(.token(tokenID: id, generatedTokens: count, acceptanceRatio: 0, cyclesCompleted: 1))
        }
        continuation.yield(.summary(DFlashSummary(
            elapsedUs: 1_000, promptTokenCount: 1, generatedTokenIDs: Array(repeating: 0, count: count),
            acceptedFromDraft: 0, acceptanceRatio: 0, blockTokens: 16, cyclesCompleted: 1,
            phaseTimingsUs: .init(prefill: 0, draft: 0, verify: 0, replay: 0))))
        continuation.finish()

        let handler = TextToolTokenLoopHandler(
            tokenizer: tokenizer, format: .json,
            tools: [["type": "function", "function": ["name": "get_weather"] as [String: any Sendable]]])
        var content = ""
        var calls: [ToolCall] = []
        var info: GenerateCompletionInfo?
        for await item in dflashGenerationStream(events, tokenLimit: 1_000, stopTokenIDs: [], handler: handler) {
            switch item {
            case .chunk(let text, _): content += text
            case .toolCall(let call): calls.append(call)
            case .info(let i): info = i
            default: break
            }
        }
        XCTAssertEqual(calls.map(\.function.name), ["get_weather"])
        XCTAssertEqual(calls.first?.function.arguments["city"]?.anyValue as? String, "Paris")
        XCTAssertFalse(content.contains("<tool_call>"), "the tool call was handed over as text: \(content)")
        XCTAssertEqual(content, "Checking.")
        XCTAssertEqual(info?.stopReason, .stop)
    }
}

// MARK: - Test doubles

/// A `TokenLoopHandler` that spells tokens as `t<id> `, like the closure the bridge used
/// to take, and records which hook each token reached.
private struct SpellingHandler: TokenLoopHandler {
    typealias Output = Generation

    final class Seen: @unchecked Sendable {
        var tokens: [Int] = []
        var stopTokens: [Int] = []
    }

    var seen = Seen()
    var holdBack: Set<Int> = []
    var stopAt: Int? = nil
    var flushAtEnd: String? = nil

    init(seen: Seen = Seen(), holdBack: Set<Int> = [], stopAt: Int? = nil, flushAtEnd: String? = nil) {
        self.seen = seen
        self.holdBack = holdBack
        self.stopAt = stopAt
        self.flushAtEnd = flushAtEnd
    }

    mutating func onToken(
        _ token: Int, logProbabilities: DeferredTokenLogProbabilities?,
        emit: (sending Generation) -> Bool
    ) -> TokenLoopDisposition {
        seen.tokens.append(token)
        if token == stopAt { return .stop }
        if holdBack.contains(token) { return .more }
        return emit(.chunk("t\(token) ", tokenId: token)) ? .more : .cancelled
    }

    mutating func onStopToken(
        _ token: Int, logProbabilities: DeferredTokenLogProbabilities?,
        emit: (sending Generation) -> Bool
    ) -> TokenLoopDisposition {
        seen.stopTokens.append(token)
        return .more
    }

    mutating func onGenerationEnd(emit: (sending Generation) -> Bool) -> TokenLoopDisposition {
        if let flushAtEnd { _ = emit(.chunk(flushAtEnd, tokenId: -1)) }
        return .more
    }

    func infoEvent(_ info: GenerateCompletionInfo) -> Generation { .info(info) }
}

/// One token per UTF-8 byte, enough to run the real text-and-tool handler without a model.
private struct ByteTokenizer: MLXLMCommon.Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { text.utf8.map(Int.init) }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(decoding: tokenIds.map(UInt8.init), as: UTF8.self)
    }
    func convertTokenToId(_ token: String) -> Int? {
        guard token.utf8.count == 1 else { return nil }
        return token.utf8.first.map(Int.init)
    }
    func convertIdToToken(_ id: Int) -> String? {
        guard (0 ... 255).contains(id) else { return nil }
        return String(decoding: [UInt8(id)], as: UTF8.self)
    }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}
