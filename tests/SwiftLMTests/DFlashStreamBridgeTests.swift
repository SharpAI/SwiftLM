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
                        try? await Task.sleep(for: .milliseconds(5))
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
            let generation = dflashGenerationStream(events.stream, tokenLimit: 1_000) { "t\($0) " }
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
        // The detokenizer seam may hold text back (nil) and release it on a later token.
        for await item in dflashGenerationStream(events, tokenLimit: 3, nextText: { $0 == 1 ? nil : "one two" }) {
            switch item {
            case .chunk(let text, _): chunks.append(text)
            case .info(let i): info = i
            default: break
            }
        }
        XCTAssertEqual(chunks, ["one two"])
        XCTAssertEqual(info?.generationTokenCount, 3)
        XCTAssertEqual(info?.promptTokenCount, 7)
        XCTAssertEqual(info?.stopReason, .length, "3 generated against a limit of 3 is a length stop")
    }
}
