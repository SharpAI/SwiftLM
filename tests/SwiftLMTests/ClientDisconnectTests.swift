import XCTest
import Foundation
import Hummingbird
import MLXLMCommon
import NIOCore
import NIOEmbedded
@testable import SwiftLM

// MARK: - Non-streaming requests stop when the client disconnects
//
// The SSE handlers cancel generation when the client goes away, because their next write
// fails and tears down the body stream. A non-streaming handler writes nothing until the
// end, so the server never noticed the client had left: the answer ran to max_tokens for
// nobody and held the generation slot the whole time. These drive the pieces with an
// EmbeddedChannel and a synthetic generation stream, no model needed.
final class ClientDisconnectTests: XCTestCase {

    // MARK: The watch

    /// Counts calls from any thread; the watch fires on the event loop.
    private final class Counter: @unchecked Sendable {
        private let lock = NSLock()
        private var count = 0
        func increment() { lock.lock(); count += 1; lock.unlock() }
        var value: Int { lock.lock(); defer { lock.unlock() }; return count }
    }

    /// Records the user events that reach the tail of the pipeline.
    private final class EventRecorder: ChannelInboundHandler {
        typealias InboundIn = ByteBuffer
        var events: [Any] = []
        func userInboundEventTriggered(context: ChannelHandlerContext, event: Any) {
            events.append(event)
            context.fireUserInboundEventTriggered(event)
        }
    }

    private func installedWatch(on channel: EmbeddedChannel, fired: Counter) throws -> EventRecorder {
        let watch = ClientDisconnectWatch { fired.increment() }
        let recorder = EventRecorder()
        try channel.pipeline.syncOperations.addHandler(watch, position: .first)
        try channel.pipeline.syncOperations.addHandler(recorder, position: .last)
        return recorder
    }

    func testWatchFiresOnceOnInputClosedAndForwardsTheEvent() throws {
        let channel = EmbeddedChannel()
        let fired = Counter()
        let recorder = try installedWatch(on: channel, fired: fired)

        channel.pipeline.fireUserInboundEventTriggered(ChannelEvent.inputClosed)
        XCTAssertEqual(fired.value, 1)
        XCTAssertEqual(recorder.events.count, 1, "the event must keep flowing to the handlers behind the watch")

        // Neither a repeat nor the eventual close fires it again.
        channel.pipeline.fireUserInboundEventTriggered(ChannelEvent.inputClosed)
        _ = try channel.finish()
        XCTAssertEqual(fired.value, 1)
    }

    func testWatchFiresOnChannelInactive() throws {
        let channel = EmbeddedChannel()
        let fired = Counter()
        _ = try installedWatch(on: channel, fired: fired)

        try channel.close().wait()
        XCTAssertEqual(fired.value, 1)
    }

    func testWatchIgnoresOtherEvents() throws {
        let channel = EmbeddedChannel()
        let fired = Counter()
        let recorder = try installedWatch(on: channel, fired: fired)

        channel.pipeline.fireUserInboundEventTriggered(ChannelEvent.outputClosed)
        XCTAssertEqual(fired.value, 0)
        XCTAssertEqual(recorder.events.count, 1)
    }

    // MARK: The wrapper around a non-streaming generation

    /// An open-ended generation: a chunk every few milliseconds until the consumer goes.
    /// `terminated` is fulfilled when the stream is torn down, which is what cancels
    /// the real producer.
    private func endlessGeneration(terminated: XCTestExpectation) -> (AsyncStream<Generation>, Task<Void, Never>) {
        let (stream, continuation) = AsyncStream<Generation>.makeStream()
        continuation.onTermination = { _ in terminated.fulfill() }
        let producer = Task {
            var token = 0
            while !Task.isCancelled {
                if case .terminated = continuation.yield(.chunk("x", tokenId: token)) { return }
                token += 1
                try? await Task.sleep(for: .milliseconds(2))
            }
        }
        return (stream, producer)
    }

    private func acquiredSlot(_ semaphore: AsyncSemaphore) async -> GenerationSlot {
        await semaphore.wait()
        return GenerationSlot(semaphore: semaphore)
    }

    func testNonStreamingGenerationStopsWhenTheClientDisconnects() async throws {
        let channel = EmbeddedChannel()
        let terminated = expectation(description: "generation stream torn down")
        let (stream, producer) = endlessGeneration(terminated: terminated)
        defer { producer.cancel() }
        let semaphore = AsyncSemaphore(limit: 1)
        let slot = await acquiredSlot(semaphore)
        let stats = ServerStats()
        await stats.requestStarted()

        let response = Task {
            try await cancellingOnClientDisconnect(channel) {
                try await handleChatNonStreaming(
                    stream: stream, modelId: "test-model", stopSequences: [], promptTokenCount: 3,
                    tokenEcho: false, slot: slot, stats: stats, genStart: Date(), prefillStart: Date())
            }
        }
        // Let it consume a few chunks, then the client goes away.
        try await Task.sleep(for: .milliseconds(40))
        channel.pipeline.fireUserInboundEventTriggered(ChannelEvent.inputClosed)

        _ = try await response.value
        await fulfillment(of: [terminated], timeout: 2)

        // The slot is free again and the request is no longer counted as active.
        let released = Task { await semaphore.wait(); return true }
        let reacquired = await withTaskGroup(of: Bool.self) { group -> Bool in
            group.addTask { await released.value }
            group.addTask { try? await Task.sleep(for: .seconds(2)); return false }
            let first = await group.next() ?? false
            group.cancelAll()
            return first
        }
        XCTAssertTrue(reacquired, "the generation slot was not released after the client disconnected")
        let snapshot = await stats.snapshot()
        XCTAssertEqual(snapshot.requestsActive, 0)
    }

    func testNonStreamingGenerationCompletesNormallyAndRemovesTheWatch() async throws {
        let channel = EmbeddedChannel()
        let (stream, continuation) = AsyncStream<Generation>.makeStream()
        for token in 1...3 { continuation.yield(.chunk("word ", tokenId: token)) }
        continuation.yield(.info(GenerateCompletionInfo(
            promptTokenCount: 3, generationTokenCount: 3, promptTime: 0, generationTime: 0,
            stopReason: .stop)))
        continuation.finish()
        let semaphore = AsyncSemaphore(limit: 1)
        let slot = await acquiredSlot(semaphore)
        let stats = ServerStats()
        await stats.requestStarted()

        let response = try await cancellingOnClientDisconnect(channel) {
            try await handleChatNonStreaming(
                stream: stream, modelId: "test-model", stopSequences: [], promptTokenCount: 3,
                tokenEcho: false, slot: slot, stats: stats, genStart: Date(), prefillStart: Date())
        }
        XCTAssertEqual(response.status, .ok)

        // The watch only covers the request: once the answer is built it is gone, so
        // the keep-alive connection closing later touches nothing.
        XCTAssertThrowsError(try channel.pipeline.syncOperations.handler(type: ClientDisconnectWatch.self))
        channel.pipeline.fireUserInboundEventTriggered(ChannelEvent.inputClosed)
        _ = try channel.finish()
    }

    func testWithoutAChannelTheBodyJustRuns() async throws {
        let value = try await cancellingOnClientDisconnect(nil) { 42 }
        XCTAssertEqual(value, 42)
    }

    func testDeferredCancellationCancelsATaskAttachedAfterTheRequest() async {
        let cancellation = DeferredCancellation()
        cancellation.cancel()
        let task = Task { try await Task.sleep(for: .seconds(10)) }
        cancellation.attach(task)
        if case .success = await task.result { XCTFail("the task should have been cancelled") }
    }
}
