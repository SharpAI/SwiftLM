import XCTest
import Foundation
import Hummingbird
import HTTPTypes
import MLXLMCommon
import NIOCore
@testable import SwiftLM

// MARK: - `tool_choice` is read and honoured
//
// The request type had no `tool_choice` field, so the decoder dropped it: "none" still
// put every tool in the prompt and could come back as a tool call, a named function did
// not narrow anything, and a misspelt value was a normal 200. These cover the decoding,
// which tools reach the prompt, the 400s, and what the two chat handlers do with a tool
// call the choice ruled out (synthetic generation stream, no model needed).
final class ToolChoiceTests: XCTestCase {

    private let messages = #"[{"role":"user","content":"hi"}]"#
    private let tools = """
        [{"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}},
         {"type":"function","function":{"name":"get_time"}}]
        """

    private func decode(_ fields: String) throws -> ChatCompletionRequest {
        let json = #"{"messages":\#(messages)\#(fields.isEmpty ? "" : ",")\#(fields)}"#
        return try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(json.utf8))
    }

    private func errorObject(_ error: Error) throws -> [String: Any] {
        let decodingError = try XCTUnwrap(error as? DecodingError)
        let top = try XCTUnwrap(JSONSerialization.jsonObject(
            with: Data(invalidRequestJSON(decodingError).utf8)) as? [String: Any])
        return try XCTUnwrap(top["error"] as? [String: Any])
    }

    // MARK: Decoding

    func testTheThreeModesDecode() throws {
        XCTAssertEqual(try decode(#""tool_choice":"none""#).toolChoice, .noTools)
        XCTAssertEqual(try decode(#""tool_choice":"auto""#).toolChoice, .auto)
        XCTAssertEqual(try decode(#""tool_choice":"required""#).toolChoice, .required)
    }

    func testANamedFunctionDecodes() throws {
        let req = try decode(#""tool_choice":{"type":"function","function":{"name":"get_weather"}}"#)
        XCTAssertEqual(req.toolChoice, .function("get_weather"))
    }

    func testAbsentAndNullAreNil() throws {
        XCTAssertNil(try decode("").toolChoice)
        XCTAssertNil(try decode(#""tool_choice":null"#).toolChoice)
    }

    // MARK: Values the API does not have are a 400 that names the field

    func testAnUnknownModeIsRejected() throws {
        XCTAssertThrowsError(try decode(#""tool_choice":"sometimes""#)) { error in
            let payload = try? self.errorObject(error)
            XCTAssertEqual(payload?["type"] as? String, "invalid_request_error")
            XCTAssertEqual(payload?["param"] as? String, "tool_choice")
            XCTAssertEqual((payload?["message"] as? String)?.contains("'sometimes'"), true)
        }
    }

    func testAnObjectThatIsNotAFunctionIsRejected() throws {
        XCTAssertThrowsError(try decode(#""tool_choice":{"type":"tool","function":{"name":"x"}}"#)) { error in
            XCTAssertEqual((try? self.errorObject(error))?["param"] as? String, "tool_choice")
        }
    }

    func testOtherTypesAreRejected() {
        XCTAssertThrowsError(try decode(#""tool_choice":1"#)) { XCTAssertTrue($0 is DecodingError) }
        XCTAssertThrowsError(try decode(#""tool_choice":{"type":"function"}"#)) {
            XCTAssertTrue($0 is DecodingError)
        }
    }

    // MARK: Which tools reach the prompt

    private func offeredNames(_ choice: String?) throws -> [String]? {
        let fields = #""tools":\#(tools)"# + (choice.map { #","tool_choice":\#($0)"# } ?? "")
        return try decode(fields).offeredTools()?.map(\.function.name)
    }

    func testNoneOffersNoTools() throws {
        XCTAssertNil(try offeredNames(#""none""#))
    }

    func testANamedFunctionOffersOnlyThatTool() throws {
        XCTAssertEqual(
            try offeredNames(#"{"type":"function","function":{"name":"get_time"}}"#), ["get_time"])
    }

    func testAutoRequiredAndAbsentOfferEveryTool() throws {
        let all = ["get_weather", "get_time"]
        XCTAssertEqual(try offeredNames(nil), all)
        XCTAssertEqual(try offeredNames(#""auto""#), all)
        XCTAssertEqual(try offeredNames(#""required""#), all)
    }

    func testAChoiceWithoutToolsChangesNothing() throws {
        XCTAssertNil(try decode(#""tool_choice":"none""#).offeredTools())
        XCTAssertNil(try decode(#""tool_choice":"auto""#).offeredTools())
        XCTAssertNil(try decode(#""tool_choice":"required""#).offeredTools())
    }

    func testAFunctionThatToolsDoesNotDeclareIsA400() throws {
        let named = #""tool_choice":{"type":"function","function":{"name":"get_stock"}}"#
        for fields in [#""tools":\#(tools),\#(named)"#, named] {
            XCTAssertThrowsError(try decode(fields).offeredTools()) { error in
                let payload = try? self.errorObject(error)
                XCTAssertEqual(payload?["type"] as? String, "invalid_request_error")
                XCTAssertEqual(payload?["param"] as? String, "tool_choice")
                XCTAssertEqual((payload?["message"] as? String)?.contains("'get_stock'"), true)
            }
        }
    }

    // MARK: Which calls a reply may carry

    func testAllowedToolCallsFollowTheChoice() throws {
        XCTAssertNil(try decode("").allowedToolCalls)
        XCTAssertNil(try decode(#""tool_choice":"auto""#).allowedToolCalls)
        XCTAssertNil(try decode(#""tool_choice":"required""#).allowedToolCalls)
        XCTAssertEqual(try decode(#""tool_choice":"none""#).allowedToolCalls, [])
        XCTAssertEqual(
            try decode(#""tool_choice":{"type":"function","function":{"name":"get_time"}}"#)
                .allowedToolCalls,
            ["get_time"])
    }

    // MARK: A call the choice ruled out is not sent

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

    /// A model that calls `get_weather` although it was told not to, then stops.
    private func weatherCallStream() -> AsyncStream<Generation> {
        let (stream, continuation) = AsyncStream<Generation>.makeStream()
        continuation.yield(.toolCall(ToolCall(function: .init(name: "get_weather", arguments: ["city": "Paris"]))))
        continuation.yield(.info(GenerateCompletionInfo(
            promptTokenCount: 3, generationTokenCount: 12,
            promptTime: 0, generationTime: 0, stopReason: .stop)))
        continuation.finish()
        return stream
    }

    private func acquiredSlot() async -> GenerationSlot {
        let semaphore = AsyncSemaphore(limit: 1)
        await semaphore.wait()
        return GenerationSlot(semaphore: semaphore)
    }

    /// The tool calls and the finish reason of a non-streaming reply.
    private func nonStreaming(allowed: Set<String>?) async throws -> (toolCalls: Int, finishReason: String?) {
        let slot = await acquiredSlot()
        let response = try await handleChatNonStreaming(
            stream: weatherCallStream(), modelId: "test-model", stopSequences: [],
            promptTokenCount: 3, tokenEcho: false, allowedToolCalls: allowed, slot: slot,
            stats: ServerStats(), genStart: Date(), prefillStart: Date())
        let sink = BodySink()
        try await response.body.write(CollectingWriter(sink: sink))
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: sink.data) as? [String: Any])
        let choice = try XCTUnwrap((json["choices"] as? [[String: Any]])?.first)
        let message = try XCTUnwrap(choice["message"] as? [String: Any])
        let toolCalls = (message["tool_calls"] as? [[String: Any]]) ?? []
        return (toolCalls.count, choice["finish_reason"] as? String)
    }

    /// Same for a streamed reply; the body is drained so the consumer runs to the end.
    private func streaming(allowed: Set<String>?) async throws -> (toolCalls: Int, finishReason: String?) {
        let stream = weatherCallStream()
        let slot = await acquiredSlot()
        let response = handleChatStreaming(
            startGeneration: { (stream, nil) }, modelId: "test-model", stopSequences: [],
            includeUsage: false, promptTokenCount: 3, tokenEcho: false,
            allowedToolCalls: allowed, slot: slot, stats: ServerStats(), genStart: Date(),
            prefillStart: Date(), emitPrefillProgress: false)
        let sink = BodySink()
        try await response.body.write(CollectingWriter(sink: sink))
        var toolCalls = 0
        var finishReason: String?
        for raw in String(decoding: sink.data, as: UTF8.self).components(separatedBy: "\n") {
            let line = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard line.hasPrefix("data: {") else { continue }
            let event = try XCTUnwrap(
                JSONSerialization.jsonObject(with: Data(line.dropFirst(6).utf8)) as? [String: Any])
            guard let choice = (event["choices"] as? [[String: Any]])?.first else { continue }
            if let calls = (choice["delta"] as? [String: Any])?["tool_calls"] as? [[String: Any]] {
                toolCalls += calls.count
            }
            if let reason = choice["finish_reason"] as? String { finishReason = reason }
        }
        return (toolCalls, finishReason)
    }

    func testNoneKeepsAToolCallOutOfTheReply() async throws {
        let plain = try await nonStreaming(allowed: [])
        XCTAssertEqual(plain.toolCalls, 0)
        XCTAssertEqual(plain.finishReason, "stop")
        let streamed = try await streaming(allowed: [])
        XCTAssertEqual(streamed.toolCalls, 0)
        XCTAssertEqual(streamed.finishReason, "stop")
    }

    func testACallToAnotherFunctionThanTheNamedOneIsNotSent() async throws {
        let plain = try await nonStreaming(allowed: ["get_time"])
        XCTAssertEqual(plain.toolCalls, 0)
        let streamed = try await streaming(allowed: ["get_time"])
        XCTAssertEqual(streamed.toolCalls, 0)
    }

    func testTheNamedFunctionAndAnUnrestrictedReplyStillCall() async throws {
        for allowed in [["get_weather"], nil] as [Set<String>?] {
            let plain = try await nonStreaming(allowed: allowed)
            XCTAssertEqual(plain.toolCalls, 1)
            XCTAssertEqual(plain.finishReason, "tool_calls")
            let streamed = try await streaming(allowed: allowed)
            XCTAssertEqual(streamed.toolCalls, 1)
            XCTAssertEqual(streamed.finishReason, "tool_calls")
        }
    }
}
