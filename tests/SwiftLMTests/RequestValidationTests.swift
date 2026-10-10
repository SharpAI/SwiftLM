import XCTest
import Foundation
@testable import SwiftLM

// MARK: - Request shapes the OpenAI API allows that the server used to reject
//
// `stop` is documented as "string / array / null". Both request types decoded only the
// array form, so `"stop": "\n"` — what several SDKs and llama.cpp-style clients send —
// failed to decode. And a decode failure was answered with a 500 `server_error`, which
// the OpenAI SDKs retry (they treat 5xx as transient), instead of the 400
// `invalid_request_error` that tells a client its request is the problem.
final class RequestValidationTests: XCTestCase {

    private func decodeChat(_ json: String) throws -> ChatCompletionRequest {
        try JSONDecoder().decode(ChatCompletionRequest.self, from: try XCTUnwrap(json.data(using: .utf8)))
    }

    private func decodeText(_ json: String) throws -> TextCompletionRequest {
        try JSONDecoder().decode(TextCompletionRequest.self, from: try XCTUnwrap(json.data(using: .utf8)))
    }

    private let messages = #"[{"role":"user","content":"hi"}]"#

    // MARK: stop: string | [string] | null

    func testChatStopAcceptsAString() throws {
        let req = try decodeChat(#"{"messages":\#(messages),"stop":"\n"}"#)
        XCTAssertEqual(req.stop?.values, ["\n"])
    }

    func testChatStopStillAcceptsAnArray() throws {
        let req = try decodeChat(#"{"messages":\#(messages),"stop":["a","b"]}"#)
        XCTAssertEqual(req.stop?.values, ["a", "b"])
    }

    func testChatStopNullAndAbsentAreNil() throws {
        XCTAssertNil(try decodeChat(#"{"messages":\#(messages),"stop":null}"#).stop)
        XCTAssertNil(try decodeChat(#"{"messages":\#(messages)}"#).stop)
    }

    func testTextCompletionStopAcceptsAString() throws {
        let req = try decodeText(#"{"prompt":"x","stop":"<|end|>"}"#)
        XCTAssertEqual(req.stop?.values, ["<|end|>"])
    }

    func testTextCompletionStopStillAcceptsAnArray() throws {
        let req = try decodeText(#"{"prompt":"x","stop":["<|end|>","---"]}"#)
        XCTAssertEqual(req.stop?.values, ["<|end|>", "---"])
    }

    /// Anything else is still a decode error — a number is not a stop sequence.
    func testStopRejectsOtherTypes() {
        XCTAssertThrowsError(try decodeChat(#"{"messages":\#(messages),"stop":42}"#)) { error in
            XCTAssertTrue(error is DecodingError)
        }
    }

    // MARK: Decode failures become 400 invalid_request_error

    private func errorObject(_ json: String) throws -> [String: Any] {
        let data = try XCTUnwrap(json.data(using: .utf8))
        let top = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        return try XCTUnwrap(top["error"] as? [String: Any])
    }

    func testTypeMismatchNamesTheParameter() throws {
        let error = try captureDecodingError { try decodeChat(#"{"messages":\#(messages),"stop":42}"#) }
        let payload = try errorObject(invalidRequestJSON(error))
        XCTAssertEqual(payload["type"] as? String, "invalid_request_error")
        XCTAssertEqual(payload["param"] as? String, "stop")
        XCTAssertTrue((payload["message"] as? String ?? "").contains("stop"))
    }

    func testMissingKeyNamesTheParameter() throws {
        let error = try captureDecodingError { try decodeChat(#"{"model":"x"}"#) }
        let payload = try errorObject(invalidRequestJSON(error))
        XCTAssertEqual(payload["type"] as? String, "invalid_request_error")
        XCTAssertEqual(payload["param"] as? String, "messages")
    }

    func testNestedKeyPathIsDotted() throws {
        let error = try captureDecodingError { try decodeChat(#"{"messages":[{"content":"hi"}]}"#) }
        let payload = try errorObject(invalidRequestJSON(error))
        XCTAssertEqual(payload["param"] as? String, "messages.0.role")
    }

    func testMalformedJSONHasNoParameter() throws {
        let error = try captureDecodingError { try decodeChat("{not json") }
        let payload = try errorObject(invalidRequestJSON(error))
        XCTAssertEqual(payload["type"] as? String, "invalid_request_error")
        XCTAssertNil(payload["param"])
        XCTAssertFalse((payload["message"] as? String ?? "").isEmpty)
    }

    private func captureDecodingError(_ body: () throws -> Any) throws -> DecodingError {
        do {
            _ = try body()
        } catch let error as DecodingError {
            return error
        }
        XCTFail("expected a DecodingError")
        throw XCTSkip("no DecodingError thrown")
    }
}
