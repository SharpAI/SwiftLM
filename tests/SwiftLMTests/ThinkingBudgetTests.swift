import XCTest
import Foundation
import MLXLMCommon
@testable import SwiftLM

/// `--thinking-budget` / `thinking_budget` cap a request's reasoning tokens. A budget the
/// request cannot honour is a 400 before any GPU work, and a request that does not think is
/// left alone.
final class ThinkingBudgetTests: XCTestCase {

    private let tokenizer = ByteTokenizer()
    /// Qwen-style tags that the byte tokenizer can spell; Qwen3.5 declares no hard-budget
    /// transition of its own, so the server supplies `.immediate`.
    private let reasoning = ReasoningConfig(
        startDelimiter: "<think>", endDelimiter: "</think>",
        promptStrategy: .templateFlag(key: "enable_thinking", defaultOn: true),
        isSpecialToken: false)

    private func components(
        budget: Int?, thinking: Bool = true, reasoning: ReasoningConfig?? = .none, maxTokens: Int? = 4096
    ) throws -> GenerationComponents? {
        try thinkingBudgetComponents(
            budget: budget, enableThinking: thinking, reasoning: reasoning ?? self.reasoning,
            tokenizer: tokenizer, parameters: GenerateParameters(maxTokens: maxTokens))
    }

    // MARK: - When a budget applies

    func testNoBudgetMeansNoComponents() throws {
        XCTAssertNil(try components(budget: nil))
    }

    func testThinkingOffLeavesTheRequestAlone() throws {
        XCTAssertNil(try components(budget: 64, thinking: false))
    }

    func testModelWithoutAReasoningProtocolIsLeftAlone() throws {
        XCTAssertNil(try components(budget: 64, reasoning: .some(nil)))
    }

    func testBudgetBuildsComponents() throws {
        XCTAssertNotNil(try components(budget: 64))
    }

    func testZeroBudgetIsAllowed() throws {
        XCTAssertNotNil(try components(budget: 0))
    }

    // MARK: - Budgets the request cannot honour

    func testMaxTokensTooSmallForBudgetAndAnswerIsRejected() {
        XCTAssertThrowsError(try components(budget: 1000, maxTokens: 500)) { error in
            guard case .insufficientGenerationTokenLimit(_, let actual) = error as? ThinkingBudgetError else {
                return XCTFail("unexpected error \(error)")
            }
            XCTAssertEqual(actual, 500)
        }
    }

    func testUnlimitedMaxTokensAlwaysFits() throws {
        XCTAssertNotNil(try components(budget: 100_000, maxTokens: nil))
    }

    func testNegativeBudgetIsRejected() {
        XCTAssertThrowsError(try components(budget: -1)) { error in
            XCTAssertEqual(error as? ThinkingBudgetError, .invalidMaximumTokenCount(-1))
        }
    }

    // MARK: - 400 body

    func testErrorBodyIsOpenAIShaped() throws {
        let body = thinkingBudgetErrorBody(.insufficientGenerationTokenLimit(required: 1100, actual: 500))
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(body.utf8)) as? [String: Any])
        let error = try XCTUnwrap(json["error"] as? [String: Any])
        XCTAssertEqual(error["type"] as? String, "invalid_request_error")
        XCTAssertEqual(error["code"] as? String, "invalid_thinking_budget")
        XCTAssertFalse((error["message"] as? String ?? "").isEmpty)
    }

    // MARK: - Empty reasoning (budget 0)

    /// A zero budget makes the model emit `</think>` first. The handler keeps `remaining`
    /// whether or not any reasoning was extracted, so this contract must hold: no reasoning,
    /// and no closing tag left in the content.
    func testEmptyReasoningDropsTheClosingTag() {
        let (reasoning, content) = extractThinkingBlock(from: "</think>\n\n9 sheep are left.", alreadyOpen: true)
        XCTAssertNil(reasoning)
        XCTAssertEqual(content, "9 sheep are left.")
    }

    func testNoTagsLeavesTheTextUntouched() {
        let (reasoning, content) = extractThinkingBlock(from: "plain answer", alreadyOpen: false)
        XCTAssertNil(reasoning)
        XCTAssertEqual(content, "plain answer")
    }

    // MARK: - Flag and request parsing

    func testFlagParsesAndDefaultsToNil() throws {
        XCTAssertEqual(try MLXServer.parse(["--model", "m", "--thinking-budget", "2048"]).thinkingBudget, 2048)
        XCTAssertNil(try MLXServer.parse(["--model", "m"]).thinkingBudget)
    }

    func testNegativeFlagIsRejected() {
        XCTAssertThrowsError(try MLXServer.parse(["--model", "m", "--thinking-budget", "-5"]))
    }

    func testRequestFieldDecodes() throws {
        let json = #"{"model":"m","messages":[{"role":"user","content":"hi"}],"thinking_budget":128}"#
        let request = try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(json.utf8))
        XCTAssertEqual(request.thinkingBudget, 128)
    }
}

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
