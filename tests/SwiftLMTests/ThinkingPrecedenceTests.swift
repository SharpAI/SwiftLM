import XCTest
@testable import SwiftLM

// MARK: - `enable_thinking` precedence (#263)
//
// A request can switch thinking in two places, and the server has `--thinking`. The order
// is the top-level `enable_thinking`, then `chat_template_kwargs.enable_thinking`, then
// the flag. It sat inline in the chat handler where no test saw it, so swapping two of
// its branches left everything green. Each case here gives the lower levels the opposite
// value, so a swapped order fails.
final class ThinkingPrecedenceTests: XCTestCase {

    func testTheTopLevelFieldWinsOverTheKwarg() {
        XCTAssertFalse(resolveThinking(topLevel: false, kwarg: true, serverFlag: true))
        XCTAssertTrue(resolveThinking(topLevel: true, kwarg: false, serverFlag: false))
    }

    func testTheTopLevelFieldWinsOverTheServerFlag() {
        XCTAssertFalse(resolveThinking(topLevel: false, kwarg: nil, serverFlag: true))
        XCTAssertTrue(resolveThinking(topLevel: true, kwarg: nil, serverFlag: false))
    }

    func testTheKwargWinsOverTheServerFlag() {
        XCTAssertFalse(resolveThinking(topLevel: nil, kwarg: false, serverFlag: true))
        XCTAssertTrue(resolveThinking(topLevel: nil, kwarg: true, serverFlag: false))
    }

    func testTheServerFlagDecidesWhenTheRequestSaysNothing() {
        XCTAssertTrue(resolveThinking(topLevel: nil, kwarg: nil, serverFlag: true))
        XCTAssertFalse(resolveThinking(topLevel: nil, kwarg: nil, serverFlag: false))
    }
}
