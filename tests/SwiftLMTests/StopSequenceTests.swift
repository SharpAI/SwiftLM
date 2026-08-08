import XCTest
import Foundation
@testable import SwiftLM

// MARK: - Contract tests for the `stop` parameter (issue #126)
//
// The stop sequence must never reach the client. These pin the parts of that contract
// that are testable without a model; the streaming-boundary half is covered by
// tests/test-contract.sh against a live server.
final class StopSequenceTests: XCTestCase {

    func testTrimsAtTheStopSequence() {
        let result = checkStopSequences("answer<|end|>trailing", stopSequences: ["<|end|>"])
        XCTAssertEqual(result?.0, "answer")
        XCTAssertEqual(result?.1, "<|end|>")
    }

    /// Earliest *in the text*, not first in the caller's list. Returning whichever entry
    /// was listed first kept everything between the real stop and that one.
    func testPicksEarliestMatchNotFirstListed() {
        let result = checkStopSequences("abXc\nUser:", stopSequences: ["\nUser:", "X"])
        XCTAssertEqual(result?.0, "ab", "must stop at X, the earliest match in the text")
        XCTAssertEqual(result?.1, "X")
    }

    func testOrderOfStopListDoesNotMatter() {
        let a = checkStopSequences("one TWO three", stopSequences: ["TWO", "three"])
        let b = checkStopSequences("one TWO three", stopSequences: ["three", "TWO"])
        XCTAssertEqual(a?.0, b?.0)
        XCTAssertEqual(a?.0, "one ")
    }

    func testNoMatchReturnsNil() {
        XCTAssertNil(checkStopSequences("nothing here", stopSequences: ["<|end|>"]))
        XCTAssertNil(checkStopSequences("anything", stopSequences: []))
    }

    /// An empty stop string matches at index 0 and would truncate every response.
    func testEmptyStopSequenceIsIgnored() {
        XCTAssertNil(checkStopSequences("real content", stopSequences: [""]))
        let result = checkStopSequences("real<|end|>", stopSequences: ["", "<|end|>"])
        XCTAssertEqual(result?.0, "real", "an empty entry must not shadow a real one")
    }

    func testStopAtStartYieldsEmptyContent() {
        let result = checkStopSequences("<|end|>everything", stopSequences: ["<|end|>"])
        XCTAssertEqual(result?.0, "")
    }

    /// Multi-byte content must not be split mid-character.
    func testUnicodeContentIsTrimmedCleanly() {
        let result = checkStopSequences("日本の首都は東京です<|end|>", stopSequences: ["<|end|>"])
        XCTAssertEqual(result?.0, "日本の首都は東京です")
    }
}
