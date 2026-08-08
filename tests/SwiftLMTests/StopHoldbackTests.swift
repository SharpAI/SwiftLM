import XCTest
import Foundation
@testable import SwiftLM

// MARK: - Regression tests for Issue #126 — stop sequences split across chunks
//
// The stop check only fires on the chunk that completes a stop sequence, so a stop split
// across two chunks had its first part streamed before the match was visible. These pin
// the withholding rule that closes that window.
final class StopHoldbackTests: XCTestCase {

    private let stops = ["\nUser:", "<|im_end|>"]

    func testNothingHeldWhenTailIsUnambiguous() {
        XCTAssertEqual(pendingStopPrefixLength("a normal answer.", stopSequences: stops), 0)
        XCTAssertEqual(pendingStopPrefixLength("", stopSequences: stops), 0)
    }

    /// The whole point: a partial stop at the tail must be withheld.
    func testHoldsPartialStopAtTheTail() {
        XCTAssertEqual(pendingStopPrefixLength("done.\nUser", stopSequences: stops), 5)
        XCTAssertEqual(pendingStopPrefixLength("answer<|im_", stopSequences: stops), 5)
        XCTAssertEqual(pendingStopPrefixLength("answer<", stopSequences: stops), 1)
    }

    /// A complete stop is the stop check's job, not the holdback's — withholding it here
    /// would double-handle it.
    func testCompleteStopIsNotHeld() {
        XCTAssertEqual(pendingStopPrefixLength("done.\nUser:", stopSequences: stops), 0)
        XCTAssertEqual(pendingStopPrefixLength("answer<|im_end|>", stopSequences: stops), 0)
    }

    /// With several stops sharing a prefix, the longest ambiguity wins.
    func testLongestAmbiguityWins() {
        let overlapping = ["<|end|>", "<|endoftext|>"]
        XCTAssertEqual(pendingStopPrefixLength("x<|end", stopSequences: overlapping), 5)
        // "<|end|" cannot complete "<|end|>"… it can: it is a proper prefix of length 6.
        XCTAssertEqual(pendingStopPrefixLength("x<|end|", stopSequences: overlapping), 6)
    }

    func testEmptyAndAbsentStopsAreIgnored() {
        XCTAssertEqual(pendingStopPrefixLength("anything", stopSequences: []), 0)
        XCTAssertEqual(pendingStopPrefixLength("anything", stopSequences: [""]), 0)
    }

    /// Multi-byte content must not be split mid-character; the count is in characters.
    func testUnicodeTailIsCountedInCharacters() {
        XCTAssertEqual(pendingStopPrefixLength("東京です", stopSequences: stops), 0)
        XCTAssertEqual(pendingStopPrefixLength("東京です\nUser", stopSequences: stops), 5)
    }

    /// End-to-end shape: feeding a stop one character at a time must never emit any
    /// prefix of it, and must emit everything before it exactly once.
    func testStreamSimulationNeverEmitsAStopPrefix() {
        let full = "The answer is 42.\nUser: next question"
        var emitted = ""
        var pending = ""
        var stopped = false

        for character in full {
            guard !stopped else { break }
            pending += String(character)
            if let (trimmed, _) = checkStopSequences(pending, stopSequences: stops) {
                emitted += trimmed
                stopped = true
                break
            }
            let hold = pendingStopPrefixLength(pending, stopSequences: stops)
            let safe = pending.count - hold
            if safe > 0 {
                emitted += String(pending.prefix(safe))
                pending = String(pending.dropFirst(safe))
            }
        }

        XCTAssertEqual(emitted, "The answer is 42.", "everything before the stop, and nothing after")
        XCTAssertFalse(emitted.contains("\nUser"), "no prefix of the stop sequence may be emitted")
        XCTAssertTrue(stopped)
    }
}
