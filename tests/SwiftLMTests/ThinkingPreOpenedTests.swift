import XCTest
import Foundation
@testable import SwiftLM

// MARK: - Regression tests for Issue #108 — templates that pre-open the thinking block
//
// Root cause: Qwen3/3.5/3.6 chat templates append `<think>\n` to the generation prompt
// when enable_thinking is true:
//
//     {%- if add_generation_prompt %}
//         {{- '<|im_start|>assistant\n' }}
//         {%- if enable_thinking is defined and enable_thinking is false %}
//             {{- '<think>\n\n</think>\n\n' }}
//         {%- else %}
//             {{- '<think>\n' }}          <-- prompt ends inside the block
//
// The model therefore emits reasoning with no opening tag of its own, terminated by
// `</think>`. ThinkingStateTracker started in .responding and extractThinkingBlock
// required both tags, so the whole reasoning section was misfiled as response content
// with a stray `</think>` inside it — the client never saw an opening tag, and replaying
// that assistant turn taught the model to omit its own first line.
//
// Fix: detect the unclosed tag in the rendered prompt tail and start the parser inside
// the thinking block. Also drains the tracker at end of stream so a response ending in
// a partial-tag suffix is not truncated.
final class ThinkingPreOpenedTests: XCTestCase {

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - 1. Detecting a pre-opened block in the prompt
    // ═══════════════════════════════════════════════════════════════════

    /// The enable_thinking=true branch of the Qwen3.6 template.
    func testDetectsQwenOpenThinkPrompt() {
        let tail = "<|im_start|>assistant\n<think>\n"
        XCTAssertTrue(ThinkingStateTracker.opensUnclosedThinkingBlock(tail))
    }

    /// The enable_thinking=false branch emits an *already closed* block — the model
    /// will produce plain content, so the parser must not start in thinking mode.
    func testClosedThinkBlockInPromptIsNotOpen() {
        let tail = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        XCTAssertFalse(ThinkingStateTracker.opensUnclosedThinkingBlock(tail))
    }

    /// Templates that do not touch the thinking channel at all.
    func testPlainPromptIsNotOpen() {
        XCTAssertFalse(ThinkingStateTracker.opensUnclosedThinkingBlock("<|im_start|>assistant\n"))
        XCTAssertFalse(ThinkingStateTracker.opensUnclosedThinkingBlock(""))
    }

    /// Prior turns in the prompt contain complete <think>…</think> pairs; only the
    /// final unmatched opening tag counts.
    func testPriorClosedBlocksDoNotConfuseDetection() {
        let withHistory = "<|im_start|>assistant\n<think>\nold reasoning\n</think>\nold answer<|im_end|>\n<|im_start|>assistant\n<think>\n"
        XCTAssertTrue(ThinkingStateTracker.opensUnclosedThinkingBlock(withHistory))

        let historyOnly = "<|im_start|>assistant\n<think>\nold reasoning\n</think>\nold answer<|im_end|>\n<|im_start|>assistant\n"
        XCTAssertFalse(ThinkingStateTracker.opensUnclosedThinkingBlock(historyOnly))
    }

    /// Gemma-4 channel markers use a different spelling but the same shape.
    func testGemmaChannelMarkers() {
        XCTAssertTrue(ThinkingStateTracker.opensUnclosedThinkingBlock("<|channel>thought\n"))
        XCTAssertFalse(ThinkingStateTracker.opensUnclosedThinkingBlock("<|channel>thought\n<channel|>"))
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - 2. Streaming with a pre-opened block
    // ═══════════════════════════════════════════════════════════════════

    /// The issue's exact shape: reasoning arrives first with no opening tag.
    /// Before the fix this landed in `content` together with a stray `</think>`.
    func testStreamingPreOpenedRoutesReasoningCorrectly() {
        var tracker = ThinkingStateTracker(startInThinking: true)
        var reasoning = ""
        var content = ""
        for chunk in ["Okay, the user", " wants a file listing.", "\n</think>\n", "Here are your files:"] {
            let (r, c) = tracker.process(chunk)
            reasoning += r
            content += c
        }
        let (fr, fc) = tracker.flush()
        reasoning += fr
        content += fc

        XCTAssertEqual(reasoning, "Okay, the user wants a file listing.\n")
        // The newline that followed the closing tag stays in content — streaming does
        // not trim it (pre-existing behaviour, matching the self-opened path).
        XCTAssertEqual(content, "\nHere are your files:")
        XCTAssertFalse(content.contains("</think>"), "closing tag must never leak into content")
    }

    /// The closing tag split across chunk boundaries must still be consumed.
    func testStreamingPreOpenedWithSplitClosingTag() {
        var tracker = ThinkingStateTracker(startInThinking: true)
        var reasoning = ""
        var content = ""
        for chunk in ["reasoning", "</", "think", ">", "answer"] {
            let (r, c) = tracker.process(chunk)
            reasoning += r
            content += c
        }
        XCTAssertEqual(reasoning, "reasoning")
        XCTAssertEqual(content, "answer")
    }

    /// A model that *does* emit its own opening tag (LM Studio-style, template did not
    /// pre-open) must keep working exactly as before.
    func testStreamingSelfOpenedUnchanged() {
        var tracker = ThinkingStateTracker(startInThinking: false)
        var reasoning = ""
        var content = ""
        for chunk in ["<think>", "thinking hard", "</think>", "the answer"] {
            let (r, c) = tracker.process(chunk)
            reasoning += r
            content += c
        }
        XCTAssertEqual(reasoning, "thinking hard")
        XCTAssertEqual(content, "the answer")
    }

    /// Generation cut short mid-reasoning: everything is reasoning, no content.
    func testStreamingPreOpenedUnterminated() {
        var tracker = ThinkingStateTracker(startInThinking: true)
        let (r, c) = tracker.process("still thinking when we hit the token limit")
        XCTAssertEqual(r, "still thinking when we hit the token limit")
        XCTAssertEqual(c, "")
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - 3. End-of-stream flush (tail truncation)
    // ═══════════════════════════════════════════════════════════════════

    /// A response whose final characters look like the start of a tag leaves the whole
    /// pending buffer held inside the tracker (it holds everything, not just the
    /// ambiguous suffix). Without a flush at end of stream that text was dropped —
    /// the response simply ended early.
    func testFlushEmitsHeldPartialTagAsContent() {
        var tracker = ThinkingStateTracker(startInThinking: false)
        let (r, c) = tracker.process("the value of x <")
        XCTAssertEqual(r, "")
        XCTAssertEqual(c, "", "a trailing '<' holds the whole buffer pending disambiguation")

        let (fr, fc) = tracker.flush()
        XCTAssertEqual(fr, "")
        XCTAssertEqual(fc, "the value of x <", "held text must be emitted at end of stream, not dropped")
    }

    /// Held text resolves normally when the stream continues — the flush path only
    /// matters when generation ends while text is still buffered.
    func testHeldTextIsReleasedWhenStreamContinues() {
        var tracker = ThinkingStateTracker(startInThinking: false)
        XCTAssertEqual(tracker.process("the value of x <").content, "")
        XCTAssertEqual(tracker.process(" 3 holds").content, "the value of x < 3 holds")
    }

    /// Same, but the stream ended while inside the thinking block.
    func testFlushEmitsHeldTextAsReasoningWhileThinking() {
        var tracker = ThinkingStateTracker(startInThinking: true)
        _ = tracker.process("thinking </")
        let (fr, fc) = tracker.flush()
        XCTAssertEqual(fr, "thinking </")
        XCTAssertEqual(fc, "")
    }

    func testFlushIsIdempotentAndEmptyWhenNothingHeld() {
        var tracker = ThinkingStateTracker(startInThinking: false)
        _ = tracker.process("complete response.")
        let first = tracker.flush()
        XCTAssertEqual(first.reasoning, "")
        XCTAssertEqual(first.content, "")
        let second = tracker.flush()
        XCTAssertEqual(second.reasoning, "")
        XCTAssertEqual(second.content, "")
    }

    // ═══════════════════════════════════════════════════════════════════
    // MARK: - 4. Non-streaming extraction
    // ═══════════════════════════════════════════════════════════════════

    func testExtractPreOpenedBlock() {
        let text = "Let me think about this.\n</think>\nThe answer is 42."
        let (thinking, remaining) = extractThinkingBlock(from: text, alreadyOpen: true)
        XCTAssertEqual(thinking, "Let me think about this.\n")
        XCTAssertEqual(remaining, "The answer is 42.")
    }

    /// Without the flag the old behaviour is preserved: no opening tag means the text
    /// is returned as-is. This is what the client was receiving before the fix.
    func testExtractPreOpenedBlockWithoutFlagIsUnchanged() {
        let text = "Let me think about this.\n</think>\nThe answer is 42."
        let (thinking, remaining) = extractThinkingBlock(from: text)
        XCTAssertNil(thinking)
        XCTAssertEqual(remaining, text)
    }

    /// If the model re-emits its own opening tag despite the pre-open, the tag-based
    /// path wins and must not double-count the reasoning.
    func testExtractPreOpenedButModelEmittedItsOwnTag() {
        let text = "<think>reasoning</think>answer"
        let (thinking, remaining) = extractThinkingBlock(from: text, alreadyOpen: true)
        XCTAssertEqual(thinking, "reasoning")
        XCTAssertEqual(remaining, "answer")
    }

    /// Truncated mid-reasoning: all reasoning, no content.
    func testExtractPreOpenedUnterminated() {
        let (thinking, remaining) = extractThinkingBlock(from: "unfinished reasoning", alreadyOpen: true)
        XCTAssertEqual(thinking, "unfinished reasoning")
        XCTAssertEqual(remaining, "")
    }

    /// A pre-opened block the model closed immediately produces no reasoning and all
    /// content — reasoning_content must be nil rather than an empty string.
    func testExtractPreOpenedEmptyReasoning() {
        let (thinking, remaining) = extractThinkingBlock(from: "</think>\nstraight to the answer", alreadyOpen: true)
        XCTAssertNil(thinking)
        XCTAssertEqual(remaining, "straight to the answer")
    }
}
