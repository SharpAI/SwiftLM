import XCTest
import MLX
import MLXLMCommon
@testable import SwiftLM

// MARK: - #200: prompt cache for sliding-window (RotatingKVCache) models
//
// Gemma 4 mixes RotatingKVCache sliding-window layers with full-attention layers.
// These tests cover the pieces that make saving/restoring such a cache safe.

final class PromptCacheRotatingTests: XCTestCase {

    /// A ring buffer fed `n` tokens one chunk at a time; values encode the token index.
    private func makeRing(maxSize: Int, tokens n: Int, chunk: Int = 1) -> RotatingKVCache {
        let cache = RotatingKVCache(maxSize: maxSize, keep: 0, step: 4)
        var i = 0
        while i < n {
            let m = min(chunk, n - i)
            let k = MLXArray((i ..< i + m).map { Float($0) }).reshaped([1, 1, m, 1])
            _ = cache.update(keys: k, values: k)
            i += m
        }
        return cache
    }

    private func flat(_ a: MLXArray) -> [Float] { a.asType(.float32).asArray(Float.self) }

    func testSnapshotSurvivesDecodeOnWrappedRing() async {
        let ring = makeRing(maxSize: 16, tokens: 40)  // wrapped
        let pc = PromptCache()
        await pc.save(tokens: Array(0 ..< 40), cache: [ring])
        let fresh = RotatingKVCache(maxSize: 16, keep: 0, step: 4)

        // Decode on the live cache after saving.
        for t in 40 ..< 50 {
            let k = MLXArray([Float(t)]).reshaped([1, 1, 1, 1])
            _ = ring.update(keys: k, values: k)
        }
        let hit = await pc.restore(newTokens: Array(0 ..< 40) + [999], into: [fresh])
        XCTAssertEqual(hit, 40)
        // The restored ring must hold exactly the window of tokens 24..<40, not later ones.
        let window = flat(fresh.state[0]).sorted()
        XCTAssertEqual(window, (24 ..< 40).map { Float($0) })
    }

    func testRestoredSnapshotSurvivesDecode() async {
        let ring = makeRing(maxSize: 16, tokens: 40)
        let pc = PromptCache()
        await pc.save(tokens: Array(0 ..< 40), cache: [ring])

        let first = RotatingKVCache(maxSize: 16, keep: 0, step: 4)
        _ = await pc.restore(newTokens: Array(0 ..< 40) + [1], into: [first])
        for t in 100 ..< 110 {
            let k = MLXArray([Float(t)]).reshaped([1, 1, 1, 1])
            _ = first.update(keys: k, values: k)
        }
        let second = RotatingKVCache(maxSize: 16, keep: 0, step: 4)
        _ = await pc.restore(newTokens: Array(0 ..< 40) + [2], into: [second])
        XCTAssertEqual(flat(second.state[0]).sorted(), (24 ..< 40).map { Float($0) })
    }

    func testRestoreMissesWhenWrappedRingWouldNeedDeepTrim() async {
        let pc = PromptCache()
        await pc.save(tokens: Array(0 ..< 40), cache: [makeRing(maxSize: 16, tokens: 40)])
        let fresh = RotatingKVCache(maxSize: 16, keep: 0, step: 4)
        // Diverges 4 tokens before the end: excess 4 on an evicted ring -> miss.
        let r = await pc.restore(newTokens: Array(0 ..< 36) + [500, 501], into: [fresh])
        XCTAssertNil(r)
        // A one-token excess is exact.
        let r2 = await pc.restore(newTokens: Array(0 ..< 39) + [500, 501], into: [fresh])
        XCTAssertEqual(r2, 39)
    }

    func testRestoreAllowsDeepTrimBeforeRingWraps() async {
        let pc = PromptCache()
        await pc.save(tokens: Array(0 ..< 10), cache: [makeRing(maxSize: 16, tokens: 10)])
        let fresh = RotatingKVCache(maxSize: 16, keep: 0, step: 4)
        let r = await pc.restore(newTokens: Array(0 ..< 6) + [500], into: [fresh])
        XCTAssertEqual(r, 6)
        XCTAssertEqual(fresh.offset, 6)
    }

    func testSaveRestoreOffsets() async {
        let pc = PromptCache()
        await pc.save(tokens: Array(0 ..< 40), cache: [makeRing(maxSize: 16, tokens: 40)])
        let fresh = RotatingKVCache(maxSize: 16, keep: 0, step: 4)
        _ = await pc.restore(newTokens: Array(0 ..< 40) + [7], into: [fresh])
        XCTAssertEqual(fresh.offset, 40)
    }

    // MARK: - sliceText

    func testSliceText_PreservesRankAndMask() {
        let tokens = MLXArray((0 ..< 10).map { Int32($0) }).reshaped([1, 10])
        let mask = MLXArray.ones([1, 10], dtype: .int32)
        let text = LMInput.Text(tokens: tokens, mask: mask)
        let tail = sliceText(text, from: 4)
        XCTAssertEqual(tail.tokens.shape, [1, 6])
        XCTAssertEqual(tail.mask?.shape, [1, 6])
        XCTAssertEqual(tail.tokens.asArray(Int32.self), [4, 5, 6, 7, 8, 9])
        let mid = sliceText(text, from: 2, to: 5)
        XCTAssertEqual(mid.tokens.shape, [1, 3])
    }

    func testSliceText_OneDimensional() {
        let text = LMInput.Text(tokens: MLXArray((0 ..< 10).map { Int32($0) }))
        XCTAssertEqual(sliceText(text, from: 7).tokens.asArray(Int32.self), [7, 8, 9])
        XCTAssertEqual(sliceText(text, from: 1, to: 3).tokens.asArray(Int32.self), [1, 2])
    }

    // MARK: - rotatingCacheBoundary

    func testRotatingCacheBoundary() {
        // <|turn>=1, other=2..
        XCTAssertEqual(rotatingCacheBoundary(promptTokens: [5, 1, 6, 7, 1, 8, 9], turnStartIds: [1, nil]), 4)
        // no marker -> end of prompt
        XCTAssertEqual(rotatingCacheBoundary(promptTokens: [5, 6, 7], turnStartIds: [1, nil]), 3)
        // marker only at index 0 -> nothing before it to cache; use the full prompt
        XCTAssertEqual(rotatingCacheBoundary(promptTokens: [1, 6, 7], turnStartIds: [1]), 3)
    }
}
