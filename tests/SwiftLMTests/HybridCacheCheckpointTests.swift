import XCTest
import MLX
import MLXLMCommon
@testable import SwiftLM

/// A hybrid prompt that edits earlier history diverges from the cached one, and recurrent
/// state can't be rewound. Checkpoints taken during prefill let it resume from the last
/// snapshot before the edit and re-prefill only what follows.
final class HybridCacheCheckpointTests: XCTestCase {

    private let imStart = 7

    private func makeHybridCache(seqLen T: Int, fill: Float) -> [any KVCache] {
        let attn = KVCacheSimple()
        _ = attn.update(keys: MLXArray.ones([1, 2, T, 4], dtype: .float16) * fill,
                        values: MLXArray.ones([1, 2, T, 4], dtype: .float16) * fill)
        let mamba = MambaCache()
        mamba.state = [MLXArray.ones([1, 3, 8]) * fill, MLXArray.ones([1, 2, 4, 4]) * fill]
        return [attn, mamba]
    }

    /// Recurrent state of layer 1 as the cache's second array (32 elements) filled with `fill`.
    private func checkpoint(at position: Int, fill: Float) -> RecurrentCheckpoint {
        RecurrentCheckpoint(position: position, layers: [
            1: [MLXArray.ones([1, 3, 8]) * fill, MLXArray.ones([1, 2, 4, 4]) * fill],
        ])
    }

    private func recurrentSum(_ cache: [any KVCache]) -> Float {
        cache[1].state[1].sum().item(Float.self)
    }

    // MARK: - Choosing checkpoint positions

    func testColdPrefillTakesAnchorThenSpacedTurnStarts() {
        // Turn starts at 0, 3, 6, 12, 20 and boundary 20.
        var tokens = Array(repeating: 1, count: 25)
        for i in [0, 3, 6, 12, 20] { tokens[i] = imStart }
        let cuts = hybridCheckpointPositions(promptTokens: tokens, imStartId: imStart,
                                             start: 0, boundary: 20, minGap: 5)
        XCTAssertEqual(cuts, [3, 12], "anchor after the system prompt, then the first start 5+ tokens on")
    }

    func testResumedPrefillSkipsAnchorAndCountsGapFromResumePoint() {
        var tokens = Array(repeating: 1, count: 30)
        for i in [0, 3, 8, 14, 25] { tokens[i] = imStart }
        let cuts = hybridCheckpointPositions(promptTokens: tokens, imStartId: imStart,
                                             start: 6, boundary: 25, minGap: 5)
        XCTAssertEqual(cuts, [14], "8 is only 2 past the resume point; 14 is 8 past it")
    }

    func testNoCheckpointsWithoutImStartOrBeforeAnchor() {
        XCTAssertEqual(hybridCheckpointPositions(promptTokens: [1, 2, 3], imStartId: nil,
                                                 start: 0, boundary: 3), [])
        // Only the boundary turn follows the system prompt: nothing to checkpoint.
        XCTAssertEqual(hybridCheckpointPositions(promptTokens: [imStart, 1, imStart, 2], imStartId: imStart,
                                                 start: 0, boundary: 2), [])
    }

    // MARK: - Thinning

    func testThinningKeepsAnchorAndNewestAndDropsTheClosestPair() {
        let all = [100, 2000, 2100, 4500, 7000, 9500].map { checkpoint(at: $0, fill: 1) }
        let kept = PromptCache.mergedCheckpoints(all, before: 20_000).map(\.position)
        XCTAssertEqual(kept.count, PromptCache.maxCheckpoints)
        XCTAssertEqual(kept.first, 100, "anchor stays")
        XCTAssertEqual(kept.last, 9500, "newest stays")
        XCTAssertFalse(kept.contains(2000) && kept.contains(2100), "the 100-token gap is thinned first")
    }

    func testMergeDropsPositionsAtOrPastEntryEndAndDuplicates() {
        let all = [checkpoint(at: 5, fill: 1), checkpoint(at: 5, fill: 2), checkpoint(at: 9, fill: 3),
                   checkpoint(at: 10, fill: 4)]
        XCTAssertEqual(PromptCache.mergedCheckpoints(all, before: 10).map(\.position), [5, 9])
    }

    // MARK: - Restoring from a checkpoint

    func testDivergentPromptResumesFromLatestCheckpointBeforeTheEdit() async {
        let pc = PromptCache()
        await pc.save(tokens: [1, 2, 3, 4, 5, 6], cache: makeHybridCache(seqLen: 6, fill: 9),
                      allowRecurrent: true,
                      checkpoints: [checkpoint(at: 2, fill: 2), checkpoint(at: 4, fill: 4)])

        // Edit at index 4: tokens 0..<4 are shared.
        let fresh: [any KVCache] = [KVCacheSimple(), MambaCache()]
        let n = await pc.restoreExactPrefix(newTokens: [1, 2, 3, 4, 8, 8, 8], limit: 7, into: fresh)

        XCTAssertEqual(n, 4)
        XCTAssertEqual(fresh[0].offset, 4, "attention layer rewound to the checkpoint")
        XCTAssertEqual(recurrentSum(fresh), 4 * 32, "recurrent state is the checkpoint's, not the entry's")
    }

    func testCheckpointAfterTheDivergenceIsNotUsed() async {
        let pc = PromptCache()
        await pc.save(tokens: [1, 2, 3, 4, 5, 6], cache: makeHybridCache(seqLen: 6, fill: 9),
                      allowRecurrent: true, checkpoints: [checkpoint(at: 5, fill: 5)])
        let n = await pc.restoreExactPrefix(newTokens: [1, 2, 3, 9, 9, 9, 9], limit: 7,
                                            into: [KVCacheSimple(), MambaCache()])
        XCTAssertNil(n, "the only checkpoint is past where the prompts diverge")
    }

    func testExactPrefixStillBeatsAnEarlierCheckpoint() async {
        let pc = PromptCache()
        await pc.save(tokens: [1, 2, 3, 4], cache: makeHybridCache(seqLen: 4, fill: 4),
                      allowRecurrent: true, checkpoints: [checkpoint(at: 2, fill: 2)])
        let fresh: [any KVCache] = [KVCacheSimple(), MambaCache()]
        let n = await pc.restoreExactPrefix(newTokens: [1, 2, 3, 4, 5, 6], limit: 6, into: fresh)
        XCTAssertEqual(n, 4)
        XCTAssertEqual(recurrentSum(fresh), 4 * 32)
    }

    func testEntryWithLongestResumePointWinsAcrossEntries() async {
        let pc = PromptCache(maxEntries: 2)
        await pc.save(tokens: [1, 2, 3, 4, 5], cache: makeHybridCache(seqLen: 5, fill: 1),
                      allowRecurrent: true, checkpoints: [checkpoint(at: 2, fill: 2)])
        await pc.save(tokens: [1, 2, 3, 4, 6, 6], cache: makeHybridCache(seqLen: 6, fill: 1),
                      allowRecurrent: true, checkpoints: [checkpoint(at: 3, fill: 3)])
        let fresh: [any KVCache] = [KVCacheSimple(), MambaCache()]
        let n = await pc.restoreExactPrefix(newTokens: [1, 2, 3, 4, 9, 9], limit: 6, into: fresh)
        XCTAssertEqual(n, 3)
        XCTAssertEqual(recurrentSum(fresh), 3 * 32)
    }

    func testWrappedRingRefusesCheckpointRestore() async {
        // A ring that has dropped tokens can't be rewound exactly.
        let ring = RotatingKVCache(maxSize: 4, keep: 0, step: 4)
        for t in 0 ..< 6 {
            let k = MLXArray([Float(t)]).reshaped([1, 1, 1, 1])
            _ = ring.update(keys: k, values: k)
        }
        let mamba = MambaCache()
        mamba.state = [MLXArray.ones([1, 3, 8]), MLXArray.ones([1, 2, 4, 4])]
        let pc = PromptCache()
        await pc.save(tokens: [1, 2, 3, 4, 5, 6], cache: [ring, mamba], allowRecurrent: true,
                      checkpoints: [checkpoint(at: 2, fill: 2)])
        let n = await pc.restoreExactPrefix(newTokens: [1, 2, 9, 9], limit: 4,
                                            into: [RotatingKVCache(maxSize: 4, keep: 0, step: 4), MambaCache()])
        XCTAssertNil(n)
    }

    func testUnwrappedRingRestoresFromCheckpoint() async {
        let ring = RotatingKVCache(maxSize: 16, keep: 0, step: 4)
        for t in 0 ..< 6 {
            let k = MLXArray([Float(t)]).reshaped([1, 1, 1, 1])
            _ = ring.update(keys: k, values: k)
        }
        let mamba = MambaCache()
        mamba.state = [MLXArray.ones([1, 3, 8]), MLXArray.ones([1, 2, 4, 4])]
        let pc = PromptCache()
        await pc.save(tokens: [1, 2, 3, 4, 5, 6], cache: [ring, mamba], allowRecurrent: true,
                      checkpoints: [checkpoint(at: 2, fill: 2)])
        let fresh: [any KVCache] = [RotatingKVCache(maxSize: 16, keep: 0, step: 4), MambaCache()]
        let n = await pc.restoreExactPrefix(newTokens: [1, 2, 9, 9], limit: 4, into: fresh)
        XCTAssertEqual(n, 2)
        XCTAssertEqual(fresh[0].offset, 2)
        XCTAssertEqual(recurrentSum(fresh), 2 * 32)
    }

    // MARK: - Checkpoints carry across turns

    func testExtensionInheritsCheckpointsAndKeepsThePreviousEndState() async {
        let pc = PromptCache()
        await pc.save(tokens: [1, 2, 3, 4], cache: makeHybridCache(seqLen: 4, fill: 4),
                      allowRecurrent: true, checkpoints: [checkpoint(at: 2, fill: 2)])
        // The next turn resumes from the whole entry, prefills on, and saves.
        let live: [any KVCache] = [KVCacheSimple(), MambaCache()]
        let resumed = await pc.restoreExactPrefix(newTokens: [1, 2, 3, 4, 5, 6, 7], limit: 7, into: live)
        XCTAssertEqual(resumed, 4)
        await pc.save(tokens: [1, 2, 3, 4, 5, 6, 7], cache: makeHybridCache(seqLen: 7, fill: 7),
                      allowRecurrent: true, checkpoints: [], resumedFrom: 4)

        // An edit at index 5 can now resume from the old end state at 4 ...
        let a: [any KVCache] = [KVCacheSimple(), MambaCache()]
        let nA = await pc.restoreExactPrefix(newTokens: [1, 2, 3, 4, 5, 8, 8], limit: 7, into: a)
        XCTAssertEqual(nA, 4)
        XCTAssertEqual(recurrentSum(a), 4 * 32)
        // ... and an edit at index 2 from the inherited checkpoint at 2.
        let b: [any KVCache] = [KVCacheSimple(), MambaCache()]
        let nB = await pc.restoreExactPrefix(newTokens: [1, 2, 8, 8, 8], limit: 5, into: b)
        XCTAssertEqual(nB, 2)
        XCTAssertEqual(recurrentSum(b), 2 * 32)
    }

    func testCheckpointRestoreThenSaveInheritsTheCheckpointsBeforeTheEdit() async {
        let pc = PromptCache()
        await pc.save(tokens: [1, 2, 3, 4, 5, 6], cache: makeHybridCache(seqLen: 6, fill: 6),
                      allowRecurrent: true,
                      checkpoints: [checkpoint(at: 2, fill: 2), checkpoint(at: 4, fill: 4)])
        let live: [any KVCache] = [KVCacheSimple(), MambaCache()]
        let resumed = await pc.restoreExactPrefix(newTokens: [1, 2, 3, 4, 8, 8], limit: 6, into: live)
        XCTAssertEqual(resumed, 4)
        // Single-entry cache: the edited prompt replaces the original.
        await pc.save(tokens: [1, 2, 3, 4, 8, 8], cache: makeHybridCache(seqLen: 6, fill: 8),
                      allowRecurrent: true, checkpoints: [], resumedFrom: 4)
        let fresh: [any KVCache] = [KVCacheSimple(), MambaCache()]
        let n = await pc.restoreExactPrefix(newTokens: [1, 2, 7, 7], limit: 4, into: fresh)
        XCTAssertEqual(n, 2, "the checkpoint at 2 survived the replacement")
    }
}
