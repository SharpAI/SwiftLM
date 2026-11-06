import XCTest
import Foundation
import MLX
@testable import DFlash
import MLXLMCommon

// MARK: - Rolling a hybrid target back after a partially accepted DFlash block
//
// Qwen3.5's gated delta net layers keep a recurrent state in a plain `MambaCache`. The
// verify pass runs the whole drafted block through it, and when only part of the block is
// accepted that state cannot be trimmed like attention KV. The runtime used to leave it as
// it was (contributions of the rejected drafts included), and generation degenerated
// after the first block. Now `armTargetRollback` checkpoints such caches, and a partial
// acceptance restores them and replays the committed tokens through the target, as
// mlx-swift-lm's draft-model loop does on a rejection with recurrent layers.
final class DFlashRecurrentRollbackTests: XCTestCase {

    private func recurrent(_ value: Float) -> MambaCache {
        let cache = MambaCache()
        cache[0] = MLXArray([value, value])
        cache[1] = MLXArray([value])
        cache.offset = 10
        return cache
    }

    private func attention(tokens: Int) -> KVCacheSimple {
        let cache = KVCacheSimple()
        let keys = MLXArray.zeros([1, 1, tokens, 2])
        _ = cache.update(keys: keys, values: keys)
        return cache
    }

    func testNeedsReplayOnlyForPlainRecurrentCaches() {
        XCTAssertFalse(DFlashRuntime.needsReplayAfterPartialAcceptance([attention(tokens: 4)]))
        XCTAssertTrue(DFlashRuntime.needsReplayAfterPartialAcceptance([attention(tokens: 4), recurrent(1)]))
        XCTAssertFalse(
            DFlashRuntime.needsReplayAfterPartialAcceptance([RecurrentRollbackCache(), MambaSnapshotCache()]),
            "rollback-capable caches roll back in place")
    }

    func testArmCheckpointsPlainRecurrentCaches() {
        let cache = recurrent(1)
        DFlashRuntime.armTargetRollback(targetCache: [cache], prefixLen: 10)
        // The verify pass moves the state on; a trim restores the checkpoint.
        cache[1] = MLXArray([99] as [Float])
        cache.offset = 26
        XCTAssertEqual(cache.trim(1), 1)
        XCTAssertEqual(cache[1]!.asArray(Float.self), [1])
        XCTAssertEqual(cache.offset, 10)
    }

    /// A partial acceptance: the recurrent state goes back to the checkpoint, attention is
    /// trimmed to the prefix, and the committed tokens are fed once more.
    func testPartialAcceptanceRestoresCheckpointTrimsAttentionAndReplays() {
        let mamba = recurrent(1)
        let attn = attention(tokens: 10)
        let caches: [KVCache] = [attn, mamba]
        DFlashRuntime.armTargetRollback(targetCache: caches, prefixLen: 10)
        // The verify pass of a 16-token block: 16 more attention positions, a moved state.
        _ = attn.update(keys: MLXArray.zeros([1, 1, 16, 2]), values: MLXArray.zeros([1, 1, 16, 2]))
        mamba[1] = MLXArray([99] as [Float])
        mamba.offset = 26

        var replayed: [Int] = []
        let committed = MLXArray([7, 8, 9].map { UInt32($0) })  // staged first + 2 accepted
        DFlashRuntime.replayCommittedTokens(caches, prefixLen: 10, committed: committed) { ids in
            XCTAssertEqual(ids.shape, [1, 3])
            replayed = ids.asArray(Int.self)
            // The forward ran against caches already at the prefix.
            XCTAssertEqual(attn.offset, 10)
            XCTAssertEqual(mamba.offset, 10)
            return MLXArray.zeros([1, 3, 4])
        }
        XCTAssertEqual(replayed, [7, 8, 9])
        XCTAssertEqual(mamba[1]!.asArray(Float.self), [1], "the rejected drafts' contributions are gone")
        XCTAssertEqual(attn.offset, 10, "the forward in this test appends nothing; the real target appends the committed tokens")
    }

    /// Nothing rejected: the state is right as it is, and the checkpoint is dropped so a
    /// later trim cannot restore stale state.
    func testFullAcceptanceKeepsStateAndDropsCheckpoint() {
        let mamba = recurrent(1)
        DFlashRuntime.armTargetRollback(targetCache: [mamba], prefixLen: 10)
        mamba[1] = MLXArray([5] as [Float])
        mamba.offset = 26
        DFlashRuntime.restoreTargetCacheAfterAcceptance(
            [mamba], targetLen: 26, acceptanceLength: 15, draftedTokens: 15)
        XCTAssertEqual(mamba[1]!.asArray(Float.self), [5])
        XCTAssertEqual(mamba.offset, 26)
        XCTAssertEqual(mamba.trim(1), 0, "no checkpoint is held any more")
        XCTAssertEqual(mamba[1]!.asArray(Float.self), [5])
    }
}
