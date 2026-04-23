// Copyright 2026 SwiftLM Contributors
// MIT License — see LICENSE file
// Based on DFlash (arXiv:2602.06036)

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - RecurrentRollbackCache

/// A cache for GatedDeltaNet (recurrent) layers that supports
/// speculative decoding rollback via innovation tape replay.
///
/// Subclasses MambaCache so that `cache as? MambaCache` succeeds in
/// Qwen35GatedDeltaNet.callAsFunction — this is critical for the normal
/// (non-armed) forward pass during prefill to work correctly.
///
/// During the verify phase, the cache is "armed" which causes the
/// GatedDeltaNet forward pass to record an innovation tape. If draft
/// tokens are rejected, the cache is rolled back by replaying only
/// the accepted steps from the tape.
public final class RecurrentRollbackCache: MambaCache, DFlashRollbackCache, @unchecked Sendable {

    /// Whether the cache is currently armed for tape recording.
    private var armed = false

    /// The recorded innovation tape: delta values per step.
    private var tape: MLXArray?
    /// The recorded keys for tape replay.
    private var tapeK: MLXArray?
    /// The recorded gates for tape replay.
    private var tapeG: MLXArray?
    /// The recorded QKV for conv state reconstruction.
    private var tapeQKV: MLXArray?

    /// Snapshot of the cache state before the verify pass.
    private var snapshotState: [MLXArray?]?

    public init(convKernelSize: Int = 4) {
        super.init()
    }

    // MARK: - Arming & Recording

    /// Arm the cache for tape recording and snapshot the current state.
    ///
    /// Uses lazy reference capture (no MLX.contiguous copy) — MLXArray is
    /// reference-counted so the old arrays remain alive after the cache is
    /// updated during the forward pass. The copy only happens if/when
    /// rollback() actually replays the tape.
    public func armRollback(prefixLen: Int = 0) {
        armed = true
        tape = nil
        tapeK = nil
        tapeG = nil
        tapeQKV = nil
        // Lazy snapshot: just hold references, no GPU copy needed
        snapshotState = [self[0], self[1]]
    }

    /// Record the innovation tape from a GatedDeltaNet forward step.
    /// Arrays are stored by reference — MLX evaluates them lazily when needed.
    public func recordTape(
        tape: MLXArray,
        k: MLXArray,
        g: MLXArray,
        qkv: MLXArray
    ) {
        self.tape = tape
        self.tapeK = k
        self.tapeG = g
        self.tapeQKV = qkv
    }

    /// Whether the cache is currently armed.
    public var isArmed: Bool { armed }

    // MARK: - Rollback

    /// Roll back the cache to the state after `nAccepted` tokens.
    /// Uses tape replay for the recurrent state (slot 1) and
    /// conv state reconstruction for slot 0.
    public func rollback(nAccepted: Int) {
        guard let snapshot = snapshotState else {
            clearTransients()
            return
        }

        // Calculate the offset to restore to
        // offset was incremented by the verify forward pass (by verifyLen tokens)
        // We need to set it to what it should be after accepting nAccepted+1 tokens
        // The Python reference doesn't explicitly manage offset in rollback,
        // but the cache offset needs to be consistent for subsequent forward passes.

        // Restore snapshot
        if snapshot.count > 0, let s0 = snapshot[0] { self[0] = s0 }
        if snapshot.count > 1, let s1 = snapshot[1] { self[1] = s1 }

        // Replay accepted steps through tape
        if let tape = tape, let tapeK = tapeK, let tapeG = tapeG,
           let state = self[1]
        {
            let acceptedSteps = nAccepted + 1
            let stateSlice = tape[0..., ..<acceptedSteps, 0..., 0...]
            let kSlice = tapeK[0..., ..<acceptedSteps, 0..., 0...]
            let gSlice = tapeG[0..., ..<acceptedSteps, 0...]

            let replayedState = DFlashKernels.tapeReplayKernel(
                tape: stateSlice,
                k: kSlice,
                g: gSlice,
                state: state
            )
            self[1] = replayedState
            self[0] = rebuildConvState(acceptedSteps: acceptedSteps)
        }

        clearTransients()
    }

    /// Get the conv kernel size from the model's GDN layer.
    /// We store it as a class property for conv state rebuilding.
    public static var defaultConvKernelSize: Int = 4

    /// Rebuild the conv state after rollback by taking the last
    /// `convKernelSize - 1` entries from the concatenated snapshot + tape QKV.
    private func rebuildConvState(acceptedSteps: Int) -> MLXArray? {
        guard let tapeQKV = tapeQKV else { return self[0] }
        let keep = RecurrentRollbackCache.defaultConvKernelSize - 1
        guard keep > 0 else { return nil }

        let prefix: MLXArray
        if let snap = snapshotState, snap.count > 0, let convState = snap[0] {
            prefix = convState
        } else {
            prefix = MLXArray.zeros(
                [tapeQKV.dim(0), keep, tapeQKV.dim(-1)],
                dtype: tapeQKV.dtype
            )
        }

        let convInput = concatenated([prefix, tapeQKV], axis: 1)
        let start = acceptedSteps
        let end = min(start + keep, convInput.dim(1))
        return convInput[0..., start ..< end, 0...]
    }

    // MARK: - Cleanup

    /// Clear all transient state (tape, snapshot, armed flag).
    public func clearTransients() {
        armed = false
        tape = nil
        tapeK = nil
        tapeG = nil
        tapeQKV = nil
        snapshotState = nil
    }

    // MARK: - Override MambaCache trim to use tape rollback instead

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        // For recurrent caches with tape, rollback handles trimming
        // Don't use the MambaCache checkpoint/trim path
        let trimmed = min(offset, n)
        offset -= trimmed
        return trimmed
    }
}

// MARK: - MambaSnapshotCache

/// Lightweight snapshot-based rollback for hybrid SSM models (e.g. Qwen3Next).
///
/// Unlike RecurrentRollbackCache, this does NOT record an innovation tape.
/// On partial acceptance, it restores the pre-verify SSM state snapshot.
/// The accepted tokens' state contributions are lost (state reverts to
/// pre-verify position), but rejected tokens' contamination is prevented.
/// Overhead: O(1) per cycle (lazy reference capture, no GPU copies).
public final class MambaSnapshotCache: MambaCache, DFlashRollbackCache, @unchecked Sendable {

    private var snapshotConv: MLXArray?
    private var snapshotRecurrent: MLXArray?
    private var armed = false

    public var isArmed: Bool { armed }

    public func armRollback(prefixLen: Int = 0) {
        armed = true
        // Lazy reference capture — no GPU copy, O(1)
        snapshotConv = self[0]
        snapshotRecurrent = self[1]
    }

    public func rollback(nAccepted: Int) {
        // Restore pre-verify state. Accepted tokens' contributions are
        // not replayed, but rejected tokens are excluded.
        self[0] = snapshotConv
        self[1] = snapshotRecurrent
        clearTransients()
    }

    public func clearTransients() {
        armed = false
        snapshotConv = nil
        snapshotRecurrent = nil
    }

    public func recordTape(tape: MLXArray, k: MLXArray, g: MLXArray, qkv: MLXArray) {
        // No tape needed for snapshot-based rollback
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        return trimmed
    }
}

