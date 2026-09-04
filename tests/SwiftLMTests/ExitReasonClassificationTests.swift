import XCTest
import NIOCore
@testable import SwiftLM

// MARK: - Coverage for classifyExitReason (Aegis Engine Protocol v1 self-reported
// exit reasons — daemon/spec/engine-protocol.md in the Aegis-AI repo).
//
// `run()`'s top-level catch is exercised manually (see docs/AEGIS_INTEGRATION.md
// and the PR description for `swift build -c release` + live-process transcripts
// against a nonexistent model path and an occupied port) rather than here, since
// it requires an actual model load / socket bind. This file pins the pure
// classification function those manual runs depend on, so the mapping itself has
// a repeatable, non-manual check.
final class ExitReasonClassificationTests: XCTestCase {

    private struct GenericLoadError: Error {}

    func testPortBindAddressInUseIsPortConflict() {
        let error = IOError(errnoCode: EADDRINUSE, reason: "bind() failed")
        XCTAssertEqual(classifyExitReason(error, phase: .portBind), "port_conflict")
    }

    func testPortBindOtherIOErrorFallsThroughToBinaryError() {
        // Only address-in-use is special-cased; any other bind-time failure
        // should not be mislabeled port_conflict.
        let error = IOError(errnoCode: EACCES, reason: "bind() failed")
        XCTAssertEqual(classifyExitReason(error, phase: .portBind), "binary_error")
    }

    func testMalformedChatTemplateIsModelLoadFailedRegardlessOfPhase() {
        let error = MalformedChatTemplate(modelId: "some/model", underlying: "bad jinja")
        // Even outside chatTemplateProbe, a MalformedChatTemplate error is always
        // a model-artifact defect.
        XCTAssertEqual(classifyExitReason(error, phase: .startup), "model_load_failed")
        XCTAssertEqual(classifyExitReason(error, phase: .chatTemplateProbe), "model_load_failed")
    }

    func testEachLoadPhaseMapsToModelLoadFailed() {
        let error = GenericLoadError()
        for phase: LoadPhase in [
            .architectureProbe, .mainModelLoad, .draftModelLoad, .mtpAssistantLoad,
            .chatTemplateProbe,
        ] {
            XCTAssertEqual(
                classifyExitReason(error, phase: phase), "model_load_failed",
                "phase \(phase) should classify as model_load_failed")
        }
    }

    func testOutOfMemoryIsIdentifiedFromMessageText() {
        struct AllocationFailure: Error, CustomStringConvertible {
            var description: String { "cannot allocate 40 GB: out of memory" }
        }
        XCTAssertEqual(
            classifyExitReason(AllocationFailure(), phase: .startup), "out_of_memory")
    }

    func testUnclassifiableErrorFallsBackToBinaryError() {
        XCTAssertEqual(classifyExitReason(GenericLoadError(), phase: .startup), "binary_error")
        XCTAssertEqual(classifyExitReason(GenericLoadError(), phase: .portBind), "binary_error")
    }
}
