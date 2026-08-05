import Foundation
import XCTest
@testable import SwiftLM

final class ModelProfilerTests: XCTestCase {
    func testDetectsNestedQwenMoEConfiguration() throws {
        let profile = try profile(json: """
            {
              "model_type": "qwen3_5_moe",
              "text_config": {
                "num_experts": 256,
                "num_experts_per_tok": 8
              }
            }
            """)

        XCTAssertTrue(profile.isMoE)
        XCTAssertEqual(profile.numExperts, 256)
        XCTAssertEqual(profile.numActiveExperts, 8)
    }

    func testKeepsDetectingTopLevelLocalExperts() throws {
        let profile = try profile(json: """
            {
              "model_type": "legacy_moe",
              "num_local_experts": 64,
              "num_experts_per_tok": 4
            }
            """)

        XCTAssertTrue(profile.isMoE)
        XCTAssertEqual(profile.numExperts, 64)
        XCTAssertEqual(profile.numActiveExperts, 4)
    }

    private func profile(json: String) throws -> ModelProfile {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        try Data(json.utf8).write(to: directory.appendingPathComponent("config.json"))
        return try XCTUnwrap(ModelProfiler.profile(modelDirectory: directory, modelId: "test-model"))
    }
}
