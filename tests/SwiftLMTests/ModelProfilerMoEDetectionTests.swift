import XCTest
import Foundation
@testable import SwiftLM

// MARK: - Regression tests for Issue #112 — MoE detection across architecture families
//
// Root cause: ModelProfiler decoded the routed-expert count from `num_local_experts`
// only — the Mixtral spelling. Qwen3/Qwen3.5 MoE use `num_experts`, DeepSeek V2/V3 and
// GLM MoE use `n_routed_experts`. Those models profiled as dense, so Server.swift
// disabled --stream-experts ("qwen3_5_moe is not MoE"), never set lazyLoad, and the
// full model was materialised into RAM — the OS then OOM-killed the process.
//
// These tests lock in that every known expert-count spelling is recognised, that the
// nested text_config form works, and that dense models stay dense.
final class ModelProfilerMoEDetectionTests: XCTestCase {

    private var tempRoot: URL!

    override func setUpWithError() throws {
        tempRoot = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("swiftlm-moe-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempRoot, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tempRoot)
    }

    /// Writes a config.json into a fresh directory and profiles it.
    private func profile(config: String, modelId: String = "test/model") throws -> ModelProfile {
        let dir = tempRoot.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try config.write(to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let profile = ModelProfiler.profile(modelDirectory: dir, modelId: modelId)
        return try XCTUnwrap(profile, "profile() returned nil for config: \(config)")
    }

    private func baseKeys(modelType: String) -> String {
        """
        "model_type": "\(modelType)",
        "num_hidden_layers": 48,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "intermediate_size": 12288,
        "vocab_size": 151936
        """
    }

    // MARK: Expert-count spellings

    /// Qwen3.5-MoE — the model from the issue report. Uses `num_experts`.
    func testQwen35MoEDetectedViaNumExperts() throws {
        let p = try profile(config: """
        {
          \(baseKeys(modelType: "qwen3_5_moe")),
          "num_experts": 128,
          "num_experts_per_tok": 8
        }
        """, modelId: "/Users/user/models/Qwen3.5-397B-A17B-mxfp8-grp32")

        XCTAssertTrue(p.isMoE, "qwen3_5_moe with num_experts=128 must profile as MoE (issue #112)")
        XCTAssertEqual(p.numExperts, 128)
        XCTAssertEqual(p.numActiveExperts, 8)
    }

    /// Mixtral / Phi-MoE — `num_local_experts`. The only spelling that worked before.
    func testMixtralDetectedViaNumLocalExperts() throws {
        let p = try profile(config: """
        {
          \(baseKeys(modelType: "mixtral")),
          "num_local_experts": 8,
          "num_experts_per_tok": 2
        }
        """)

        XCTAssertTrue(p.isMoE)
        XCTAssertEqual(p.numExperts, 8)
        XCTAssertEqual(p.numActiveExperts, 2)
    }

    /// DeepSeek V3 / GLM4 MoE / MiniMax — `n_routed_experts`.
    func testDeepSeekDetectedViaNRoutedExperts() throws {
        let p = try profile(config: """
        {
          \(baseKeys(modelType: "deepseek_v3")),
          "n_routed_experts": 256,
          "num_experts_per_tok": 8
        }
        """)

        XCTAssertTrue(p.isMoE)
        XCTAssertEqual(p.numExperts, 256)
        XCTAssertEqual(p.numActiveExperts, 8)
    }

    /// Multimodal wrappers nest the language-model config under `text_config`.
    func testNestedTextConfigExpertCount() throws {
        let p = try profile(config: """
        {
          "model_type": "qwen3_vl_moe",
          "text_config": {
            "num_hidden_layers": 48,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "intermediate_size": 12288,
            "vocab_size": 151936,
            "num_experts": 128,
            "num_experts_per_tok": 8
          }
        }
        """)

        XCTAssertTrue(p.isMoE)
        XCTAssertEqual(p.numExperts, 128)
        XCTAssertEqual(p.numActiveExperts, 8)
        XCTAssertEqual(p.numLayers, 48, "layer count must still come from text_config")
    }

    // MARK: Fallback + negative cases

    /// A config using an expert-count key we have not seen still profiles as MoE
    /// when model_type says so — the OOM-safe direction.
    func testUnknownExpertKeyFallsBackToModelType() throws {
        let p = try profile(config: """
        {
          \(baseKeys(modelType: "some_new_moe")),
          "n_future_experts": 64
        }
        """)

        XCTAssertTrue(p.isMoE, "model_type containing 'moe' must be treated as MoE")
        XCTAssertNil(p.numExperts, "no known expert key present, so the count stays unknown")
    }

    func testModelTypeHeuristic() {
        for type in ["qwen3_5_moe", "glm4_moe", "GLM-MoE", "mixtral", "dbrx", "grok_1"] {
            XCTAssertTrue(ModelProfiler.modelTypeImpliesMoE(type), "\(type) should imply MoE")
        }
        for type in ["qwen3_5", "llama", "gemma3", "phi3", "mistral", "unknown"] {
            XCTAssertFalse(ModelProfiler.modelTypeImpliesMoE(type), "\(type) must not imply MoE")
        }
    }

    /// Dense models must stay dense — otherwise --stream-experts would be honoured
    /// on a model with no experts to stream.
    func testDenseModelIsNotMoE() throws {
        let p = try profile(config: """
        { \(baseKeys(modelType: "qwen3_5")) }
        """)

        XCTAssertFalse(p.isMoE)
        XCTAssertNil(p.numExperts)
    }

    /// A single "expert" is a dense FFN in disguise; the pre-fix guard required > 1
    /// and that threshold must survive.
    func testSingleExpertIsNotMoE() throws {
        let p = try profile(config: """
        {
          \(baseKeys(modelType: "custom_arch")),
          "num_experts": 1
        }
        """)

        XCTAssertFalse(p.isMoE, "num_experts=1 is a dense FFN, not a MoE")
    }

    func testMissingConfigReturnsNil() {
        let dir = tempRoot.appendingPathComponent("empty")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        XCTAssertNil(ModelProfiler.profile(modelDirectory: dir, modelId: "test/model"))
    }
}
