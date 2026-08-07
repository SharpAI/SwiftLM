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

    /// The exact config shape from the runtime validation on #114:
    /// `lmstudio-community/Qwen3.6-35B-A3B-MLX-8bit` — a multimodal MoE whose
    /// top level carries no expert keys at all, only `vision_config` and a
    /// `text_config` holding `num_experts`. Reported as `qwen3_5_moe is not MoE`
    /// before the fix, with the model profiled at a few GB instead of 37.7 GB.
    func testQwen36VisionMoENestedConfig() throws {
        let p = try profile(config: """
        {
          "model_type": "qwen3_5_moe",
          "architectures": ["Qwen3_5_MoeForConditionalGeneration"],
          "image_token_id": 151655,
          "vision_config": { "depth": 27, "hidden_size": 1152 },
          "text_config": {
            "model_type": "qwen3_5_moe_text",
            "num_hidden_layers": 40,
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "vocab_size": 248320,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512
          }
        }
        """, modelId: "lmstudio-community/Qwen3.6-35B-A3B-MLX-8bit")

        XCTAssertTrue(p.isMoE)
        // The counts must come from text_config. Without the nested lookup these are
        // nil even though the model_type heuristic would still flag the model as MoE,
        // so this is what distinguishes a real fix from an accidental pass.
        XCTAssertEqual(p.numExperts, 256)
        XCTAssertEqual(p.numActiveExperts, 8)
        XCTAssertEqual(p.numLayers, 40, "layer count must come from text_config")
        XCTAssertEqual(p.numKVHeads, 2)
        XCTAssertEqual(p.headDim, 256)
    }

    /// DeepSeek-VL2 nests the language model under `language_config`, and its
    /// model_type ("deepseek_vl_v2") contains no "moe" — so before the generic search
    /// nothing detected it at all. Shape verified against deepseek-ai/deepseek-vl2-tiny.
    func testLanguageConfigNesting() throws {
        let p = try profile(config: """
        {
          "model_type": "deepseek_vl_v2",
          "vision_config": { "width": 1024 },
          "language_config": {
            "num_hidden_layers": 12,
            "hidden_size": 1280,
            "n_routed_experts": 64,
            "num_experts_per_tok": 6
          }
        }
        """)

        XCTAssertTrue(p.isMoE, "language_config.n_routed_experts must be found")
        XCTAssertEqual(p.numExperts, 64)
        XCTAssertEqual(p.numActiveExperts, 6)
    }

    /// Qwen3-Omni nests two levels down, and carries a *different* active count under
    /// talker_config than under thinker_config — so the walk must be deterministic and
    /// must prefer the thinker. Shape verified against Qwen/Qwen3-Omni-30B-A3B-Instruct.
    func testTwoLevelNestingPrefersThinker() throws {
        let config = """
        {
          "model_type": "qwen3_omni_moe",
          "talker_config": {
            "text_config": { "num_experts": 128, "num_experts_per_tok": 6 }
          },
          "thinker_config": {
            "text_config": {
              "num_hidden_layers": 48,
              "hidden_size": 2048,
              "num_experts": 128,
              "num_experts_per_tok": 8
            }
          }
        }
        """
        // One run is enough: Swift seeds dictionary hashing per process, so repeating in
        // this process would reproduce the same order every time. The guarantee comes
        // from the explicit sibling ordering in findExpertCounts, not from repetition.
        let p = try profile(config: config)
        XCTAssertTrue(p.isMoE)
        XCTAssertEqual(p.numExperts, 128)
        XCTAssertEqual(p.numActiveExperts, 8, "must come from thinker_config, not talker_config")
    }

    /// An outer count still wins over anything nested, at any depth.
    func testOuterCountBeatsDeeplyNestedOne() throws {
        let p = try profile(config: """
        {
          \(baseKeys(modelType: "custom_arch")),
          "num_local_experts": 8,
          "num_experts_per_tok": 2,
          "thinker_config": { "text_config": { "num_experts": 128, "num_experts_per_tok": 8 } }
        }
        """)

        XCTAssertEqual(p.numExperts, 8)
        XCTAssertEqual(p.numActiveExperts, 2, "active count pairs with the container the total came from")
    }

    /// A count inside an encoder must never outrank the language model's, even though
    /// the encoder sits at a shallower depth (review finding on #125).
    func testNonLanguageContainersAreSkipped() throws {
        let p = try profile(config: """
        {
          "model_type": "some_omni",
          "audio_config": { "num_experts": 4, "num_experts_per_tok": 1 },
          "vision_config": { "num_experts": 2, "num_experts_per_tok": 1 },
          "thinker_config": {
            "text_config": { "num_hidden_layers": 40, "num_experts": 128, "num_experts_per_tok": 8 }
          }
        }
        """)

        XCTAssertEqual(p.numExperts, 128, "the encoders' counts must be ignored entirely")
        XCTAssertEqual(p.numActiveExperts, 8)
    }

    /// A container that declares a total but no per-token count inherits the nearest
    /// ancestor's, which is how wrappers that hoist num_experts_per_tok are written.
    func testActiveCountInheritedFromAncestor() throws {
        let p = try profile(config: """
        {
          "model_type": "wrapper_moe",
          "num_experts_per_tok": 8,
          "text_config": { "num_hidden_layers": 40, "num_experts": 128 }
        }
        """)

        XCTAssertEqual(p.numExperts, 128)
        XCTAssertEqual(p.numActiveExperts, 8, "inherited from the top level")
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

    /// An explicit expert count wins over the model_type heuristic. A config that says
    /// it has one expert is dense even when its name contains "moe" — otherwise a
    /// dense-converted or ablation checkpoint would enable SSD streaming with no experts
    /// to stream (review finding on #114).
    func testExplicitSingleExpertBeatsModelTypeHeuristic() throws {
        let p = try profile(config: """
        {
          \(baseKeys(modelType: "qwen3_5_moe")),
          "num_experts": 1
        }
        """)

        XCTAssertFalse(p.isMoE, "an explicit count of 1 is authoritative over the name heuristic")
        XCTAssertEqual(p.numExperts, 1)
    }

    /// A degenerate top-level value must not mask a real count nested under text_config.
    /// Multimodal wrappers can carry a placeholder at the top level (review finding on #114).
    func testDegenerateTopLevelDoesNotShadowNestedCount() throws {
        let p = try profile(config: """
        {
          "model_type": "some_wrapper",
          "num_experts": 0,
          "text_config": {
            "num_hidden_layers": 40,
            "hidden_size": 2048,
            "num_experts": 128,
            "num_experts_per_tok": 8
          }
        }
        """)

        XCTAssertTrue(p.isMoE, "the nested count is the real one")
        XCTAssertEqual(p.numExperts, 128, "a top-level 0 must not shadow text_config")
        XCTAssertEqual(p.numActiveExperts, 8)
    }

    /// A lone explicit 0 is a placeholder, not a declaration: the heuristic must still
    /// apply, or a wrapper whose real count uses an unknown spelling would profile as
    /// dense and re-open the #112 OOM path (review finding on #114).
    func testLoneZeroDoesNotSuppressHeuristic() throws {
        let p = try profile(config: """
        {
          \(baseKeys(modelType: "some_new_moe")),
          "num_experts": 0
        }
        """)

        XCTAssertTrue(p.isMoE, "a 0 placeholder must not veto the model_type heuristic")
        XCTAssertNil(p.numExperts)
    }

    /// An explicit outer count of 1 stays authoritative even when a stale nested count
    /// survives from the base model of a dense conversion (review finding on #114).
    func testExplicitOneBeatsStaleNestedCount() throws {
        let p = try profile(config: """
        {
          \(baseKeys(modelType: "custom_arch")),
          "num_local_experts": 1,
          "text_config": { "num_experts": 128 }
        }
        """)

        XCTAssertFalse(p.isMoE, "outer explicit 1 wins over a stale nested 128")
        XCTAssertEqual(p.numExperts, 1)
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
