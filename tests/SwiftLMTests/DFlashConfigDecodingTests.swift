import XCTest
import Foundation
@testable import DFlash

// MARK: - Regression tests for Issue #121 — rope_theta moved into rope_parameters
//
// z-lab/Qwen3.5-4B-DFlash ships no top-level rope_theta; the value lives under
// rope_parameters.rope_theta (transformers' rope-settings migration). The synthesized
// Codable init treated every listed key as required — property defaults never apply to
// synthesized decoding — so the whole config failed with keyNotFound("rope_theta") and
// the draft model never loaded. This kept the dflash-speculative-decoding CI job red.
final class DFlashConfigDecodingTests: XCTestCase {

    /// The verbatim config.json of z-lab/Qwen3.5-4B-DFlash (fetched 2026-08-06).
    private let realConfig = #"""
{
  "architectures": [
    "DFlashDraftModel"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "auto_map": {
    "AutoModel": "dflash.DFlashDraftModel"
  },
  "bos_token_id": null,
  "dflash_config": {
    "block_size": 16,
    "mask_token_id": 248077,
    "target_layer_ids": [
      1,
      5,
      9,
      13,
      17,
      21,
      25,
      29
    ]
  },
  "dtype": "bfloat16",
  "eos_token_id": 248044,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 2560,
  "initializer_range": 0.02,
  "intermediate_size": 9216,
  "layer_types": [
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention"
  ],
  "max_position_embeddings": 262144,
  "max_window_layers": 6,
  "model_type": "qwen3",
  "num_attention_heads": 32,
  "num_hidden_layers": 6,
  "num_key_value_heads": 8,
  "num_target_layers": 32,
  "pad_token_id": null,
  "rms_norm_eps": 1e-06,
  "rope_parameters": {
    "rope_theta": 10000000,
    "rope_type": "default"
  },
  "sliding_window": 4096,
  "tie_word_embeddings": true,
  "transformers_version": "5.7.0",
  "use_cache": true,
  "use_sliding_window": true,
  "vocab_size": 248320
}
"""#

    func testRealDFlashConfigDecodes() throws {
        let c = try JSONDecoder().decode(
            DFlashDraftConfiguration.self, from: Data(realConfig.utf8))
        XCTAssertEqual(c.ropeTheta, 10_000_000, "must come from rope_parameters.rope_theta")
        XCTAssertEqual(c.numHiddenLayers, 6)
        XCTAssertEqual(c.blockSize, 16)
        XCTAssertEqual(c.numTargetLayers, 32)
    }

    /// Older checkpoints with the flat key keep working.
    func testFlatRopeThetaStillWins() throws {
        let json = #"{"rope_theta": 5000000, "rope_parameters": {"rope_theta": 1}}"#
        let c = try JSONDecoder().decode(DFlashDraftConfiguration.self, from: Data(json.utf8))
        XCTAssertEqual(c.ropeTheta, 5_000_000, "flat key takes precedence when present")
    }

    /// With neither form present, the documented default applies — previously any
    /// missing listed key failed the decode outright.
    func testAbsentKeysFallBackToDefaults() throws {
        let c = try JSONDecoder().decode(DFlashDraftConfiguration.self, from: Data("{}".utf8))
        XCTAssertEqual(c.ropeTheta, 1_000_000)
        XCTAssertEqual(c.numHiddenLayers, 4)
    }
}
