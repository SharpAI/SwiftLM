import XCTest
import Foundation
@testable import MLXInferenceCore

// MARK: - Regression tests for Issue #160 — "Specify Alternate Model Location"
//
// A user with models on an external drive (downloaded via `hf download --local-dir`,
// entirely outside the app's own cache) had no way to point SwiftBuddy at that folder
// directly. Unlike issue #110's hand-copied-into-the-cache layouts, these directories
// can be anywhere on disk — the functions under test here must work independent of
// `ModelStorage.cacheRoot` (no `cacheRootOverride` needed).
final class LocalModelDirectoryTests: XCTestCase {

    private var externalDir: URL!

    override func setUpWithError() throws {
        // Deliberately NOT under any cacheRoot — simulates an external drive folder.
        externalDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("swiftlm-external-model-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: externalDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: externalDir)
    }

    private func writeConfig(_ json: String) throws {
        try json.write(
            to: externalDir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
    }

    private func writeWeights(bytes: Int = 4096) throws {
        try Data(repeating: 0x7, count: bytes)
            .write(to: externalDir.appendingPathComponent("model.safetensors"))
    }

    // MARK: - isLocalDirectoryPath

    func testIsLocalDirectoryPathTrueForExistingDirectory() {
        XCTAssertTrue(ModelStorage.isLocalDirectoryPath(externalDir.path))
    }

    func testIsLocalDirectoryPathFalseForHFStyleId() {
        // A real HF repo id never happens to exist as a literal filesystem path from cwd.
        XCTAssertFalse(ModelStorage.isLocalDirectoryPath("mlx-community/Qwen3-8B-4bit"))
    }

    func testIsLocalDirectoryPathFalseForAFile() throws {
        try writeConfig(#"{"model_type":"qwen3"}"#)
        let filePath = externalDir.appendingPathComponent("config.json").path
        XCTAssertFalse(
            ModelStorage.isLocalDirectoryPath(filePath),
            "a file, not a directory, must not be treated as a local model directory")
    }

    func testIsLocalDirectoryPathFalseForNonexistentPath() {
        let missing = externalDir.appendingPathComponent("does-not-exist").path
        XCTAssertFalse(ModelStorage.isLocalDirectoryPath(missing))
    }

    // MARK: - validateLocalModelDirectory

    func testValidateLocalModelDirectoryAcceptsValidModel() throws {
        try writeConfig(#"{"model_type":"qwen3","num_hidden_layers":28}"#)
        try writeWeights()
        XCTAssertTrue(ModelStorage.validateLocalModelDirectory(externalDir))
    }

    func testValidateLocalModelDirectoryRejectsMissingConfig() throws {
        try writeWeights()
        XCTAssertFalse(
            ModelStorage.validateLocalModelDirectory(externalDir),
            "a folder with weights but no config.json is not a usable model")
    }

    func testValidateLocalModelDirectoryRejectsMissingWeights() throws {
        try writeConfig(#"{"model_type":"qwen3"}"#)
        XCTAssertFalse(
            ModelStorage.validateLocalModelDirectory(externalDir),
            "a folder with only config.json and no weights is not a usable model")
    }

    func testValidateLocalModelDirectoryRejectsEmptyFolder() {
        XCTAssertFalse(ModelStorage.validateLocalModelDirectory(externalDir))
    }

    // MARK: - readModelConfig(inDirectory:) / readMaxContextLength(inDirectory:)

    func testReadModelConfigInDirectory() throws {
        try writeConfig(#"{"model_type":"qwen3","max_position_embeddings":32768}"#)
        let config = ModelStorage.readModelConfig(inDirectory: externalDir)
        XCTAssertEqual(config?["model_type"] as? String, "qwen3")
    }

    func testReadModelConfigInDirectoryNilWhenMissing() {
        XCTAssertNil(ModelStorage.readModelConfig(inDirectory: externalDir))
    }

    func testReadMaxContextLengthInDirectoryTopLevel() throws {
        try writeConfig(#"{"model_type":"qwen3","max_position_embeddings":32768}"#)
        XCTAssertEqual(ModelStorage.readMaxContextLength(inDirectory: externalDir), 32768)
    }

    func testReadMaxContextLengthInDirectoryNestedTextConfig() throws {
        // VLM/MoE-style configs nest it under text_config — same convention
        // readMaxContextLength(for:) already handles for HF-cache models.
        try writeConfig(#"{"model_type":"qwen3_vl","text_config":{"max_position_embeddings":131072}}"#)
        XCTAssertEqual(ModelStorage.readMaxContextLength(inDirectory: externalDir), 131_072)
    }

    func testReadMaxContextLengthInDirectoryNilWhenAbsent() throws {
        try writeConfig(#"{"model_type":"qwen3"}"#)
        XCTAssertNil(ModelStorage.readMaxContextLength(inDirectory: externalDir))
    }

    // MARK: - configIndicatesMoE
    //
    // A local directory never matches a ModelCatalog entry (the catalog only
    // lists HuggingFace ids), so InferenceEngine falls back to this to decide
    // whether to default SSD expert streaming on — without it, a local MoE
    // model would silently load with streaming disabled.

    func testConfigIndicatesMoETopLevelNumExperts() {
        let config: [String: Any] = ["model_type": "qwen3_moe", "num_experts": 128]
        XCTAssertTrue(ModelStorage.configIndicatesMoE(config))
    }

    func testConfigIndicatesMoENRoutedExperts() {
        let config: [String: Any] = ["model_type": "deepseek_v3", "n_routed_experts": 256]
        XCTAssertTrue(ModelStorage.configIndicatesMoE(config))
    }

    func testConfigIndicatesMoENestedInTextConfig() {
        // The common one-level VLM/multimodal wrapper shape.
        let config: [String: Any] = [
            "model_type": "qwen3_vl_moe",
            "text_config": ["num_local_experts": 64],
        ]
        XCTAssertTrue(ModelStorage.configIndicatesMoE(config))
    }

    func testConfigIndicatesMoEFalseForDenseModel() {
        let config: [String: Any] = ["model_type": "qwen3", "num_hidden_layers": 28]
        XCTAssertFalse(ModelStorage.configIndicatesMoE(config))
    }

    func testConfigIndicatesMoEFalseForZeroExpertCount() {
        // A placeholder/zero value must not be treated as "this is MoE" —
        // matches ModelProfiler.findExpertCounts' "positive values only" rule.
        let config: [String: Any] = ["model_type": "qwen3", "num_experts": 0]
        XCTAssertFalse(ModelStorage.configIndicatesMoE(config))
    }
}
