import XCTest
import Foundation
@testable import MLXInferenceCore

// MARK: - Regression tests for Issue #110 — hand-copied models are not detected
//
// The scan only accepted `models--org--name/snapshots/<hash>/` and the Swift Hub
// `models/org/name/` layout. A user who copies a model folder into the cache root —
// with no `models--` prefix, or with the weights sitting directly inside the
// `models--org--name` folder and no `snapshots/` level — got a model the app silently
// refused to list, with no indication why.
//
// These tests build real directory trees under a temporary cache root (via
// ModelStorage.cacheRootOverride) and lock in every layout the scan accepts, plus the
// rejection diagnostics.
final class ModelStorageLayoutTests: XCTestCase {

    private var root: URL!

    override func setUpWithError() throws {
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("swiftlm-storage-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        ModelStorage.cacheRootOverride = root
    }

    override func tearDownWithError() throws {
        ModelStorage.cacheRootOverride = nil
        try? FileManager.default.removeItem(at: root)
    }

    // MARK: Fixtures

    /// Writes a minimally valid single-file model (config.json + a >1KB safetensors).
    @discardableResult
    private func makeModel(at relativePath: String, weightBytes: Int = 4096) throws -> URL {
        let dir = root.appendingPathComponent(relativePath, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try #"{"model_type":"qwen3","num_hidden_layers":28}"#
            .write(to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try Data(repeating: 0x7, count: weightBytes)
            .write(to: dir.appendingPathComponent("model.safetensors"))
        return dir
    }

    private func scannedIds() -> [String] {
        ModelStorage.scanDownloadedModels().map(\.modelId).sorted()
    }

    // MARK: - 1. Layouts that already worked (must not regress)

    func testHubCacheLayoutWithSnapshot() throws {
        try makeModel(at: "models--mlx-community--Qwen3-8B-4bit/snapshots/abc123")
        try "abc123".write(
            to: root.appendingPathComponent("models--mlx-community--Qwen3-8B-4bit/refs/main")
                .creatingParent(),
            atomically: true, encoding: .utf8)

        XCTAssertEqual(scannedIds(), ["mlx-community/Qwen3-8B-4bit"])
        XCTAssertTrue(ModelStorage.isDownloaded("mlx-community/Qwen3-8B-4bit"))
    }

    func testMaterializedModelsLayout() throws {
        try makeModel(at: "models/mlx-community/Qwen3-4B-4bit")
        XCTAssertEqual(scannedIds(), ["mlx-community/Qwen3-4B-4bit"])
        XCTAssertTrue(ModelStorage.isDownloaded("mlx-community/Qwen3-4B-4bit"))
    }

    // MARK: - 2. Hand-copied layouts (issue #110)

    /// The `models--org--name` folder copied by hand, weights directly inside and no
    /// `snapshots/` level at all.
    func testHubPrefixedFolderWithoutSnapshots() throws {
        try makeModel(at: "models--mlx-community--Qwen3.5-27B-oQ6")

        XCTAssertEqual(scannedIds(), ["mlx-community/Qwen3.5-27B-oQ6"])
        XCTAssertTrue(ModelStorage.isDownloaded("mlx-community/Qwen3.5-27B-oQ6"))
        XCTAssertNotNil(ModelStorage.readModelConfig(for: "mlx-community/Qwen3.5-27B-oQ6"))
    }

    /// A bare model folder dropped into the cache root, no org level.
    func testPlainFolderWithoutOrganization() throws {
        try makeModel(at: "Qwen3.5-27B-oQ6")

        XCTAssertEqual(scannedIds(), ["Qwen3.5-27B-oQ6"])
        XCTAssertTrue(ModelStorage.isDownloaded("Qwen3.5-27B-oQ6"))
    }

    /// An `org/name` pair copied in without the `models--` mangling.
    func testPlainOrganizationAndNameFolder() throws {
        try makeModel(at: "mlx-community/Qwen3.5-27B-oQ6")

        XCTAssertEqual(scannedIds(), ["mlx-community/Qwen3.5-27B-oQ6"])
        XCTAssertTrue(ModelStorage.isDownloaded("mlx-community/Qwen3.5-27B-oQ6"))
    }

    func testShardedHandCopiedModelWithIndex() throws {
        let dir = root.appendingPathComponent("mlx-community/Qwen3-32B-4bit", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try #"{"model_type":"qwen3"}"#
            .write(to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        for shard in ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"] {
            try Data(repeating: 0x7, count: 4096).write(to: dir.appendingPathComponent(shard))
        }
        let index = """
        {"metadata":{"total_size":8192},"weight_map":{\
        "a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}
        """
        try index.write(to: dir.appendingPathComponent("model.safetensors.index.json"),
                        atomically: true, encoding: .utf8)

        XCTAssertEqual(scannedIds(), ["mlx-community/Qwen3-32B-4bit"])
    }

    // MARK: - 3. Things that must still be rejected

    func testIncompleteDownloadIsNotListed() throws {
        let dir = try makeModel(at: "mlx-community/Qwen3-8B-4bit")
        try Data(repeating: 0, count: 16)
            .write(to: dir.appendingPathComponent("model.safetensors.incomplete"))

        XCTAssertEqual(scannedIds(), [])
        XCTAssertFalse(ModelStorage.isDownloaded("mlx-community/Qwen3-8B-4bit"))
    }

    func testFolderWithoutWeightsIsNotListed() throws {
        let dir = root.appendingPathComponent("mlx-community/NoWeights", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try #"{"model_type":"qwen3"}"#
            .write(to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)

        XCTAssertEqual(scannedIds(), [])
    }

    func testUnrelatedDirectoriesAreIgnored() throws {
        try FileManager.default.createDirectory(
            at: root.appendingPathComponent(".locks/models--x"), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: root.appendingPathComponent("random-notes"), withIntermediateDirectories: true)
        try "not a model".write(to: root.appendingPathComponent("random-notes/todo.txt"),
                                atomically: true, encoding: .utf8)

        XCTAssertEqual(scannedIds(), [])
        XCTAssertTrue(ModelStorage.diagnoseUnrecognizedDirectories().isEmpty,
                      "directories with no config.json are not model candidates")
    }

    // MARK: - 4. Diagnostics

    func testDiagnosticsExplainMissingWeights() throws {
        let dir = root.appendingPathComponent("mlx-community/BrokenModel", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try #"{"model_type":"qwen3"}"#
            .write(to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)

        let findings = ModelStorage.diagnoseUnrecognizedDirectories()
        XCTAssertEqual(findings.count, 1)
        XCTAssertEqual(findings.first?.directory.lastPathComponent, "BrokenModel")
        XCTAssertTrue(findings.first?.reason.contains("no .safetensors weights") == true,
                      "got: \(findings.first?.reason ?? "nil")")
    }

    func testDiagnosticsExplainMissingIndex() throws {
        let dir = root.appendingPathComponent("ShardedNoIndex", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try #"{"model_type":"qwen3"}"#
            .write(to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try Data(repeating: 0x7, count: 4096)
            .write(to: dir.appendingPathComponent("model-00001-of-00002.safetensors"))

        let findings = ModelStorage.diagnoseUnrecognizedDirectories()
        XCTAssertEqual(findings.count, 1)
        XCTAssertTrue(findings.first?.reason.contains("index.json is missing") == true,
                      "got: \(findings.first?.reason ?? "nil")")
    }

    func testDiagnosticsExplainIncompleteFiles() throws {
        let dir = try makeModel(at: "mlx-community/Partial")
        try Data(repeating: 0, count: 16)
            .write(to: dir.appendingPathComponent("model.safetensors.incomplete"))

        let findings = ModelStorage.diagnoseUnrecognizedDirectories()
        XCTAssertEqual(findings.count, 1)
        XCTAssertTrue(findings.first?.reason.contains(".incomplete") == true,
                      "got: \(findings.first?.reason ?? "nil")")
    }

    /// A model the scan *does* accept must never be reported as unrecognized.
    func testDiagnosticsAreSilentForValidModels() throws {
        try makeModel(at: "mlx-community/Qwen3-8B-4bit")
        try makeModel(at: "PlainCopy")

        XCTAssertEqual(scannedIds(), ["PlainCopy", "mlx-community/Qwen3-8B-4bit"])
        let findings = ModelStorage.diagnoseUnrecognizedDirectories()
        XCTAssertTrue(findings.isEmpty,
                      "got: \(findings.map { "\($0.directory.path): \($0.reason)" })")
    }

    // MARK: - 5. snapshotDirectory resolution
    //
    // SSD expert streaming points its reader at snapshotDirectory(), so a path that
    // does not exist misconfigures streaming rather than failing loudly.

    func testSnapshotDirectoryResolvesForEveryLayout() throws {
        let cases: [(path: String, id: String)] = [
            ("models--mlx-community--WithSnapshot/snapshots/abc", "mlx-community/WithSnapshot"),
            ("models/mlx-community/Materialized", "mlx-community/Materialized"),
            ("models--mlx-community--NoSnapshots", "mlx-community/NoSnapshots"),
            ("mlx-community/PlainOrg", "mlx-community/PlainOrg"),
            ("PlainBare", "PlainBare"),
        ]
        for testCase in cases {
            try makeModel(at: testCase.path)
        }

        for testCase in cases {
            let resolved = ModelStorage.snapshotDirectory(for: testCase.id)
            XCTAssertTrue(
                FileManager.default.fileExists(atPath: resolved.appendingPathComponent("config.json").path),
                "\(testCase.id) resolved to \(resolved.path), which holds no config.json")
        }
    }

    /// With nothing on disk the legacy hub path is still returned, so callers that
    /// construct a download destination keep working.
    func testSnapshotDirectoryFallsBackToHubPathWhenAbsent() {
        let resolved = ModelStorage.snapshotDirectory(for: "mlx-community/NotThere")
        XCTAssertEqual(resolved.path,
                       root.appendingPathComponent("models--mlx-community--NotThere/snapshots/main").path)
    }

    // MARK: - 6. Deletion safety

    /// delete() calls removeItem on every directory associated with an id, so a bogus id
    /// must never resolve to the cache root or the shared `models/` wrapper. The earlier
    /// version of this test only checked plainDirectory's return value and never called
    /// delete(), so it passed even while `delete("models/")` wiped the whole tree.
    func testDeleteWithHostileIdsLeavesTheCacheIntact() throws {
        try makeModel(at: "models/mlx-community/Qwen3-4B-4bit")
        try makeModel(at: "models/mlx-community/Qwen3-8B-4bit")
        try makeModel(at: "HandCopied")
        let survivors = [
            root.appendingPathComponent("models/mlx-community/Qwen3-4B-4bit"),
            root.appendingPathComponent("models/mlx-community/Qwen3-8B-4bit"),
            root.appendingPathComponent("HandCopied"),
        ]

        // Each of these reduces to the cache root or the `models/` wrapper.
        for hostile in ["", "models", "models/", "/models", "models//", "./models", "..", "../..", "a/.."] {
            try? ModelStorage.delete(hostile)
            XCTAssertTrue(FileManager.default.fileExists(atPath: root.path),
                          "cache root removed by delete(\"\(hostile)\")")
            for survivor in survivors {
                XCTAssertTrue(FileManager.default.fileExists(atPath: survivor.path),
                              "delete(\"\(hostile)\") removed \(survivor.lastPathComponent)")
            }
        }
        XCTAssertEqual(scannedIds().count, 3, "no model should have been deleted")
    }

    func testIsSafeModelDirectoryRejectsRootAndWrapper() {
        XCTAssertFalse(ModelStorage.isSafeModelDirectory(root))
        XCTAssertFalse(ModelStorage.isSafeModelDirectory(root.appendingPathComponent("models")))
        XCTAssertFalse(ModelStorage.isSafeModelDirectory(root.appendingPathComponent("..")))
        XCTAssertFalse(ModelStorage.isSafeModelDirectory(root.deletingLastPathComponent()))
        XCTAssertTrue(ModelStorage.isSafeModelDirectory(root.appendingPathComponent("SomeModel")))
        XCTAssertTrue(ModelStorage.isSafeModelDirectory(root.appendingPathComponent("models/org/name")))
    }

    func testDeleteRemovesHandCopiedFolder() throws {
        let dir = try makeModel(at: "mlx-community/Qwen3.5-27B-oQ6")
        XCTAssertTrue(ModelStorage.isDownloaded("mlx-community/Qwen3.5-27B-oQ6"))

        try ModelStorage.delete("mlx-community/Qwen3.5-27B-oQ6")

        XCTAssertFalse(FileManager.default.fileExists(atPath: dir.path))
        XCTAssertEqual(scannedIds(), [])
    }
}

private extension URL {
    /// Creates the parent directory of this URL and returns the URL unchanged.
    func creatingParent() -> URL {
        try? FileManager.default.createDirectory(
            at: deletingLastPathComponent(), withIntermediateDirectories: true)
        return self
    }
}
