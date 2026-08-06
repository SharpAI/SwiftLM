// ModelStorage.swift — Platform-aware model storage resolution
// macOS: ~/.cache/huggingface/hub/ (or $HF_HUB_CACHE / $HF_HOME/hub — same as huggingface-cli)
// iOS:   ~/Library/Application Support/SwiftBuddy/Models/ (persistent, excluded from iCloud)

import Foundation

public enum ModelStorage {

    // MARK: — Platform Paths

    /// Test hook: when set, overrides `cacheRoot` entirely. Deliberately not public —
    /// tests reach it through `@testable import`.
    nonisolated(unsafe) static var cacheRootOverride: URL?

    /// Root directory where model files are stored on this platform.
    /// This is the `downloadBase` passed to `HubApi`.
    public static var cacheRoot: URL {
        if let cacheRootOverride { return cacheRootOverride }
        #if os(macOS)
        // macOS: Single source of truth with Python (huggingface-cli / mlx_lm).
        // Precedence matches huggingface_hub: HF_HUB_CACHE points at the hub
        // directory itself, HF_HOME at its parent.
        let environment = ProcessInfo.processInfo.environment
        if let hubCache = environment["HF_HUB_CACHE"], !hubCache.isEmpty {
            return URL(fileURLWithPath: hubCache)
        }
        if let hfHome = environment["HF_HOME"], !hfHome.isEmpty {
            return URL(fileURLWithPath: hfHome).appendingPathComponent("hub")
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
        #else
        // iOS: Application Support — persistent, NOT purgeable, excluded from iCloud
        return applicationSupportModelsRoot
        #endif
    }

    /// iOS-specific persistent models directory.
    public static var applicationSupportModelsRoot: URL {
        let base = FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask)
            .first!
            .appendingPathComponent("SwiftBuddy/Models", isDirectory: true)
        ensureDirectory(base)
        return base
    }

    /// HuggingFace hub subdirectory name for a model ID.
    /// e.g. "mlx-community/Qwen2.5-7B-Instruct-4bit"
    ///   → "models--mlx-community--Qwen2.5-7B-Instruct-4bit"
    public static func hubDirName(for modelId: String) -> String {
        "models--" + modelId.replacingOccurrences(of: "/", with: "--")
    }

    /// Local cache directory for a model, or nil if not downloaded.
    public static func cacheDirectory(for modelId: String) -> URL? {
        materializedDirectory(for: modelId) ?? hubCacheDirectory(for: modelId) ?? plainDirectory(for: modelId)
    }

    /// A model folder copied straight into the cache root, with no `models--` prefix
    /// and no `snapshots/` level — e.g. `<cacheRoot>/Qwen3.5-27B-oQ6/` or
    /// `<cacheRoot>/mlx-community/Qwen3.5-27B-oQ6/` (issue #110).
    static func plainDirectory(for modelId: String) -> URL? {
        guard let dir = plainDirectoryURL(for: modelId) else { return nil }
        return directoryExists(dir) ? dir : nil
    }

    private static func plainDirectoryURL(for modelId: String) -> URL? {
        let url = modelId.split(separator: "/").reduce(cacheRoot) { url, component in
            url.appendingPathComponent(String(component), isDirectory: true)
        }
        return isSafeModelDirectory(url) ? url : nil
    }

    /// Whether a URL is a location a model may legitimately occupy, i.e. a strict
    /// descendant of the cache root that is neither the root itself nor the shared
    /// `models/` wrapper.
    ///
    /// `delete()` calls `FileManager.removeItem` on the directories associated with an
    /// id, so this is a destructive-action guard and must resolve the path rather than
    /// compare strings: an id of `"models/"`, `"/models"` or `"./models"` all reduce to
    /// `<cacheRoot>/models`, and `".."` components escape the root entirely.
    static func isSafeModelDirectory(_ url: URL) -> Bool {
        let root = cacheRoot.standardizedFileURL
        let candidate = url.standardizedFileURL
        let rootComponents = root.pathComponents
        let candidateComponents = candidate.pathComponents

        // Must be strictly below the cache root, by at least one component.
        guard candidateComponents.count > rootComponents.count,
            Array(candidateComponents.prefix(rootComponents.count)) == rootComponents
        else { return false }

        // Never the shared layout wrapper itself.
        let relative = candidateComponents.dropFirst(rootComponents.count)
        if relative.count == 1 && relative.first == "models" { return false }
        return true
    }

    /// Swift Hub's materialized repository directory.
    /// `HubApi(downloadBase: cacheRoot).snapshot(from:)` writes here.
    public static func materializedDirectory(for modelId: String) -> URL? {
        let dir = materializedDirectoryURL(for: modelId)
        return directoryExists(dir) ? dir : nil
    }

    private static func materializedDirectoryURL(for modelId: String) -> URL {
        cacheRoot
            .appendingPathComponent("models", isDirectory: true)
            .appendingPathComponent(modelId, isDirectory: true)
    }

    /// Hugging Face hub cache directory used by Python tools and older SwiftBuddy paths.
    public static func hubCacheDirectory(for modelId: String) -> URL? {
        let dir = hubCacheDirectoryURL(for: modelId)
        return directoryExists(dir) ? dir : nil
    }

    private static func hubCacheDirectoryURL(for modelId: String) -> URL {
        cacheRoot.appendingPathComponent(hubDirName(for: modelId), isDirectory: true)
    }

    private static func directoryExists(_ url: URL) -> Bool {
        var isDirectory: ObjCBool = false
        return FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) && isDirectory.boolValue
    }

    /// The directory holding a model's weight files.
    ///
    /// Resolves through every supported layout — see `scanDownloadedModels()` — and
    /// only falls back to the canonical hub path when none of them exist. Returning a
    /// path that does not exist would silently misconfigure consumers such as SSD
    /// expert streaming, which points its reader at this directory.
    public static func snapshotDirectory(for modelId: String) -> URL {
        modelContentDirectories(for: modelId).first ?? cacheRoot
            .appendingPathComponent(hubDirName(for: modelId))
            .appendingPathComponent("snapshots/main")
    }

    /// Resolve the active snapshot directory for a model in the Hugging Face hub cache.
    /// Prefer refs/main because snapshot directories are usually commit hashes, not "main".
    public static func resolvedSnapshotDirectory(for modelId: String) -> URL? {
        guard let dir = hubCacheDirectory(for: modelId) else { return nil }

        let snapshotsDir = dir.appendingPathComponent("snapshots", isDirectory: true)
        guard FileManager.default.fileExists(atPath: snapshotsDir.path) else { return nil }

        let refsMain = dir.appendingPathComponent("refs/main")
        if let hashString = try? String(contentsOf: refsMain, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !hashString.isEmpty {
            let snapshot = snapshotsDir.appendingPathComponent(hashString, isDirectory: true)
            if FileManager.default.fileExists(atPath: snapshot.path) {
                return snapshot
            }
        }

        let mainSnapshot = snapshotsDir.appendingPathComponent("main", isDirectory: true)
        if FileManager.default.fileExists(atPath: mainSnapshot.path) {
            return mainSnapshot
        }

        guard let contents = try? FileManager.default.contentsOfDirectory(
            at: snapshotsDir,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else { return nil }

        let directories = contents.filter { url in
            (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
        }
        return directories.count == 1 ? directories[0] : nil
    }

    /// The directory to load a model from when it does not live at the materialized
    /// `<cacheRoot>/models/<id>` path that `HubApi` resolves, i.e. when the user copied
    /// it in by hand or it came from `huggingface-cli`. Returns nil when the standard
    /// path applies, so callers keep the normal id-based flow (and its download
    /// behaviour) for everything else.
    public static func localLoadDirectory(for modelId: String) -> URL? {
        guard materializedDirectory(for: modelId) == nil else { return nil }
        guard isDownloaded(modelId) else { return nil }
        return modelContentDirectories(for: modelId).first
    }

    public static func isDownloaded(_ modelId: String) -> Bool {
        verifyModelIntegrity(for: modelId, logFailures: false)
    }

    // MARK: — Model Config Inspection

    /// Read the model's maximum context length from its config.json.
    /// Checks `text_config.max_position_embeddings` first (VLM/MoE models),
    /// then falls back to top-level `max_position_embeddings`.
    public static func readMaxContextLength(for modelId: String) -> Int? {
        guard let config = readModelConfig(for: modelId) else { return nil }

        // VLM/MoE models nest the context length in text_config
        if let textConfig = config["text_config"] as? [String: Any],
           let maxPos = textConfig["max_position_embeddings"] as? Int {
            return maxPos
        }

        // Standard LLMs have it at top level
        if let maxPos = config["max_position_embeddings"] as? Int {
            return maxPos
        }

        // Fallback: some models use n_ctx or max_seq_len
        if let nCtx = config["n_ctx"] as? Int { return nCtx }
        if let maxSeq = config["max_seq_len"] as? Int { return maxSeq }

        return nil
    }

    /// Read the raw config.json dictionary for a downloaded model.
    /// Verifies that all required safetensors files are present in the snapshot directory.
    /// This prevents the engine from entering `.ready` state if a download was interrupted or corrupted.
    public static func verifyModelIntegrity(for modelId: String) -> Bool {
        verifyModelIntegrity(for: modelId, logFailures: true)
    }

    private static func verifyModelIntegrity(for modelId: String, logFailures: Bool) -> Bool {
        if hasIncompleteFiles(for: modelId) {
            if logFailures { print("[ModelStorage] Integrity Check Failed: Incomplete download files remain for \(modelId)") }
            return false
        }

        for directory in modelContentDirectories(for: modelId) {
            if validateModelFiles(in: directory, logFailures: logFailures) {
                return true
            }
        }

        if logFailures { print("[ModelStorage] Integrity Check Failed: No valid model files found for \(modelId)") }
        return false
    }

    private static func modelContentDirectories(for modelId: String) -> [URL] {
        var directories: [URL] = []
        func append(_ url: URL?) {
            guard let url, !directories.contains(url) else { return }
            directories.append(url)
        }

        append(materializedDirectory(for: modelId))
        append(resolvedSnapshotDirectory(for: modelId))
        // Issue #110: a `models--org--name` folder copied by hand often holds the
        // weights directly, with no `snapshots/<hash>/` level for the HF cache layout.
        append(hubCacheDirectory(for: modelId))
        // …and a folder copied in without the `models--` prefix at all.
        append(plainDirectory(for: modelId))
        return directories
    }

    private static func validateModelFiles(in snapshotDir: URL, logFailures: Bool) -> Bool {
        // 0. Verify core metadata files
        let requiredJsonFiles = ["config.json", "tokenizer.json"]
        for file in requiredJsonFiles {
            let path = snapshotDir.appendingPathComponent(file)
            if !FileManager.default.fileExists(atPath: path.path) {
                // Some models might not have tokenizer.json if they use tokenizer.model, so we only strictly enforce config.json
                if file == "config.json" {
                    if logFailures { print("[ModelStorage] Integrity Check Failed: Missing \(file) in \(snapshotDir.path)") }
                    return false
                }
            } else if fileSizeResolvingSymlink(path) == 0 {
                if logFailures { print("[ModelStorage] Integrity Check Failed: \(file) is corrupted (0 bytes)") }
                return false
            }
        }

        // 1. Try to read model.safetensors.index.json
        let indexJsonPath = snapshotDir.appendingPathComponent("model.safetensors.index.json")
        if FileManager.default.fileExists(atPath: indexJsonPath.path) {
            guard let data = try? Data(contentsOf: indexJsonPath),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let weightMap = json["weight_map"] as? [String: String] else {
                return false
            }
            // Collect all unique safetensors filenames
            let requiredFiles = Set(weightMap.values)
            var totalShardBytes: Int64 = 0
            for file in requiredFiles {
                let filePath = snapshotDir.appendingPathComponent(file)
                guard let size = fileSizeResolvingSymlink(filePath) else {
                    if logFailures { print("[ModelStorage] Integrity Check Failed: Missing \(file)") }
                    return false
                }
                guard size > 1024 else {
                    if logFailures { print("[ModelStorage] Integrity Check Failed: \(file) is too small (\(size) bytes)") }
                    return false
                }
                totalShardBytes += size
            }

            if let metadata = json["metadata"] as? [String: Any],
               let expectedTensorBytes = int64Value(metadata["total_size"]),
               totalShardBytes < expectedTensorBytes {
                if logFailures {
                    print("[ModelStorage] Integrity Check Failed: shard bytes \(totalShardBytes) below index total_size \(expectedTensorBytes)")
                }
                return false
            }
            return true
        }

        // 2. If no index.json, it might be a single safetensors file model
        let singleSafetensors = snapshotDir.appendingPathComponent("model.safetensors")
        if let size = fileSizeResolvingSymlink(singleSafetensors), size > 1024 {
            return true
        }

        if logFailures { print("[ModelStorage] Integrity Check Failed: No safetensors found in \(snapshotDir.path)") }
        return false
    }

    public static func readModelConfig(for modelId: String) -> [String: Any]? {
        for directory in modelContentDirectories(for: modelId) {
            let configPath = directory.appendingPathComponent("config.json")
            guard let data = try? Data(contentsOf: configPath),
                  let config = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { continue }
            return config
        }
        return nil
    }

    // MARK: — Disk Operations

    /// Total bytes used by all model files on disk.
    public static func totalDiskUsage() -> Int64 {
        guard FileManager.default.fileExists(atPath: cacheRoot.path) else { return 0 }
        return directorySize(at: cacheRoot)
    }

    /// Bytes used by a specific model on disk.
    public static func sizeOnDisk(for modelId: String) -> Int64 {
        associatedDirectories(for: modelId).reduce(Int64(0)) { $0 + directorySize(at: $1) }
    }

    /// Delete all cached files for a model.
    public static func delete(_ modelId: String) throws {
        var firstError: Error?
        for dir in associatedDirectories(for: modelId) {
            do {
                try FileManager.default.removeItem(at: dir)
            } catch {
                if firstError == nil { firstError = error }
            }
        }
        if let firstError { throw firstError }
    }

    // MARK: — iCloud Exclusion (iOS)

    /// Mark a URL as excluded from iCloud backup.
    /// Call this after creating any model storage directory on iOS.
    public static func excludeFromBackup(_ url: URL) {
        var mutable = url
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? mutable.setResourceValues(values)
    }

    // MARK: — Scan

    public struct ScannedModel: Sendable {
        public let modelId: String
        public let cacheDirectory: URL
        public let sizeBytes: Int64
        public let modifiedDate: Date?
    }

    /// Scan the cache root and return all recognizable downloaded models.
    ///
    /// Recognised layouts, all rooted at `cacheRoot`:
    ///   - `models--org--name/snapshots/<hash>/`  — the huggingface-cli cache layout
    ///   - `models--org--name/`                   — the same folder copied by hand,
    ///                                              weights sitting directly inside
    ///   - `models/org/name/`                     — Swift Hub's materialized layout
    ///   - `org/name/` or `name/`                 — a model folder copied straight in
    ///
    /// The last two forms exist because users copy model folders into the cache by
    /// hand and expect them to appear (issue #110). Directories that look like a model
    /// but fail verification are reported by `diagnoseUnrecognizedDirectories()`.
    public static func scanDownloadedModels() -> [ScannedModel] {
        guard FileManager.default.fileExists(atPath: cacheRoot.path),
              let contents = try? FileManager.default.contentsOfDirectory(
                at: cacheRoot,
                includingPropertiesForKeys: [.contentModificationDateKey],
                options: [.skipsHiddenFiles]
              )
        else { return [] }

        var resultsById: [String: ScannedModel] = [:]
        for dir in contents {
            if dir.lastPathComponent.hasPrefix("models--") {
                let modelId = dir.lastPathComponent
                    .replacingOccurrences(of: "^models--", with: "", options: .regularExpression)
                    .replacingOccurrences(of: "--", with: "/")
                addScannedModelIfDownloaded(modelId: modelId, dir: dir, resultsById: &resultsById)
            } else if dir.lastPathComponent == "models" {
                guard let organizations = try? FileManager.default.contentsOfDirectory(
                    at: dir,
                    includingPropertiesForKeys: [.contentModificationDateKey],
                    options: [.skipsHiddenFiles]
                ) else { continue }

                for organization in organizations where directoryExists(organization) {
                    guard let modelDirs = try? FileManager.default.contentsOfDirectory(
                        at: organization,
                        includingPropertiesForKeys: [.contentModificationDateKey],
                        options: [.skipsHiddenFiles]
                    ) else { continue }

                    for modelDir in modelDirs where directoryExists(modelDir) {
                        let modelId = "\(organization.lastPathComponent)/\(modelDir.lastPathComponent)"
                        addScannedModelIfDownloaded(modelId: modelId, dir: modelDir, resultsById: &resultsById)
                    }
                }
            } else if directoryExists(dir) {
                // Hand-copied folder: either the model itself, or an org directory
                // holding one (issue #110).
                let name = dir.lastPathComponent
                if containsModelConfig(dir) {
                    addScannedModelIfDownloaded(modelId: name, dir: dir, resultsById: &resultsById)
                } else {
                    let children = (try? FileManager.default.contentsOfDirectory(
                        at: dir,
                        includingPropertiesForKeys: [.contentModificationDateKey],
                        options: [.skipsHiddenFiles]
                    )) ?? []
                    for child in children where directoryExists(child) && containsModelConfig(child) {
                        addScannedModelIfDownloaded(
                            modelId: "\(name)/\(child.lastPathComponent)",
                            dir: child,
                            resultsById: &resultsById
                        )
                    }
                }
            }
        }
        return resultsById.values.sorted { ($0.modifiedDate ?? .distantPast) > ($1.modifiedDate ?? .distantPast) }
    }

    /// A directory holding a model's own files (as opposed to a cache wrapper).
    private static func containsModelConfig(_ directory: URL) -> Bool {
        FileManager.default.fileExists(atPath: directory.appendingPathComponent("config.json").path)
    }

    /// Directories under the cache root that look like a model but were not returned
    /// by `scanDownloadedModels()`, paired with the reason each was rejected.
    ///
    /// Without this, a model the app refuses to list is indistinguishable from one it
    /// never looked at — which is what made issue #110 hard to diagnose.
    /// - Parameter knownModels: the result of a `scanDownloadedModels()` the caller
    ///   already performed; omit to run a fresh scan.
    public static func diagnoseUnrecognizedDirectories(
        knownModels: [ScannedModel]? = nil
    ) -> [(directory: URL, reason: String)] {
        // Compare canonical paths: contentsOfDirectory resolves symlinks (/var →
        // /private/var) while cacheDirectory builds paths from cacheRoot verbatim,
        // so raw string comparison would report every valid model as unrecognized.
        let recognized = Set((knownModels ?? scanDownloadedModels()).map { canonicalPath($0.cacheDirectory) })
        guard let contents = try? FileManager.default.contentsOfDirectory(
            at: cacheRoot,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) else { return [] }

        var findings: [(directory: URL, reason: String)] = []
        for dir in contents where directoryExists(dir) && dir.lastPathComponent != "models" {
            let candidates = containsModelConfig(dir)
                ? [dir]
                : ((try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])) ?? [])
                    .filter { directoryExists($0) && containsModelConfig($0) }

            for candidate in candidates where !recognized.contains(canonicalPath(candidate)) {
                findings.append((candidate, rejectionReason(for: candidate)))
            }
        }
        return findings
    }

    private static func canonicalPath(_ url: URL) -> String {
        url.resolvingSymlinksInPath().standardizedFileURL.path
    }

    private static func rejectionReason(for directory: URL) -> String {
        if countIncompleteFiles(in: directory) > 0 {
            return "contains .incomplete files — finish or delete the partial download"
        }
        let indexPath = directory.appendingPathComponent("model.safetensors.index.json")
        let hasIndex = FileManager.default.fileExists(atPath: indexPath.path)
        let single = directory.appendingPathComponent("model.safetensors")
        let hasSingle = (fileSizeResolvingSymlink(single) ?? 0) > 1024
        if !hasIndex && !hasSingle {
            let shards = (try? FileManager.default.contentsOfDirectory(atPath: directory.path))?
                .filter { $0.hasSuffix(".safetensors") } ?? []
            if shards.isEmpty {
                return "no .safetensors weights found (GGUF and .npz models are not supported)"
            }
            return "sharded weights present but model.safetensors.index.json is missing"
        }
        return "weight files missing or truncated relative to model.safetensors.index.json"
    }

    private static func addScannedModelIfDownloaded(
        modelId: String,
        dir: URL,
        resultsById: inout [String: ScannedModel]
    ) {
        guard isDownloaded(modelId) else { return }

        let modified = (try? dir.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate
        let candidate = ScannedModel(
            modelId: modelId,
            cacheDirectory: cacheDirectory(for: modelId) ?? dir,
            sizeBytes: sizeOnDisk(for: modelId),
            modifiedDate: modified
        )

        if let existing = resultsById[modelId],
           (existing.modifiedDate ?? .distantPast) >= (candidate.modifiedDate ?? .distantPast) {
            return
        }
        resultsById[modelId] = candidate
    }

    // MARK: — Incomplete Downloads

    /// A model whose download was interrupted and can be resumed.
    public struct IncompleteDownload: Identifiable, Sendable {
        public let id: String  // modelId
        public let cacheDirectory: URL
        /// Bytes downloaded so far (sum of complete + incomplete files)
        public let downloadedBytes: Int64
        /// When the partial download was last modified
        public let lastModified: Date?
    }

    /// Check whether a model directory has any `.incomplete` partial files (iOS path)
    /// or incomplete blobs (macOS HubApi path).
    public static func hasIncompleteFiles(for modelId: String) -> Bool {
        associatedDirectories(for: modelId).contains { countIncompleteFiles(in: $0) > 0 }
    }

    /// Scan the cache root for model directories that have partial downloads
    /// but are NOT fully downloaded (i.e. `isDownloaded()` returns false, or
    /// the directory contains `.incomplete` files).
    public static func scanIncompleteDownloads() -> [IncompleteDownload] {
        guard FileManager.default.fileExists(atPath: cacheRoot.path),
              let contents = try? FileManager.default.contentsOfDirectory(
                at: cacheRoot,
                includingPropertiesForKeys: [.contentModificationDateKey],
                options: [.skipsHiddenFiles]
              )
        else { return [] }

        var resultsById: [String: IncompleteDownload] = [:]
        for dir in contents {
            if dir.lastPathComponent.hasPrefix("models--") {
                let modelId = dir.lastPathComponent
                    .replacingOccurrences(of: "^models--", with: "", options: .regularExpression)
                    .replacingOccurrences(of: "--", with: "/")
                addIncompleteDownloadIfNeeded(modelId: modelId, dir: dir, resultsById: &resultsById)
            } else if dir.lastPathComponent == "models" {
                guard let organizations = try? FileManager.default.contentsOfDirectory(
                    at: dir,
                    includingPropertiesForKeys: [.contentModificationDateKey],
                    options: [.skipsHiddenFiles]
                ) else { continue }

                for organization in organizations where directoryExists(organization) {
                    guard let modelDirs = try? FileManager.default.contentsOfDirectory(
                        at: organization,
                        includingPropertiesForKeys: [.contentModificationDateKey],
                        options: [.skipsHiddenFiles]
                    ) else { continue }

                    for modelDir in modelDirs where directoryExists(modelDir) {
                        let modelId = "\(organization.lastPathComponent)/\(modelDir.lastPathComponent)"
                        addIncompleteDownloadIfNeeded(modelId: modelId, dir: modelDir, resultsById: &resultsById)
                    }
                }
            }
        }
        return resultsById.values.sorted { ($0.lastModified ?? .distantPast) > ($1.lastModified ?? .distantPast) }
    }

    private static func addIncompleteDownloadIfNeeded(
        modelId: String,
        dir: URL,
        resultsById: inout [String: IncompleteDownload]
    ) {
        // Skip fully completed models unless they have leftover .incomplete files.
        if isDownloaded(modelId) && !hasIncompleteFiles(for: modelId) {
            return
        }

        // Must have SOME content (not just an empty directory).
        let size = directorySize(at: dir)
        guard size > 0 else { return }

        let modified = (try? dir.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate
        let candidate = IncompleteDownload(
            id: modelId,
            cacheDirectory: dir,
            downloadedBytes: size,
            lastModified: modified
        )

        if let existing = resultsById[modelId],
           (existing.lastModified ?? .distantPast) >= (candidate.lastModified ?? .distantPast) {
            return
        }
        resultsById[modelId] = candidate
    }

    /// Count `.incomplete` files in a directory tree.
    private static func countIncompleteFiles(in directory: URL) -> Int {
        guard let enumerator = FileManager.default.enumerator(
            at: directory,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) else { return 0 }

        var count = 0
        for case let fileURL as URL in enumerator {
            if fileURL.pathExtension == "incomplete" {
                count += 1
            }
        }
        return count
    }

    // MARK: — Helpers

    private static func associatedDirectories(for modelId: String) -> [URL] {
        var candidates = [
            materializedDirectoryURL(for: modelId),
            hubCacheDirectoryURL(for: modelId),
        ]
        // Only a hand-copied folder that actually exists — plainDirectoryURL for an
        // unknown id can point at an unrelated directory, and delete() walks this list.
        if let plain = plainDirectory(for: modelId) {
            candidates.append(plain)
        }
        // Every path here is a removeItem target. An empty id makes
        // appendingPathComponent("") a no-op, so materializedDirectoryURL(for: "")
        // resolves to the shared `models/` wrapper and would delete every downloaded
        // model; the same guard covers `..` escapes on any of the three layouts.
        candidates = candidates.filter { isSafeModelDirectory($0) }

        var seen = Set<String>()
        return candidates.filter { url in
            guard directoryExists(url), !seen.contains(url.path) else { return false }
            seen.insert(url.path)
            return true
        }
    }

    private static func fileSizeResolvingSymlink(_ url: URL) -> Int64? {
        let resolved = url.resolvingSymlinksInPath()
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: resolved.path) else { return nil }
        if let size = attrs[.size] as? Int64 { return size }
        if let size = attrs[.size] as? NSNumber { return size.int64Value }
        return nil
    }

    private static func int64Value(_ value: Any?) -> Int64? {
        switch value {
        case let value as Int64: return value
        case let value as Int: return Int64(value)
        case let value as NSNumber: return value.int64Value
        case let value as String: return Int64(value)
        default: return nil
        }
    }

    private static func ensureDirectory(_ url: URL) {
        guard !FileManager.default.fileExists(atPath: url.path) else { return }
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        #if !os(macOS)
        excludeFromBackup(url)
        #endif
    }

    static func directorySize(at url: URL) -> Int64 {
        guard let enumerator = FileManager.default.enumerator(
            at: url,
            includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else { return 0 }

        var total: Int64 = 0
        for case let fileURL as URL in enumerator {
            let size = (try? fileURL.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
            total += Int64(size)
        }
        return total
    }
}
