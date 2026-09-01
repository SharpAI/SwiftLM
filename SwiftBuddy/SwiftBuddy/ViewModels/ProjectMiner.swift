import Foundation
import MLXInferenceCore

@MainActor
public final class ProjectMiner: ObservableObject {
    public static let shared = ProjectMiner()
    
    @Published public var isMining: Bool = false
    @Published public var progress: Double = 0.0
    @Published public var status: String = "Idle"
    
    // Ignore heavy or binary files
    private let ignoredExtensions: Set<String> = [
        "png", "jpg", "jpeg", "gif", "pdf", "zip", "tar", "gz", "mp4", "mov", "app", 
        "metallib", "bin", "o", "a", "dylib", "so"
    ]
    private let ignoredDirectories: Set<String> = [
        ".git", ".build", "build", "DerivedData", "Pods", "node_modules", "vendor"
    ]
    
    private init() {}
    
    /// Recursively mines a directory, chunks large files, and sequentially prompts the ExtractionService
    public func mineDirectory(url: URL, wingName: String, engine: InferenceEngine) async {
        guard !isMining else { return }
        isMining = true
        progress = 0.0
        status = "Scanning directory structure..."
        
        // 1. Gather Files
        var validFiles: [URL] = []
        guard let enumerator = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.isDirectoryKey]) else {
            status = "Error: Could not read directory."
            isMining = false
            return
        }
        
        while let fileURL = enumerator.nextObject() as? URL {
            let resourceValues = try? fileURL.resourceValues(forKeys: [.isDirectoryKey])
            let isDirectory = resourceValues?.isDirectory ?? false
            
            if isDirectory {
                if ignoredDirectories.contains(fileURL.lastPathComponent) {
                    enumerator.skipDescendants()
                }
                continue
            }
            
            let ext = fileURL.pathExtension.lowercased()
            if ignoredExtensions.contains(ext) { continue }
            
            validFiles.append(fileURL)
        }
        
        if validFiles.isEmpty {
            status = "No processable text files found."
            isMining = false
            return
        }
        
        // 2. Process Files Sequentially
        let totalFiles = Double(validFiles.count)
        
        for (index, fileURL) in validFiles.enumerated() {
            status = "Mining (\(index + 1)/\(validFiles.count)): \(fileURL.lastPathComponent)"
            
            guard let content = try? String(contentsOf: fileURL, encoding: .utf8) else {
                continue // Skip if not valid UTF8 text
            }
            
            // 3. Chunk if necessary (~4000 characters to stay safe inside typical context limits without token math overhead)
            let chunks = chunkBySentences(text: content, maxChars: 4000)
            
            for chunk in chunks {
                // Prepend context about what file this is
                let contextBlock = "File: \(fileURL.lastPathComponent)\nPath: \(fileURL.relativePath)\n\n\(chunk)"
                
                // Keep calling ExtractionService. We wait until the model finishes before proceeding.
                await ExtractionService.shared.mine(textBlock: contextBlock, wing: wingName, engine: engine)
            }
            
            progress = Double(index + 1) / totalFiles
            
            // Artificial delay to prevent thermal runaway if desired
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
        
        status = "Mining Complete. Ingested \(validFiles.count) files."
        isMining = false
    }
    
    /// Basic semantic chunking: splits by newline to preserve paragraph/code structure but caps size
    internal func chunkBySentences(text: String, maxChars: Int) -> [String] {
        var chunks: [String] = []
        var currentChunk = ""
        
        let lines = text.components(separatedBy: .newlines)
        
        for line in lines {
            if currentChunk.count + line.count > maxChars {
                if !currentChunk.isEmpty {
                    chunks.append(currentChunk)
                    currentChunk = ""
                }
            }
            currentChunk += line + "\n"
        }
        
        if !currentChunk.isEmpty {
            chunks.append(currentChunk)
        }
        
        return chunks
    }
}
