import XCTest
@testable import SwiftBuddy

final class ProjectMinerTests: XCTestCase {
    
    var miner: ProjectMiner!
    
    @MainActor
    override func setUpWithError() throws {
        miner = ProjectMiner.shared
    }
    
    // MARK: - Safe Chunking Tests
    
    @MainActor
    func testChunkBySentences_RespectsLimits() {
        let longLine = String(repeating: "A", count: 1000)
        let text = "\(longLine)\n\(longLine)\n\(longLine)\n\(longLine)\n\(longLine)\n" // 5000+ chars
        
        let chunks = miner.chunkBySentences(text: text, maxChars: 4000)
        
        // At 1001 chars per line, it should fit 3 lines into chunk 1 (~3003) and 2 lines into chunk 2 (~2002)
        XCTAssertEqual(chunks.count, 2)
        XCTAssertTrue(chunks[0].count <= 4000)
        XCTAssertTrue(chunks[1].count <= 4000)
    }
    
    @MainActor
    func testChunkBySentences_ShortText() {
        let shortText = "Hello world\nThis is fine."
        let chunks = miner.chunkBySentences(text: shortText, maxChars: 4000)
        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks[0], "Hello world\nThis is fine.\n")
    }
}
