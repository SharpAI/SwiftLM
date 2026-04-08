import XCTest
import SwiftData
@testable import SwiftBuddy

final class Tier6AdvancedTests: XCTestCase {
    
    var service: MemoryPalaceService!
    var modelContainer: ModelContainer!
    
    @MainActor
    override func setUpWithError() throws {
        let config = ModelConfiguration(isStoredInMemoryOnly: true)
        modelContainer = try ModelContainer(for: PalaceWing.self, PalaceRoom.self, MemoryEntry.self, KnowledgeGraphTriple.self, configurations: config)
        
        service = MemoryPalaceService.shared
        service.modelContext = modelContainer.mainContext
        
        let context = modelContainer.mainContext
        for w in try context.fetch(FetchDescriptor<PalaceWing>()) { context.delete(w) }
        for t in try context.fetch(FetchDescriptor<KnowledgeGraphTriple>()) { context.delete(t) }
        try context.save()
    }
    
    @MainActor
    func testFeature31_SpecialistDiaryHooks() async throws {
        // Mocks MempalaceTools bridging mempalace_diary_save
        let result = try await MemoryPalaceTools.handleToolCall(name: "mempalace_diary_save", arguments: [
            "wing": "agent_alpha",
            "reflection": "I noticed the user prefers swiftdata over coredata"
        ])
        
        XCTAssertEqual(result, "Saved reflection to diary.")
        
        let context = modelContainer.mainContext
        let fetchDesc = FetchDescriptor<MemoryEntry>(predicate: #Predicate { $0.room?.name == "diary" })
        let entries = try context.fetch(fetchDesc)
        
        XCTAssertEqual(entries.count, 1)
        XCTAssertEqual(entries.first?.text, "I noticed the user prefers swiftdata over coredata")
        XCTAssertEqual(entries.first?.room?.wing?.name, "agent_alpha")
    }
    
    func testFeature32_AAAKCompressionEngine() {
        let rawParagraph = "The user is working on the Memory Palace and they was specifically using the AAAK text protocol to make it compress sentences."
        let compressed = AAAKCompressionEngine.compress(rawParagraph)
        
        XCTAssertTrue(compressed.hasPrefix("AAAK<<"))
        XCTAssertTrue(compressed.hasSuffix(">>"))
        
        // Stop words should be missing
        XCTAssertFalse(compressed.lowercased().contains("the "))
        XCTAssertFalse(compressed.lowercased().contains("is "))
        
        // Semantic nouns remain
        XCTAssertTrue(compressed.lowercased().contains("user"))
        XCTAssertTrue(compressed.lowercased().contains("memory"))
        
        // Verify length reduction
        XCTAssertTrue(compressed.count < rawParagraph.count)
    }
    
    @MainActor
    func testFeature33_WakeUpContextAssembly() throws {
        // 1. Inject Triples
        try service.addTriple(subject: "Simba", predicate: "building", object: "SwiftBuddy")
        
        // 2. Inject Diary entries
        try service.saveMemory(wingName: "alpha", roomName: "diary", text: "Reflecting on code coverage constraints today.", type: "hall_events")
        try service.saveMemory(wingName: "alpha", roomName: "diary", text: "Thinking about ordering a pizza for dinner.", type: "hall_events")
        
        let wakeup = try service.getWakeupContext(wingName: "alpha")
        print("WAKEUP DEBUG:", wakeup)
        
        XCTAssertTrue(wakeup.contains("--- L0/L1 WAKEUP CONTEXT ---"))
        XCTAssertTrue(wakeup.contains("[Simba] building SwiftBuddy"))
        XCTAssertTrue(wakeup.contains("- Thinking about ordering a pizza for dinner."))
    }
    
    func testFeature34_AutoSaveRollingHook() {
        let observer = AutoSaveObserver(threshold: 3)
        var flushCalled = false
        
        observer.onFlushRequired = {
            flushCalled = true
        }
        
        observer.recordMessage() // 1
        XCTAssertFalse(flushCalled)
        
        observer.recordMessage() // 2
        XCTAssertFalse(flushCalled)
        
        observer.recordMessage() // 3 -> Triggers flush
        XCTAssertTrue(flushCalled)
        XCTAssertEqual(observer.currentBufferCount(), 0)
    }
}
