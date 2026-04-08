import XCTest
import SwiftData
@testable import SwiftBuddy

@MainActor
final class MemoryPalaceServiceTests: XCTestCase {
    
    var service: MemoryPalaceService!
    var modelContainer: ModelContainer!
    
    override func setUpWithError() throws {
        let config = ModelConfiguration(isStoredInMemoryOnly: true)
        modelContainer = try ModelContainer(for: PalaceWing.self, PalaceRoom.self, MemoryEntry.self, KnowledgeGraphTriple.self, configurations: config)
        
        // Re-initialize the service to use our memory context
        service = MemoryPalaceService.shared
        service.modelContext = modelContainer.mainContext
        
        // Clean start
        let context = modelContainer.mainContext
        let fetchDesc = FetchDescriptor<PalaceWing>()
        let existingWings = try context.fetch(fetchDesc)
        for w in existingWings {
            context.delete(w)
        }
        
        let tripleFetch = FetchDescriptor<KnowledgeGraphTriple>()
        let existingTriples = try context.fetch(tripleFetch)
        for t in existingTriples {
            context.delete(t)
        }
        try context.save()
    }
    
    // MARK: - Tier 1: Core Palace Structure
    
    func testFeature1_WingsCRUD() throws {
        print("Starting test...")
        // Create (implicit via saveMemory)
        try service.saveMemory(wingName: "test_wing_1", roomName: "general", text: "memory 1")
        print("Saved memory 1")
        try service.saveMemory(wingName: "test_wing_2", roomName: "general", text: "memory 2")
        print("Saved memory 2")
        
        // List wings
        let wings = try service.listWings()
        print("wings: \(wings)")
        XCTAssertEqual(wings.count, 2)
        XCTAssertTrue(wings.contains("test_wing_1"))
        XCTAssertTrue(wings.contains("test_wing_2"))
        
        // Delete wing
        try service.deleteWing("test_wing_1")
        print("Deleted wing 1")
        
        let remainingWings = try service.listWings()
        print("Remaining wings: \(remainingWings)")
        XCTAssertEqual(remainingWings.count, 1)
        XCTAssertTrue(remainingWings.contains("test_wing_2"))
        XCTAssertFalse(remainingWings.contains("test_wing_1"))
        
        // Verify cascading delete: memory should be gone
        let memories = try service.searchMemories(query: "memory 1", wingName: "test_wing_1")
        print("Memories count: \(memories.count)")
        XCTAssertTrue(memories.isEmpty)
    }
    func testFeature3_HallDiscoveries() throws {
        try service.saveMemory(wingName: "science", roomName: "physics", text: "gravitational waves exist", type: "hall_discoveries")
        let memories = try service.searchMemories(query: "waves", wingName: "science")
        XCTAssertEqual(memories.first?.hallType, "hall_discoveries")
    }
    
    // MARK: - Tier 1: Tunnels
    func testFeature6_Tunnels() throws {
        try service.saveMemory(wingName: "kai", roomName: "auth-migration", text: "using JWT")
        try service.saveMemory(wingName: "driftwood", roomName: "auth-migration", text: "implemented JWT")
        
        let relatedWings = try service.findTunnels(roomName: "auth-migration")
        XCTAssertEqual(relatedWings.count, 2)
        XCTAssertTrue(relatedWings.contains("kai"))
        XCTAssertTrue(relatedWings.contains("driftwood"))
    }
    
    // MARK: - Tier 2: Search & Retrieval
    func testFeature9_10_SearchFilters() throws {
        try service.saveMemory(wingName: "simba", roomName: "security", text: "password max len 64", type: "hall_facts")
        try service.saveMemory(wingName: "simba", roomName: "billing", text: "using stripe", type: "hall_facts")
        try service.saveMemory(wingName: "simba", roomName: "security", text: "found xss bug", type: "hall_events")
        
        // Filter by Room
        let securityMemories = try service.searchMemories(query: "password", wingName: "simba", roomName: "security")
        XCTAssertTrue(securityMemories.contains { $0.room?.name == "security" })
        XCTAssertFalse(securityMemories.contains { $0.room?.name == "billing" })
        
        // Filter by Hall
        let factsMemories = try service.searchMemories(query: "password stripe", wingName: "simba", hallType: "hall_facts")
        XCTAssertTrue(factsMemories.contains { $0.hallType == "hall_facts" })
        XCTAssertFalse(factsMemories.contains { $0.hallType == "hall_events" })
    }
    
    func testFeature11_CrossWingSearch() throws {
        try service.saveMemory(wingName: "orion", roomName: "db", text: "postgres is better")
        try service.saveMemory(wingName: "nova", roomName: "db", text: "postgres used here")
        
        let memories = try service.searchAllMemories(query: "postgres")
        XCTAssertTrue(memories.contains { $0.room?.wing?.name == "orion" })
        XCTAssertTrue(memories.contains { $0.room?.wing?.name == "nova" })
    }
    
    func testFeature12_DuplicateDetection() throws {
        try service.saveMemory(wingName: "dupe", roomName: "general", text: "this is exact text")
        let didSave = try service.saveMemory(wingName: "dupe", roomName: "general", text: "this is exact text")
        
        XCTAssertFalse(didSave, "Duplicate detection should prevent saving exact semantic matches")
        let check = try service.searchMemories(query: "exact text", wingName: "dupe")
        XCTAssertEqual(check.count, 1)
    }
    
    // MARK: - Tier 4: Tool Calling (MCP Parity)
    func testFeature20to25_MCP_Taxonomy_Status_Delete() throws {
        try service.saveMemory(wingName: "mcp_wing", roomName: "tools", text: "this will be deleted")
        try service.saveMemory(wingName: "mcp_wing", roomName: "tools", text: "this stays here", type: "hall_facts")
        
        // Status
        let status = try service.getPalaceStatus()
        XCTAssertGreaterThanOrEqual(status.wings, 1)
        
        // Taxonomy
        let tax = try service.getTaxonomy()
        XCTAssertTrue(tax.contains("Wing: mcp_wing"))
        
        // Closet
        let closet = try service.getCloset(wingName: "mcp_wing", roomName: "tools")
        XCTAssertTrue(closet.contains("[hall_facts] this stays here"))
        XCTAssertTrue(closet.contains("Closet for mcp_wing/tools"))
        
        // Delete Memory
        try service.deleteMemory(wingName: "mcp_wing", roomName: "tools", textMatch: "will be deleted")
    }
    
    // MARK: - Tier 5: Temporal Knowledge Graph
    func testFeature26to30_KnowledgeGraph() throws {
        // Assert creation
        try service.addTriple(subject: "Simba", predicate: "uses_language", object: "Swift 6")
        try service.addTriple(subject: "Simba", predicate: "builds", object: "SwiftBuddy")
        
        // Assert queries
        let properties = try service.queryEntity("Simba")
        XCTAssertEqual(properties.count, 2)
        XCTAssertTrue(properties.contains { $0.predicate == "uses_language" && $0.object == "Swift 6" })
        
        // Assert Contradiction Detection & Temporal Overwrite (Subject + Predicate is Unique)
        try service.addTriple(subject: "simba", predicate: "uses_language", object: "Swift 6.1 Strict Concurrency")
        
        let newProps = try service.queryEntity("simba")
        XCTAssertEqual(newProps.count, 2, "Duplicate predicates should overwrite, not append")
        
        let target = newProps.first { $0.predicate == "uses_language" }
        XCTAssertEqual(target?.object, "Swift 6.1 Strict Concurrency", "Temporal overwrite failed")
    }
}
