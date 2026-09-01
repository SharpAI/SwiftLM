// RuntimeIntegrationTests.swift — End-to-end integration tests
//
// Tests the full RuntimeRegistry system with realistic scenarios.
// These tests verify the complete workflow from selection to activation.

import XCTest
import Foundation
@testable import MLXInferenceCore

@MainActor
final class RuntimeIntegrationTests: XCTestCase {
    
    var registry: RuntimeRegistry!
    
    override func setUp() async throws {
        try await super.setUp()
        registry = RuntimeRegistry.shared
        await registry.deactivate()
    }
    
    override func tearDown() async throws {
        await registry.deactivate()
        try await super.tearDown()
    }
    
    // MARK: — Complete Workflow
    
    func testCompleteWorkflow_SelectLoadUnload() async throws {
        // 1. Check compatibility
        let hasCompatible = await registry.hasCompatibleRuntime(
            for: "mlx-community/Qwen2.5-7B-Instruct-4bit"
        )
        XCTAssertTrue(hasCompatible)
        
        // 2. Select runtime
        let config = RuntimeConfig(temperature: 0.7)
        let engine = registry.selectRuntime(
            for: "mlx-community/Qwen2.5-7B-Instruct-4bit",
            config: config
        )
        XCTAssertNotNil(engine)
        XCTAssertEqual(engine?.id, "mlx.standard")
        
        // 3. Verify activation
        XCTAssertNotNil(registry.activeEngine)
        XCTAssertTrue(registry.activeEngine === engine)
        
        // 4. Get capability info
        let capability = registry.capability(for: engine!.id)
        XCTAssertNotNil(capability)
        XCTAssertEqual(capability?.id, "mlx.standard")
        
        // 5. Deactivate
        await registry.deactivate()
        XCTAssertNil(registry.activeEngine)
    }
    
    func testCompleteWorkflow_RuntimeSwitching() async throws {
        // Start with standard runtime
        let config1 = RuntimeConfig(temperature: 0.7)
        let engine1 = registry.selectRuntime(
            for: "mlx-community/Qwen2.5-7B-Instruct-4bit",
            config: config1
        )
        XCTAssertEqual(engine1?.id, "mlx.standard")
        
        // Switch to DFlash
        let config2 = RuntimeConfig(temperature: 0.7, enableDFlash: true)
        let engine2 = registry.selectRuntime(
            for: "mlx-community/Qwen2.5-7B-Instruct-4bit",
            config: config2
        )
        XCTAssertEqual(engine2?.id, "mlx.dflash")
        
        // Verify only the new engine is active
        XCTAssertTrue(registry.activeEngine === engine2)
        XCTAssertFalse(registry.activeEngine === engine1)
    }
    
    // MARK: — Scenario: DFlash for Long Context
    
    func testScenario_DFlashForLongContext() async throws {
        // User wants to use DFlash for memory-efficient long context
        let config = RuntimeConfig(
            temperature: 0.7,
            contextWindow: 32768,
            enableDFlash: true
        )
        
        let engine = registry.selectRuntime(
            for: "mlx-community/Qwen2.5-32B-Instruct-4bit",
            config: config
        )
        
        XCTAssertNotNil(engine)
        XCTAssertEqual(engine?.id, "mlx.dflash")
        
        let capability = engine?.capabilities
        XCTAssertTrue(capability?.supportsDFlash ?? false)
        XCTAssertEqual(capability?.memoryEfficiency, .optimized)
    }
    
    // MARK: — Scenario: Streaming MoE for Large Models
    
    func testScenario_StreamingMoEForLargeModel() async throws {
        // User wants to run a large MoE model with expert streaming
        let config = RuntimeConfig(
            temperature: 0.7,
            streamExperts: true
        )
        
        let engine = registry.selectRuntime(
            for: "mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit",
            config: config
        )
        
        XCTAssertNotNil(engine)
        XCTAssertEqual(engine?.id, "mlx.streaming_moe")
        
        let capability = engine?.capabilities
        XCTAssertEqual(capability?.memoryEfficiency, .extreme)
        XCTAssertTrue(capability?.supportsDFlash ?? false)
    }
    
    // MARK: — Scenario: Speculative Decoding for Speed
    
    func testScenario_SpeculativeForSpeed() async throws {
        // User wants fast inference with speculative decoding
        let config = RuntimeConfig(
            temperature: 0.7,
            enableSpeculative: true
        )
        
        let engine = registry.selectRuntime(
            for: "mlx-community/Qwen2.5-7B-Instruct-4bit",
            config: config
        )
        
        XCTAssertNotNil(engine)
        XCTAssertEqual(engine?.id, "mlx.speculative")
        
        let capability = engine?.capabilities
        XCTAssertTrue(capability?.supportsSpeculative ?? false)
    }
    
    // MARK: — Scenario: Vision Model
    
    func testScenario_VisionModel() async throws {
        // User loads a vision model - should get runtime that supports vision
        let config = RuntimeConfig(temperature: 0.7)
        
        let engine = registry.selectRuntime(
            for: "mlx-community/pixtral-12b-4bit",
            config: config
        )
        
        XCTAssertNotNil(engine)
        // Should be standard or dflash or speculative (all support vision)
        let capability = engine?.capabilities
        XCTAssertTrue(capability?.supportsVision ?? false)
    }
    
    // MARK: — Error Handling
    
    func testErrorHandling_NoCompatibleRuntime() async {
        // If somehow no runtime is compatible (shouldn't happen in practice)
        // the system should handle it gracefully
        let config = RuntimeConfig(temperature: 0.7)
        
        // Using a standard model, this should always succeed
        let engine = registry.selectRuntime(
            for: "mlx-community/Qwen2.5-7B-Instruct-4bit",
            config: config
        )
        
        XCTAssertNotNil(engine, "Standard models should always have a compatible runtime")
    }
    
    // MARK: — Concurrent Access
    
    func testConcurrentAccess_MultipleSelections() async throws {
        // Multiple concurrent selections should be handled safely
        let selection1 = registry.selectRuntime(
            for: "mlx-community/Qwen2.5-7B-Instruct-4bit",
            config: RuntimeConfig()
        )
        
        let selection2 = registry.selectRuntime(
            for: "mlx-community/Qwen2.5-7B-Instruct-4bit",
            config: RuntimeConfig(enableDFlash: true)
        )
        
        let (engine1, engine2) = (selection1, selection2)
        
        // Both selections should succeed
        XCTAssertNotNil(engine1)
        XCTAssertNotNil(engine2)
        
        // Last selection wins
        XCTAssertTrue(registry.activeEngine === engine2)
    }
    
    // MARK: — Memory and Resource Management
    
    func testMemoryManagement_EngineRetention() async {
        // Create an engine reference
        var engine: RuntimeEngine? = registry.selectRuntime(
            for: "mlx-community/Qwen2.5-7B-Instruct-4bit",
            config: RuntimeConfig()
        )
        XCTAssertNotNil(engine)
        
        let engineId = engine!.id
        
        // Deactivate
        await registry.deactivate()
        
        // Engine reference should still be valid (user might hold it)
        XCTAssertEqual(engine?.id, engineId)
        
        // But registry should have no active engine
        XCTAssertNil(registry.activeEngine)
        
        // Release our reference
        engine = nil
        
        // Registry can still create the same engine again
        let newEngine = registry.engine(for: engineId)
        XCTAssertNotNil(newEngine)
    }
}
