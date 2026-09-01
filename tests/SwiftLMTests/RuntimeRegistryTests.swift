// RuntimeRegistryTests.swift — Unit tests for RuntimeRegistry
//
// Tests the core registry functionality: registration, lookup, activation,
// and compatibility checking.

import XCTest
import Foundation
@testable import MLXInferenceCore

@MainActor
final class RuntimeRegistryTests: XCTestCase {
    
    var registry: RuntimeRegistry!
    
    override func setUp() async throws {
        try await super.setUp()
        // Use a fresh registry instance for each test
        // Note: In production, RuntimeRegistry.shared is a singleton
        // For testing, we access it directly but reset state if needed
        registry = RuntimeRegistry.shared
    }
    
    override func tearDown() async throws {
        // Deactivate any active engine
        await registry.deactivate()
        try await super.tearDown()
    }
    
    // MARK: — Built-in Runtimes
    
    func testBuiltInRuntimesAreRegistered() {
        // All four built-in runtimes should be registered
        let runtimes = registry.availableRuntimes
        XCTAssertEqual(runtimes.count, 4, "Should have 4 built-in runtimes")
        
        let ids = Set(runtimes.map { $0.id })
        XCTAssertTrue(ids.contains("mlx.standard"))
        XCTAssertTrue(ids.contains("mlx.dflash"))
        XCTAssertTrue(ids.contains("mlx.speculative"))
        XCTAssertTrue(ids.contains("mlx.streaming_moe"))
    }
    
    func testStandardRuntimeCapabilities() {
        let capability = registry.capability(for: "mlx.standard")
        XCTAssertNotNil(capability)
        XCTAssertEqual(capability?.id, "mlx.standard")
        XCTAssertEqual(capability?.displayName, "MLX Standard")
        XCTAssertTrue(capability?.supportsStreaming ?? false)
        XCTAssertTrue(capability?.supportsVision ?? false)
        XCTAssertTrue(capability?.supportsAudio ?? false)
        XCTAssertFalse(capability?.supportsDFlash ?? true)
        XCTAssertFalse(capability?.supportsSpeculative ?? true)
        XCTAssertEqual(capability?.memoryEfficiency, .standard)
    }
    
    func testDFlashRuntimeCapabilities() {
        let capability = registry.capability(for: "mlx.dflash")
        XCTAssertNotNil(capability)
        XCTAssertTrue(capability?.supportsDFlash ?? false)
        XCTAssertEqual(capability?.memoryEfficiency, .optimized)
    }
    
    func testSpeculativeRuntimeCapabilities() {
        let capability = registry.capability(for: "mlx.speculative")
        XCTAssertNotNil(capability)
        XCTAssertTrue(capability?.supportsSpeculative ?? false)
        XCTAssertEqual(capability?.memoryEfficiency, .standard)
    }
    
    func testStreamingMoERuntimeCapabilities() {
        let capability = registry.capability(for: "mlx.streaming_moe")
        XCTAssertNotNil(capability)
        XCTAssertTrue(capability?.supportsDFlash ?? false)
        XCTAssertEqual(capability?.memoryEfficiency, .extreme)
        // Streaming MoE doesn't support vision/audio
        XCTAssertFalse(capability?.supportsVision ?? true)
        XCTAssertFalse(capability?.supportsAudio ?? true)
    }
    
    // MARK: — Engine Retrieval
    
    func testEngineRetrieval() {
        // Should be able to get engines for all registered runtimes
        let standardEngine = registry.engine(for: "mlx.standard")
        XCTAssertNotNil(standardEngine)
        XCTAssertEqual(standardEngine?.id, "mlx.standard")
        
        let dflashEngine = registry.engine(for: "mlx.dflash")
        XCTAssertNotNil(dflashEngine)
        XCTAssertEqual(dflashEngine?.id, "mlx.dflash")
    }
    
    func testEngineRetrieval_NonExistent() {
        let engine = registry.engine(for: "nonexistent.runtime")
        XCTAssertNil(engine, "Should return nil for non-existent runtime")
    }
    
    func testEngineRetrieval_IsCached() {
        // The registry creates new instances each time unless it's the active engine
        // This is by design - engines are heavyweight and we don't want multiple instances
        let engine1 = registry.engine(for: "mlx.standard")
        let engine2 = registry.engine(for: "mlx.standard")
        
        // These will be different instances since neither is active
        XCTAssertFalse(engine1 === engine2, "Non-active engines are not cached")
        
        // But if we activate an engine, subsequent calls should return the same instance
        _ = registry.selectRuntime(for: "model-id", config: RuntimeConfig())
        let active1 = registry.engine(for: "mlx.standard")
        let active2 = registry.engine(for: "mlx.standard")
        XCTAssertTrue(active1 === active2, "Active engine should be cached")
    }
    
    // MARK: — Compatibility Checking
    
    func testCompatibilityCheck_MoEModel() async {
        // Streaming MoE runtime should be compatible with MoE models
        let compatible = await registry.hasCompatibleRuntime(for: "mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit")
        XCTAssertTrue(compatible, "Should have compatible runtime for MoE model")
    }
    
    func testCompatibilityCheck_VisionModel() async {
        // Standard and DFlash runtimes support vision
        let compatible = await registry.hasCompatibleRuntime(for: "mlx-community/pixtral-12b-4bit")
        XCTAssertTrue(compatible, "Should have compatible runtime for vision model")
    }
    
    func testCompatibilityCheck_StandardModel() async {
        // Any model should be compatible with at least standard runtime
        let compatible = await registry.hasCompatibleRuntime(for: "mlx-community/Qwen2.5-7B-Instruct-4bit")
        XCTAssertTrue(compatible, "Standard models should always have compatible runtime")
    }
    
    // MARK: — Runtime Selection and Activation
    
    func testSelectRuntime_WithDFlashPreference() {
        let config = RuntimeConfig(
            temperature: 0.7,
            enableDFlash: true
        )
        
        let engine = registry.selectRuntime(
            for: "mlx-community/Qwen2.5-7B-Instruct-4bit",
            config: config
        )
        
        XCTAssertNotNil(engine)
        XCTAssertEqual(engine?.id, "mlx.dflash", "Should select DFlash runtime when requested")
    }
    
    func testSelectRuntime_WithSpeculativePreference() {
        let config = RuntimeConfig(
            temperature: 0.7,
            enableSpeculative: true
        )
        
        let engine = registry.selectRuntime(
            for: "mlx-community/Qwen2.5-7B-Instruct-4bit",
            config: config
        )
        
        XCTAssertNotNil(engine)
        XCTAssertEqual(engine?.id, "mlx.speculative", "Should select speculative runtime when requested")
    }
    
    func testSelectRuntime_DefaultsToStandard() {
        let config = RuntimeConfig(temperature: 0.7)
        
        let engine = registry.selectRuntime(
            for: "mlx-community/Qwen2.5-7B-Instruct-4bit",
            config: config
        )
        
        XCTAssertNotNil(engine)
        XCTAssertEqual(engine?.id, "mlx.standard", "Should default to standard runtime")
    }
    
    func testActiveEngineTracking() {
        XCTAssertNil(registry.activeEngine, "Should start with no active engine")
        
        let config = RuntimeConfig(temperature: 0.7)
        let engine = registry.selectRuntime(
            for: "mlx-community/Qwen2.5-7B-Instruct-4bit",
            config: config
        )
        
        XCTAssertNotNil(registry.activeEngine)
        XCTAssertTrue(registry.activeEngine === engine, "Active engine should match selected engine")
    }
    
    func testDeactivation() async {
        // First activate an engine
        let config = RuntimeConfig(temperature: 0.7)
        _ = registry.selectRuntime(
            for: "mlx-community/Qwen2.5-7B-Instruct-4bit",
            config: config
        )
        
        XCTAssertNotNil(registry.activeEngine, "Should have active engine")
        
        // Deactivate
        await registry.deactivate()
        
        XCTAssertNil(registry.activeEngine, "Active engine should be nil after deactivation")
    }
    
    func testRuntimeSwitching() {
        // Start with standard runtime
        let config1 = RuntimeConfig(temperature: 0.7)
        let engine1 = registry.selectRuntime(
            for: "mlx-community/Qwen2.5-7B-Instruct-4bit",
            config: config1
        )
        XCTAssertEqual(engine1?.id, "mlx.standard")
        
        // Switch to DFlash runtime
        let config2 = RuntimeConfig(temperature: 0.7, enableDFlash: true)
        let engine2 = registry.selectRuntime(
            for: "mlx-community/Qwen2.5-7B-Instruct-4bit",
            config: config2
        )
        XCTAssertEqual(engine2?.id, "mlx.dflash")
        
        // Active engine should be the new one
        XCTAssertTrue(registry.activeEngine === engine2)
    }
}
