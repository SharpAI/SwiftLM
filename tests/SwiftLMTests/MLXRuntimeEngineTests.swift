// MLXRuntimeEngineTests.swift — Unit tests for MLXRuntimeEngine adapter
//
// Tests the adapter that wraps InferenceEngine with RuntimeEngine protocol.
// Note: These are mostly structural tests since full inference testing
// requires model files.

import XCTest
import Foundation
@testable import MLXInferenceCore

@MainActor
final class MLXRuntimeEngineTests: XCTestCase {
    
    // MARK: — Initialization
    
    func testInit_StandardMode() {
        let engine = MLXRuntimeEngine(mode: .standard)
        XCTAssertEqual(engine.id, "mlx.standard")
        XCTAssertEqual(engine.displayName, "MLX Standard")
        XCTAssertEqual(engine.mode, .standard)
    }
    
    func testInit_DFlashMode() {
        let engine = MLXRuntimeEngine(mode: .dflash)
        XCTAssertEqual(engine.id, "mlx.dflash")
        XCTAssertEqual(engine.displayName, "MLX + DFlash")
        XCTAssertEqual(engine.mode, .dflash)
    }
    
    func testInit_SpeculativeMode() {
        let engine = MLXRuntimeEngine(mode: .speculative)
        XCTAssertEqual(engine.id, "mlx.speculative")
        XCTAssertEqual(engine.displayName, "MLX Speculative")
        XCTAssertEqual(engine.mode, .speculative)
    }
    
    func testInit_StreamingMoEMode() {
        let engine = MLXRuntimeEngine(mode: .streamingMoE)
        XCTAssertEqual(engine.id, "mlx.streaming_moe")
        XCTAssertEqual(engine.displayName, "MLX Streaming MoE")
        XCTAssertEqual(engine.mode, .streamingMoE)
    }
    
    // MARK: — Capabilities
    
    func testCapabilities_StandardMode() {
        let engine = MLXRuntimeEngine(mode: .standard)
        let capabilities = engine.capabilities
        
        XCTAssertTrue(capabilities.supportsStreaming)
        XCTAssertTrue(capabilities.supportsVision)
        XCTAssertTrue(capabilities.supportsAudio)
        XCTAssertTrue(capabilities.supportsToolCalling)
        XCTAssertFalse(capabilities.supportsDFlash)
        XCTAssertFalse(capabilities.supportsSpeculative)
        XCTAssertEqual(capabilities.memoryEfficiency, .standard)
    }
    
    func testCapabilities_DFlashMode() {
        let engine = MLXRuntimeEngine(mode: .dflash)
        let capabilities = engine.capabilities
        
        XCTAssertTrue(capabilities.supportsDFlash)
        XCTAssertTrue(capabilities.supportsVision)
        XCTAssertTrue(capabilities.supportsAudio)
        XCTAssertEqual(capabilities.memoryEfficiency, .optimized)
    }
    
    func testCapabilities_SpeculativeMode() {
        let engine = MLXRuntimeEngine(mode: .speculative)
        let capabilities = engine.capabilities
        
        XCTAssertTrue(capabilities.supportsSpeculative)
        XCTAssertTrue(capabilities.supportsVision)
        XCTAssertTrue(capabilities.supportsAudio)
        XCTAssertEqual(capabilities.memoryEfficiency, .standard)
    }
    
    func testCapabilities_StreamingMoEMode() {
        let engine = MLXRuntimeEngine(mode: .streamingMoE)
        let capabilities = engine.capabilities
        
        XCTAssertTrue(capabilities.supportsDFlash)
        XCTAssertFalse(capabilities.supportsVision)
        XCTAssertFalse(capabilities.supportsAudio)
        XCTAssertEqual(capabilities.memoryEfficiency, .extreme)
    }
    
    // MARK: — State Management
    
    func testInitialState() {
        let engine = MLXRuntimeEngine(mode: .standard)
        XCTAssertEqual(engine.state, .idle)
        XCTAssertNil(engine.loadedModelId)
        XCTAssertEqual(engine.activeContextTokens, 0)
        XCTAssertEqual(engine.maxContextWindow, 0)
    }
    
    func testInferenceEngineAccess() {
        let engine = MLXRuntimeEngine(mode: .standard)
        // Should expose inferenceEngine for legacy bridge
        XCTAssertNotNil(engine.inferenceEngine)
    }
    
    // MARK: — Compatibility Checks
    
    func testCompatibility_StandardModeAcceptsAllModels() async {
        let engine = MLXRuntimeEngine(mode: .standard)
        
        let compatible1 = await engine.isCompatible(modelId: "mlx-community/Qwen2.5-7B-Instruct-4bit")
        XCTAssertTrue(compatible1)
        
        let compatible2 = await engine.isCompatible(modelId: "mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit")
        XCTAssertTrue(compatible2)
        
        let compatible3 = await engine.isCompatible(modelId: "mlx-community/pixtral-12b-4bit")
        XCTAssertTrue(compatible3)
    }
    
    func testCompatibility_StreamingMoERejectsMoEModels() async {
        let engine = MLXRuntimeEngine(mode: .streamingMoE)
        
        // Should accept MoE models
        let compatible1 = await engine.isCompatible(modelId: "mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit")
        XCTAssertTrue(compatible1, "Streaming MoE should accept MoE models")
        
        // Should reject vision models
        let compatible2 = await engine.isCompatible(modelId: "mlx-community/pixtral-12b-4bit")
        XCTAssertFalse(compatible2, "Streaming MoE should reject vision models")
    }
    
    func testCompatibility_HandlesUnknownModels() async {
        let engine = MLXRuntimeEngine(mode: .standard)
        
        // Unknown models should be accepted (conservative approach)
        let compatible = await engine.isCompatible(modelId: "unknown/model-id")
        XCTAssertTrue(compatible, "Unknown models should be accepted by standard runtime")
    }
    
    // MARK: — RuntimeConfig Mapping
    
    func testRuntimeConfig_MapsToGenerationConfig() async {
        let engine = MLXRuntimeEngine(mode: .standard)
        let config = RuntimeConfig(
            temperature: 0.5,
            topP: 0.95,
            topK: 50,
            minP: 0.1,
            repetitionPenalty: 1.2,
            maxTokens: 4096,
            enableDFlash: true,
            streamExperts: true
        )
        
        // Load method should accept RuntimeConfig
        // (We can't test actual loading without model files, but we can test the signature)
        do {
            try await engine.load(modelId: "mlx-community/Qwen2.5-7B-Instruct-4bit", config: config)
        } catch {
            // Expected to fail without actual model files
            // Just verifying the config parameter is accepted
        }
    }
    
    // MARK: — Multiple Instances
    
    func testMultipleInstances_AreSeparate() {
        let engine1 = MLXRuntimeEngine(mode: .standard)
        let engine2 = MLXRuntimeEngine(mode: .dflash)
        
        XCTAssertFalse(engine1 === engine2, "Different instances should not be identical")
        XCTAssertNotEqual(engine1.id, engine2.id)
        XCTAssertNotEqual(engine1.mode, engine2.mode)
    }
    
    func testMultipleInstances_SameMode() {
        let engine1 = MLXRuntimeEngine(mode: .standard)
        let engine2 = MLXRuntimeEngine(mode: .standard)
        
        // Different instances, even with same mode
        XCTAssertFalse(engine1 === engine2)
        XCTAssertEqual(engine1.id, engine2.id)
        XCTAssertEqual(engine1.mode, engine2.mode)
    }
    
    // MARK: — StopGeneration
    
    func testStopGeneration_DoesNotCrash() async {
        let engine = MLXRuntimeEngine(mode: .standard)
        // Should be safe to call even when not generating
        await engine.stopGeneration()
    }
    
    func testUnload_DoesNotCrash() async {
        let engine = MLXRuntimeEngine(mode: .standard)
        // Should be safe to call even when nothing is loaded
        await engine.unload()
        XCTAssertEqual(engine.state, .idle)
    }
}
