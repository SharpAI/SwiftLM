import XCTest
import Foundation
@testable import SwiftLM
import MLXLMCommon

final class MultimodalTokenExtractionTests: XCTestCase {

    func testExtractMultimodalTokens_Defaults() throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }
        
        let config = ModelConfiguration(directory: tempDir).resolved(modelDirectory: tempDir, tokenizerDirectory: tempDir)
        
        let tokens = OmniModelFactory.extractMultimodalTokens(configuration: config)
        XCTAssertEqual(tokens.numAudio, 128)
        XCTAssertEqual(tokens.boa, 255010)
        XCTAssertEqual(tokens.eoa, 255011)
    }

    func testExtractMultimodalTokens_FromConfig() throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }
        
        let jsonDict: [String: Any] = [
            "subsampling_conv_channels": [256],
            "boa_token_id": 999990,
            "eoa_token_id": 999991
        ]
        
        let jsonData = try JSONSerialization.data(withJSONObject: jsonDict)
        let configURL = tempDir.appendingPathComponent("config.json")
        try jsonData.write(to: configURL)
        
        let config = ModelConfiguration(directory: tempDir).resolved(modelDirectory: tempDir, tokenizerDirectory: tempDir)
        let tokens = OmniModelFactory.extractMultimodalTokens(configuration: config)
        
        XCTAssertEqual(tokens.numAudio, 256)
        XCTAssertEqual(tokens.boa, 999990)
        XCTAssertEqual(tokens.eoa, 999991)
    }

    func testExtractMultimodalTokens_FromAudioConfigFallback() throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }
        
        let jsonDict: [String: Any] = [
            "audio_config": [
                "num_audio_embeddings": 512,
                "boa_token_id": 888880,
                "eoa_token_id": 888881
            ]
        ]
        
        let jsonData = try JSONSerialization.data(withJSONObject: jsonDict)
        let configURL = tempDir.appendingPathComponent("config.json")
        try jsonData.write(to: configURL)
        
        let config = ModelConfiguration(directory: tempDir).resolved(modelDirectory: tempDir, tokenizerDirectory: tempDir)
        let tokens = OmniModelFactory.extractMultimodalTokens(configuration: config)
        
        XCTAssertEqual(tokens.numAudio, 512)
        XCTAssertEqual(tokens.boa, 888880)
        XCTAssertEqual(tokens.eoa, 888881)
    }
}
