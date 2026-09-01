// RuntimeService.swift — SwiftBuddy bridge to RuntimeRegistry
import SwiftUI
import Combine
import MLXInferenceCore

/// SwiftBuddy's bridge to the RuntimeRegistry system.
/// Provides UI-friendly access to runtime selection and monitoring.
@MainActor
final class RuntimeService: ObservableObject {
    
    // MARK: — Published State
    
    @Published var registry = RuntimeRegistry.shared
    @Published var selectedRuntimeId: String {
        didSet {
            if selectedRuntimeId != oldValue {
                savePreferredRuntime()
                // Reset to default preset when runtime changes
                selectedPresetId = RuntimePresetLibrary.presets(for: selectedRuntimeId).first?.id
            }
        }
    }
    @Published var selectedPresetId: String? {
        didSet {
            if selectedPresetId != oldValue {
                savePreferredPreset()
            }
        }
    }
    @Published var isReady: Bool = false
    @Published var currentModelId: String?
    
    // MARK: — Convenience
    
    var activeEngine: RuntimeEngine? {
        registry.activeEngine
    }
    
    var availablePresets: [RuntimePreset] {
        RuntimePresetLibrary.presets(for: selectedRuntimeId)
    }
    
    var currentPreset: RuntimePreset? {
        guard let id = selectedPresetId else { return nil }
        return RuntimePresetLibrary.preset(id: id)
    }
    
    var currentConfig: RuntimeConfig {
        currentPreset?.config ?? RuntimeConfig()
    }
    
    /// Get model-aware preset recommendations based on detected hardware.
    /// TODO: Re-enable when SystemInfo and ModelProfileRegistry are implemented
    var recommendedPresetsForCurrentModel: [RuntimePreset] {
        // For now, just return all available presets for the selected runtime
        return availablePresets
    }
    
    /// Get model profile if available for current model.
    /// TODO: Re-enable when ModelProfileRegistry is implemented
    var currentModelProfile: String? {
        return currentModelId
    }
    
    var availableRuntimes: [RuntimeCapability] {
        registry.availableRuntimes
    }
    
    var activeCapability: RuntimeCapability? {
        guard let engine = registry.activeEngine else { return nil }
        return registry.capability(for: engine.id)
    }
    
    // MARK: — Initialization
    
    init() {
        // Load preferred runtime (default: mlx.standard)
        let savedRuntime = UserDefaults.standard.string(forKey: "swiftbuddy.preferredRuntime")
        self.selectedRuntimeId = savedRuntime ?? "mlx.standard"
        
        // Load preferred preset
        self.selectedPresetId = UserDefaults.standard.string(forKey: "swiftbuddy.preferredPreset")
    }
    
    // MARK: — Model Loading
    
    /// Load a model with the currently selected runtime.
    /// Returns true if successful, false otherwise.
    @discardableResult
    func loadModel(
        _ modelId: String,
        config: RuntimeConfig? = nil
    ) async -> Bool {
        isReady = false
        currentModelId = modelId
        
        // Auto-select best preset based on model (if not already configured)
        // TODO: Re-enable intelligent preset selection when ModelProfileRegistry is implemented
        if config == nil, selectedPresetId == nil {
            if let firstPreset = availablePresets.first {
                selectedPresetId = firstPreset.id
                print("[RuntimeService] Auto-selected preset: \(firstPreset.name)")
            }
        }
        
        let finalConfig = config ?? currentConfig
        
        // Get or create the selected runtime
        guard let engine = registry.engine(for: selectedRuntimeId) else {
            print("[RuntimeService] Failed to get runtime: \(selectedRuntimeId)")
            return false
        }
        
        // Activate it (unloads any other active runtime)
        _ = registry.selectRuntime(for: modelId, config: finalConfig)
        
        // Load the model
        do {
            try await engine.load(modelId: modelId, config: finalConfig)
            isReady = (engine.state == .ready(modelId: modelId))
            
            // Success - log the configuration
            if isReady {
                print("[RuntimeService] ✓ \(modelId) loaded successfully")
                print("[RuntimeService]   Context: \(finalConfig.contextWindow ?? 8192) tokens")
                print("[RuntimeService]   Preset: \(currentPreset?.name ?? "Custom")")
            }
            
            return isReady
        } catch {
            print("[RuntimeService] Failed to load \(modelId): \(error)")
            return false
        }
    }
    
    /// Unload the active model and runtime.
    func unloadModel() async {
        registry.deactivate()
        isReady = false
        currentModelId = nil
    }
    
    // MARK: — Runtime Selection
    
    /// Check if a model is compatible with the selected runtime.
    func isCompatible(modelId: String) async -> Bool {
        guard let engine = registry.engine(for: selectedRuntimeId) else {
            return false
        }
        return await engine.isCompatible(modelId: modelId)
    }
    
    /// Get the best runtime for a given model and config.
    /// Returns the runtime ID, or nil if none are compatible.
    func recommendRuntime(for modelId: String, config: RuntimeConfig) -> String? {
        return RuntimeSelector.selectRuntime(
            modelId: modelId,
            preferredConfig: config,
            availableRuntimes: registry.availableRuntimes
        )
    }
    
    // MARK: — Persistence
    
    private func savePreferredRuntime() {
        UserDefaults.standard.set(selectedRuntimeId, forKey: "swiftbuddy.preferredRuntime")
    }
    
    private func savePreferredPreset() {
        if let presetId = selectedPresetId {
            UserDefaults.standard.set(presetId, forKey: "swiftbuddy.preferredPreset")
        }
    }
    
    // MARK: — Legacy InferenceEngine Bridge
    
    /// For gradual migration: get the underlying InferenceEngine from the active MLXRuntimeEngine.
    /// This allows SwiftBuddy to continue using InferenceEngine directly while we transition.
    var legacyEngine: InferenceEngine? {
        guard let mlxEngine = registry.activeEngine as? MLXRuntimeEngine else {
            return nil
        }
        return mlxEngine.inferenceEngine
    }
}
