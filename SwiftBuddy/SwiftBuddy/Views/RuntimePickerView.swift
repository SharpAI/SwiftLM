// RuntimePickerView.swift — Runtime engine selection UI for SwiftBuddy
import SwiftUI
import MLXInferenceCore

struct RuntimePickerView: View {
    @StateObject private var runtimeService = RuntimeService()
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // 1. Runtime Engine Selector
            runtimeSection
            
            // 2. Preset Selector (for selected runtime)
            if !runtimeService.availablePresets.isEmpty {
                presetSection
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity)
        .background(SwiftBuddyTheme.background.opacity(0.3))
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .strokeBorder(Color.white.opacity(0.07), lineWidth: 1)
        )
    }
    
    // MARK: — Runtime Section
    
    @ViewBuilder
    private var runtimeSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Image(systemName: "cpu.fill")
                    .foregroundStyle(SwiftBuddyTheme.accent)
                Text("Runtime Engine")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(SwiftBuddyTheme.textPrimary)
                Spacer()
            }
            
            // Runtime picker
            Picker("", selection: $runtimeService.selectedRuntimeId) {
                ForEach(runtimeService.availableRuntimes, id: \.id) { capability in
                    HStack {
                        Text(capability.displayName)
                        Spacer()
                        memoryBadge(for: capability.memoryEfficiency)
                    }
                    .tag(capability.id)
                }
            }
            .pickerStyle(.menu)
            .frame(maxWidth: .infinity)
            .padding(10)
            .background(SwiftBuddyTheme.background.opacity(0.6))
            .clipShape(RoundedRectangle(cornerRadius: 8))
            
            // Runtime description
            if let active = runtimeService.activeCapability {
                Text(runtimeDescription(for: active))
                    .font(.caption)
                    .foregroundStyle(SwiftBuddyTheme.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
    
    // MARK: — Preset Section
    
    @ViewBuilder
    private var presetSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Image(systemName: "slider.horizontal.3")
                    .foregroundStyle(SwiftBuddyTheme.warning)
                Text("Configuration Preset")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(SwiftBuddyTheme.textPrimary)
                Spacer()
            }
            
            // Preset picker
            Picker("", selection: $runtimeService.selectedPresetId) {
                ForEach(runtimeService.availablePresets) { preset in
                    HStack {
                        Image(systemName: preset.icon)
                        Text(preset.name)
                        Spacer()
                    }
                    .tag(preset.id as String?)
                }
            }
            .pickerStyle(.menu)
            .frame(maxWidth: .infinity)
            .padding(10)
            .background(SwiftBuddyTheme.background.opacity(0.6))
            .clipShape(RoundedRectangle(cornerRadius: 8))
            
            // Preset details
            if let preset = runtimeService.currentPreset {
                VStack(alignment: .leading, spacing: 8) {
                    // Description
                    Text(preset.description)
                        .font(.caption)
                        .foregroundStyle(SwiftBuddyTheme.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                    
                    // Configuration badges
                    configBadges(for: preset.config)
                }
            }
        }
    }
    
    // MARK: — Badge Views
    
    @ViewBuilder
    private func configBadges(for config: RuntimeConfig) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                // Temperature
                paramBadge("Temp", value: String(format: "%.1f", config.temperature), color: .red)
                // Top-P
                paramBadge("Top-P", value: String(format: "%.2f", config.topP), color: .blue)
                // Repetition Penalty
                if let repPenalty = config.repetitionPenalty {
                    paramBadge("Rep", value: String(format: "%.2f", repPenalty), color: .purple)
                }
            }
            
            HStack(spacing: 8) {
                // Context window
                if let contextWindow = config.contextWindow {
                    paramBadge("Context", value: formatTokens(contextWindow), color: .green)
                }
                // Max tokens
                paramBadge("Max", value: formatTokens(config.maxTokens), color: .orange)
            }
        }
    }
    
    @ViewBuilder
    private func paramBadge(_ label: String, value: String, color: Color) -> some View {
        HStack(spacing: 4) {
            Text(label)
                .font(.caption2.weight(.medium))
            Text(value)
                .font(.caption2.weight(.bold))
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(color.opacity(0.15))
        .foregroundStyle(color)
        .clipShape(Capsule())
    }
    
    private func formatTokens(_ count: Int) -> String {
        if count >= 1000 {
            return "\(count / 1000)K"
        }
        return "\(count)"
    }
    
    private func memoryBadge(for efficiency: RuntimeCapability.MemoryEfficiency) -> some View {
        let (color, icon): (Color, String) = {
            switch efficiency {
            case .standard:  return (SwiftBuddyTheme.textTertiary, "memorychip")
            case .optimized: return (SwiftBuddyTheme.warning, "memorychip.fill")
            case .extreme:   return (SwiftBuddyTheme.success, "bolt.fill")
            }
        }()
        
        return Label(efficiency.rawValue.capitalized, systemImage: icon)
            .font(.caption2.weight(.medium))
            .foregroundStyle(color)
    }
    
    private func runtimeDescription(for capability: RuntimeCapability) -> String {
        switch capability.id {
        case "mlx.standard":
            return "Standard MLX inference. Balanced performance for most models. Native support for vision-language (VLM) and audio-language (ALM) models. Best for: Gemma, Qwen, Llama, Phi, Mistral families."
        case "mlx.dflash":
            return "DFlash memory-optimized attention. Reduces RAM usage for large context windows with efficient attention computation. Best for: long-context scenarios (40K-100K tokens). Note: May exhibit repetition on some models due to greedy decoding."
        case "mlx.speculative":
            return "Speculative decoding for speed. Load a small draft model (e.g., 9B) alongside main model to generate candidate tokens and verify in bulk. Accelerates in-RAM inference. Best for: speed-critical applications with draft+target model pairs."
        case "mlx.streaming_moe":
            return "SSD Expert Streaming (10x faster). Streams Mixture-of-Experts layers directly from NVMe SSD to GPU. Enables 122B-397B MoE models on 24GB machines with ~16-22GB RAM. Uses concurrent pread, cross-projection batching. Best for: Qwen3.5-122B/397B, DeepSeek-V3/V4, Mixtral-8x22B."
        default:
            return capability.displayName
        }
    }
}

#Preview {
    ZStack {
        SwiftBuddyTheme.background.ignoresSafeArea()
        RuntimePickerView()
            .padding()
    }
}
