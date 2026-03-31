// ModelPickerView.swift — Device-aware model selection + download
import SwiftUI

struct ModelPickerView: View {
    @EnvironmentObject private var engine: InferenceEngine
    let onSelect: (String) -> Void

    @State private var device = DeviceProfile.current
    @State private var selectedModelId: String? = nil

    private var recommendedModels: [ModelEntry] {
        ModelCatalog.recommended(for: device)
    }
    private var otherModels: [ModelEntry] {
        ModelCatalog.all.filter { model in
            !recommendedModels.contains(where: { $0.id == model.id })
        }
    }

    var body: some View {
        NavigationStack {
            List {
                // Device info header
                Section {
                    HStack {
                        Label(String(format: "%.0f GB RAM", device.physicalRAMGB), systemImage: "memorychip")
                        Spacer()
                        Text("Apple Silicon").foregroundStyle(.secondary)
                    }
                    .font(.subheadline)
                }

                // Recommended models
                Section("Recommended for this device") {
                    ForEach(recommendedModels) { model in
                        ModelRow(model: model, fitStatus: ModelCatalog.fitStatus(for: model, on: device)) {
                            selectedModelId = model.id
                            onSelect(model.id)
                        }
                    }
                }

                // Larger models (need flash streaming)
                if !otherModels.isEmpty {
                    Section("Larger models (flash streaming required)") {
                        ForEach(otherModels) { model in
                            ModelRow(model: model, fitStatus: ModelCatalog.fitStatus(for: model, on: device)) {
                                selectedModelId = model.id
                                onSelect(model.id)
                            }
                        }
                    }
                }
            }
            .listStyle(.insetGrouped)
            .navigationTitle("Choose Model")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { onSelect("") }
                }
            }
        }
        #if os(macOS)
        .frame(minWidth: 480, minHeight: 560)
        #endif
    }
}

// MARK: — Model Row

struct ModelRow: View {
    let model: ModelEntry
    let fitStatus: ModelCatalog.FitStatus
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 12) {
                // Model icon
                ZStack {
                    RoundedRectangle(cornerRadius: 10)
                        .fill(iconBackground)
                        .frame(width: 44, height: 44)
                    Image(systemName: iconName)
                        .font(.title3)
                        .foregroundStyle(.white)
                }

                VStack(alignment: .leading, spacing: 3) {
                    HStack {
                        Text(model.displayName)
                            .font(.headline)
                            .foregroundStyle(.primary)
                        if let badge = model.badge {
                            Text(badge)
                                .font(.caption2)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Color.accentColor.opacity(0.1))
                                .clipShape(Capsule())
                        }
                    }
                    Text("\(model.parameterSize) · \(model.quantization) · ~\(String(format: "%.1f", model.ramRequiredGB))GB RAM")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Spacer()

                fitBadge
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private var fitBadge: some View {
        Group {
            switch fitStatus {
            case .fits:
                Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
            case .tight:
                Image(systemName: "exclamationmark.circle.fill").foregroundStyle(.orange)
            case .requiresFlash:
                Label("Flash", systemImage: "bolt.fill")
                    .font(.caption).foregroundStyle(.purple)
            case .tooLarge:
                Image(systemName: "xmark.circle.fill").foregroundStyle(.red)
            }
        }
    }

    private var iconBackground: LinearGradient {
        switch fitStatus {
        case .fits: return LinearGradient(colors: [.blue, .cyan], startPoint: .topLeading, endPoint: .bottomTrailing)
        case .tight: return LinearGradient(colors: [.orange, .yellow], startPoint: .topLeading, endPoint: .bottomTrailing)
        case .requiresFlash: return LinearGradient(colors: [.purple, .indigo], startPoint: .topLeading, endPoint: .bottomTrailing)
        case .tooLarge: return LinearGradient(colors: [.gray, .gray], startPoint: .topLeading, endPoint: .bottomTrailing)
        }
    }

    private var iconName: String {
        if model.isMoE { return "square.grid.3x3.fill" }
        switch model.parameterSize {
        case let s where s.contains("0.5"): return "hare.fill"
        case let s where s.contains("3"): return "bolt.fill"
        case let s where s.contains("7"): return "brain"
        default: return "sparkles"
        }
    }
}
