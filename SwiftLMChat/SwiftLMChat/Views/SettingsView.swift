// SettingsView.swift — Inference and appearance settings
import SwiftUI

struct SettingsView: View {
    @ObservedObject var viewModel: ChatViewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section("Generation") {
                    HStack {
                        Text("Temperature")
                        Spacer()
                        Text(String(format: "%.2f", viewModel.config.temperature))
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                    }
                    Slider(value: Binding(
                        get: { Double(viewModel.config.temperature) },
                        set: { viewModel.config.temperature = Float($0) }
                    ), in: 0...2, step: 0.05)

                    HStack {
                        Text("Max Tokens")
                        Spacer()
                        Text("\(viewModel.config.maxTokens)")
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                    }
                    Slider(value: Binding(
                        get: { Double(viewModel.config.maxTokens) },
                        set: { viewModel.config.maxTokens = Int($0) }
                    ), in: 128...8192, step: 128)

                    HStack {
                        Text("Top P")
                        Spacer()
                        Text(String(format: "%.2f", viewModel.config.topP))
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                    }
                    Slider(value: Binding(
                        get: { Double(viewModel.config.topP) },
                        set: { viewModel.config.topP = Float($0) }
                    ), in: 0...1, step: 0.05)
                }

                Section("Advanced") {
                    Toggle("Thinking Mode", isOn: Binding(
                        get: { viewModel.config.enableThinking },
                        set: { viewModel.config.enableThinking = $0 }
                    ))
                    Text("Enables step-by-step reasoning for models that support <think> blocks (e.g. Qwen3, DeepSeek-R1).")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Section("System Prompt") {
                    TextEditor(text: $viewModel.systemPrompt)
                        .frame(minHeight: 80)
                        .font(.callout)
                }

                Section {
                    Button("Reset to Defaults", role: .destructive) {
                        viewModel.config = .default
                        viewModel.systemPrompt = ""
                    }
                }

                Section("About") {
                    LabeledContent("SwiftLM Chat", value: "1.0")
                    LabeledContent("Engine", value: "MLX Swift")
                    LabeledContent("Backend", value: "Metal GPU")
                }
            }
            .navigationTitle("Settings")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        #if os(macOS)
        .frame(width: 420, height: 560)
        #endif
    }
}
