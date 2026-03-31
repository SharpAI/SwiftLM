// RootView.swift — Adaptive root layout: sidebar on Mac, stack on iOS
import SwiftUI

struct RootView: View {
    @EnvironmentObject private var engine: InferenceEngine
    @StateObject private var viewModel = ChatViewModel()
    @State private var showModelPicker = false
    @State private var showSettings = false

    var body: some View {
        Group {
            #if os(macOS)
            macOSLayout
            #else
            iOSLayout
            #endif
        }
        .sheet(isPresented: $showModelPicker) {
            ModelPickerView(onSelect: { modelId in
                showModelPicker = false
                Task { await engine.load(modelId: modelId) }
            })
            .environmentObject(engine)
        }
        .sheet(isPresented: $showSettings) {
            SettingsView(viewModel: viewModel)
        }
        .onReceive(NotificationCenter.default.publisher(for: .showModelPicker)) { _ in
            showModelPicker = true
        }
        .onAppear {
            viewModel.engine = engine
            // Auto-show model picker if no model loaded
            if case .idle = engine.state { showModelPicker = true }
        }
        .onChange(of: engine.state) { _, state in
            if case .idle = state { showModelPicker = true }
        }
    }

    #if os(macOS)
    private var macOSLayout: some View {
        NavigationSplitView {
            VStack(alignment: .leading, spacing: 0) {
                // Sidebar: model status + new chat
                modelStatusView
                Divider()
                List {
                    Label("New Chat", systemImage: "plus.bubble")
                        .onTapGesture { viewModel.newConversation() }
                }
                .listStyle(.sidebar)
            }
            .frame(minWidth: 200)
        } detail: {
            ChatView(viewModel: viewModel, showSettings: $showSettings, showModelPicker: $showModelPicker)
        }
        .navigationTitle("")
    }
    #endif

    private var iOSLayout: some View {
        ChatView(viewModel: viewModel, showSettings: $showSettings, showModelPicker: $showModelPicker)
            .environmentObject(engine)
    }

    private var modelStatusView: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Image(systemName: "cpu")
                    .foregroundStyle(.secondary)
                Text("SwiftLM")
                    .font(.headline)
                Spacer()
            }
            engineStateView
        }
        .padding()
    }

    @ViewBuilder
    private var engineStateView: some View {
        switch engine.state {
        case .idle:
            Button("Load Model") { showModelPicker = true }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
        case .loading:
            Label("Loading…", systemImage: "arrow.2.circlepath")
                .font(.caption).foregroundStyle(.secondary)
        case .downloading(let progress, let speed):
            VStack(alignment: .leading, spacing: 2) {
                ProgressView(value: progress)
                Text("\(Int(progress * 100))% · \(speed)")
                    .font(.caption2).foregroundStyle(.secondary)
            }
        case .ready(let modelId):
            Label(modelId.components(separatedBy: "/").last ?? modelId, systemImage: "checkmark.circle.fill")
                .font(.caption).foregroundStyle(.green)
                .lineLimit(1)
        case .generating:
            Label("Generating…", systemImage: "ellipsis.bubble")
                .font(.caption).foregroundStyle(.blue)
        case .error(let msg):
            Label(msg, systemImage: "exclamationmark.triangle")
                .font(.caption).foregroundStyle(.red)
                .lineLimit(2)
        }
    }
}
