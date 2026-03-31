// ChatView.swift — Main chat interface (iOS + macOS)
import SwiftUI

struct ChatView: View {
    @ObservedObject var viewModel: ChatViewModel
    @Binding var showSettings: Bool
    @Binding var showModelPicker: Bool
    @EnvironmentObject private var engine: InferenceEngine

    @State private var inputText = ""
    @State private var scrollProxy: ScrollViewProxy? = nil
    @FocusState private var inputFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            // Message list
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        ForEach(viewModel.messages) { message in
                            MessageBubble(message: message)
                                .id(message.id)
                        }
                        // Live streaming bubble
                        if !viewModel.streamingText.isEmpty || viewModel.thinkingText != nil {
                            StreamingBubble(
                                text: viewModel.streamingText,
                                thinkingText: viewModel.thinkingText
                            )
                            .id("streaming")
                        }
                        // Spacer anchor for auto-scroll
                        Color.clear.frame(height: 1).id("bottom")
                    }
                    .padding()
                }
                .onChange(of: viewModel.streamingText) { _, _ in
                    withAnimation(.easeOut(duration: 0.1)) {
                        proxy.scrollTo("bottom")
                    }
                }
                .onAppear { scrollProxy = proxy }
            }

            Divider()

            // Input bar
            inputBar
        }
        .navigationTitle("SwiftLM Chat")
        #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar { iOSToolbar }
        #else
        .toolbar { macOSToolbar }
        #endif
    }

    // MARK: — Input Bar

    private var inputBar: some View {
        HStack(alignment: .bottom, spacing: 8) {
            TextField("Message", text: $inputText, axis: .vertical)
                .textFieldStyle(.plain)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color(.systemGray).opacity(0.1))
                .clipShape(RoundedRectangle(cornerRadius: 20))
                .lineLimit(1...8)
                .focused($inputFocused)
                .onSubmit {
                    #if os(macOS)
                    sendMessage()
                    #endif
                }

            if viewModel.isGenerating {
                Button(action: viewModel.stopGeneration) {
                    Image(systemName: "stop.circle.fill")
                        .font(.title2)
                        .foregroundStyle(.red)
                }
                .buttonStyle(.plain)
            } else {
                Button(action: sendMessage) {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.title2)
                        .foregroundStyle(inputText.trimmingCharacters(in: .whitespaces).isEmpty ? .gray : .blue)
                }
                .buttonStyle(.plain)
                .disabled(inputText.trimmingCharacters(in: .whitespaces).isEmpty)
                .keyboardShortcut(.return, modifiers: .command)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.regularMaterial)
    }

    private func sendMessage() {
        let text = inputText.trimmingCharacters(in: .whitespaces)
        guard !text.isEmpty, !viewModel.isGenerating else { return }
        inputText = ""
        Task { await viewModel.send(text) }
    }

    // MARK: — Toolbars

    #if os(iOS)
    @ToolbarContentBuilder
    private var iOSToolbar: some ToolbarContent {
        ToolbarItem(placement: .topBarLeading) {
            Button { showModelPicker = true } label: {
                Label("Model", systemImage: "cpu")
            }
        }
        ToolbarItem(placement: .topBarTrailing) {
            Button { viewModel.newConversation() } label: {
                Label("New Chat", systemImage: "square.and.pencil")
            }
        }
        ToolbarItem(placement: .topBarTrailing) {
            Button { showSettings = true } label: {
                Label("Settings", systemImage: "slider.horizontal.3")
            }
        }
    }
    #endif

    #if os(macOS)
    @ToolbarContentBuilder
    private var macOSToolbar: some ToolbarContent {
        ToolbarItem {
            Button { viewModel.newConversation() } label: {
                Label("New Chat", systemImage: "square.and.pencil")
            }
        }
        ToolbarItem {
            Button { showSettings = true } label: {
                Label("Settings", systemImage: "slider.horizontal.3")
            }
        }
    }
    #endif
}
