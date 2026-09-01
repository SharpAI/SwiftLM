// OnboardingMessageView.swift — Chat-style onboarding message UI
import SwiftUI

/// Displays onboarding messages as system messages in the chat interface.
struct OnboardingMessageView: View {
    let message: OnboardingMessage
    let onAction: (OnboardingAction) -> Void
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Message text with markdown support
            Text(.init(message.text))
                .font(.body)
                .foregroundStyle(SwiftBuddyTheme.textPrimary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
            
            // Action buttons
            if !message.actions.isEmpty {
                VStack(spacing: 12) {
                    ForEach(message.actions) { action in
                        actionButton(for: action)
                    }
                }
                .padding(.top, 8)
            }
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(SwiftBuddyTheme.accent.opacity(0.08))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .strokeBorder(SwiftBuddyTheme.accent.opacity(0.2), lineWidth: 1)
        )
    }
    
    @ViewBuilder
    private func actionButton(for action: OnboardingAction) -> some View {
        Button {
            onAction(action)
        } label: {
            HStack(spacing: 12) {
                Image(systemName: action.icon)
                    .font(.body.weight(.medium))
                
                Text(action.title)
                    .font(.body.weight(.medium))
                
                Spacer()
                
                Image(systemName: "chevron.right")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 14)
            .frame(maxWidth: .infinity)
            .background(buttonBackground(for: action))
            .foregroundStyle(buttonForeground(for: action))
            .clipShape(RoundedRectangle(cornerRadius: 10))
            .overlay(
                RoundedRectangle(cornerRadius: 10)
                    .strokeBorder(buttonBorder(for: action), lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
    }
    
    private func buttonBackground(for action: OnboardingAction) -> Color {
        switch action {
        case .download:
            return SwiftBuddyTheme.accent.opacity(0.15)
        case .skip:
            return SwiftBuddyTheme.background.opacity(0.5)
        default:
            return SwiftBuddyTheme.background.opacity(0.3)
        }
    }
    
    private func buttonForeground(for action: OnboardingAction) -> Color {
        switch action {
        case .download:
            return SwiftBuddyTheme.accent
        default:
            return SwiftBuddyTheme.textPrimary
        }
    }
    
    private func buttonBorder(for action: OnboardingAction) -> Color {
        switch action {
        case .download:
            return SwiftBuddyTheme.accent.opacity(0.3)
        default:
            return Color.white.opacity(0.1)
        }
    }
}

// MARK: — Preview

#Preview {
    ZStack {
        SwiftBuddyTheme.background.ignoresSafeArea()
        
        ScrollView {
            VStack(spacing: 20) {
                OnboardingMessageView(
                    message: OnboardingMessage(
                        text: """
                        👋 **Welcome to SwiftBuddy!**
                        
                        I'm your native Swift AI assistant running on Apple Silicon with MLX inference.
                        
                        To get started, I recommend downloading **Qwen2.5 7B Instruct** (4.5 GB):
                        
                        ✨ **Why this model?**
                        • Excellent instruction-following capabilities
                        • Optimized for Apple Silicon with 4-bit quantization
                        • Fast inference on M-series chips
                        • Great for coding, writing, and general assistance
                        
                        Would you like me to download and set it up for you?
                        """,
                        actions: [
                            .download(modelId: "mlx-community/Qwen2.5-7B-Instruct-4bit", displayName: "Qwen2.5 7B"),
                            .skip,
                            .browse
                        ]
                    ),
                    onAction: { action in
                        print("Action: \(action)")
                    }
                )
                
                OnboardingMessageView(
                    message: OnboardingMessage(
                        text: """
                        ✅ **Qwen2.5 7B is ready!**
                        
                        Your model has been downloaded and configured with the **Balanced** preset.
                        
                        🎉 **You're all set!** Try asking me anything.
                        """,
                        actions: [
                            .openSettings,
                            .startChatting
                        ]
                    ),
                    onAction: { action in
                        print("Action: \(action)")
                    }
                )
            }
            .padding()
        }
    }
}
