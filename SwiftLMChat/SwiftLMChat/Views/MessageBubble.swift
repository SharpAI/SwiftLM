// MessageBubble.swift — Chat message bubble + live streaming bubble
import SwiftUI

// MARK: — Static Message Bubble

struct MessageBubble: View {
    let message: ChatMessage

    var isUser: Bool { message.role == .user }

    var body: some View {
        HStack(alignment: .bottom, spacing: 8) {
            if isUser { Spacer(minLength: 60) }

            if !isUser {
                // Avatar
                Circle()
                    .fill(LinearGradient(colors: [.blue, .purple], startPoint: .topLeading, endPoint: .bottomTrailing))
                    .frame(width: 28, height: 28)
                    .overlay(
                        Image(systemName: "cpu")
                            .font(.caption2)
                            .foregroundStyle(.white)
                    )
            }

            VStack(alignment: isUser ? .trailing : .leading, spacing: 4) {
                Text(message.content)
                    .textSelection(.enabled)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(isUser ? Color.accentColor : Color(white: 0.92))
                    .foregroundStyle(isUser ? .white : .primary)
                    .clipShape(BubbleShape(isUser: isUser))

                Text(message.timestamp, style: .time)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            if !isUser { Spacer(minLength: 60) }
        }
    }
}

// MARK: — Live Streaming Bubble

struct StreamingBubble: View {
    let text: String
    let thinkingText: String?
    @State private var dotPhase = 0

    var body: some View {
        HStack(alignment: .bottom, spacing: 8) {
            Circle()
                .fill(LinearGradient(colors: [.blue, .purple], startPoint: .topLeading, endPoint: .bottomTrailing))
                .frame(width: 28, height: 28)
                .overlay(
                    Image(systemName: "cpu")
                        .font(.caption2)
                        .foregroundStyle(.white)
                )

            VStack(alignment: .leading, spacing: 6) {
                // Thinking section (collapsed by default, shows thought process)
                if let thinking = thinkingText, !thinking.isEmpty {
                    DisclosureGroup {
                        Text(thinking)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .padding(8)
                            .background(Color(white: 0.95))
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                    } label: {
                        Label("Thinking…", systemImage: "brain")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                // Response text
                if !text.isEmpty {
                    HStack(alignment: .bottom, spacing: 2) {
                        Text(text)
                            .textSelection(.enabled)
                        // Blinking cursor
                        RoundedRectangle(cornerRadius: 1)
                            .frame(width: 2, height: 16)
                            .foregroundStyle(.blue)
                            .opacity(dotPhase % 2 == 0 ? 1 : 0)
                            .animation(.easeInOut(duration: 0.5).repeatForever(), value: dotPhase)
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color(white: 0.92))
                    .clipShape(BubbleShape(isUser: false))
                } else {
                    // Typing dots when no text yet
                    TypingIndicator()
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(Color(white: 0.92))
                        .clipShape(BubbleShape(isUser: false))
                }
            }

            Spacer(minLength: 60)
        }
        .onAppear { dotPhase = 1 }
    }
}

// MARK: — Typing Indicator

struct TypingIndicator: View {
    @State private var phase = 0

    var body: some View {
        HStack(spacing: 4) {
            ForEach(0..<3) { i in
                Circle()
                    .frame(width: 6, height: 6)
                    .foregroundStyle(.secondary)
                    .scaleEffect(phase == i ? 1.3 : 0.8)
                    .animation(
                        .easeInOut(duration: 0.4).repeatForever().delay(Double(i) * 0.15),
                        value: phase
                    )
            }
        }
        .onAppear {
            withAnimation { phase = 1 }
        }
    }
}

// MARK: — Bubble Shape

struct BubbleShape: Shape {
    let isUser: Bool
    let radius: CGFloat = 16

    func path(in rect: CGRect) -> Path {
        var path = Path()
        let tl = isUser ? radius : 4
        let tr = isUser ? 4 : radius
        let bl = radius
        let br = radius

        path.move(to: CGPoint(x: rect.minX + tl, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.maxX - tr, y: rect.minY))
        path.addQuadCurve(to: CGPoint(x: rect.maxX, y: rect.minY + tr),
                          control: CGPoint(x: rect.maxX, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY - br))
        path.addQuadCurve(to: CGPoint(x: rect.maxX - br, y: rect.maxY),
                          control: CGPoint(x: rect.maxX, y: rect.maxY))
        path.addLine(to: CGPoint(x: rect.minX + bl, y: rect.maxY))
        path.addQuadCurve(to: CGPoint(x: rect.minX, y: rect.maxY - bl),
                          control: CGPoint(x: rect.minX, y: rect.maxY))
        path.addLine(to: CGPoint(x: rect.minX, y: rect.minY + tl))
        path.addQuadCurve(to: CGPoint(x: rect.minX + tl, y: rect.minY),
                          control: CGPoint(x: rect.minX, y: rect.minY))
        path.closeSubpath()
        return path
    }
}
