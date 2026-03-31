// SwiftLMChatApp.swift — App entry point (iOS + macOS)
import SwiftUI

@main
struct SwiftLMChatApp: App {
    @StateObject private var engine = InferenceEngine()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(engine)
        }
        #if os(macOS)
        .commands {
            CommandGroup(replacing: .newItem) {}
            CommandMenu("Model") {
                Button("Choose Model…") {
                    NotificationCenter.default.post(name: .showModelPicker, object: nil)
                }.keyboardShortcut("m", modifiers: [.command, .shift])
                Button("Unload Model") {
                    engine.unload()
                }
            }
        }
        #endif
    }
}

extension Notification.Name {
    static let showModelPicker = Notification.Name("showModelPicker")
}
