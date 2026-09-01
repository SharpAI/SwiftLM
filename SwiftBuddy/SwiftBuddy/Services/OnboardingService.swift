// OnboardingService.swift — Conversational onboarding for first-time users
import SwiftUI
import Combine

/// Manages the conversational onboarding experience for new users.
@MainActor
final class OnboardingService: ObservableObject {
    
    // MARK: — Published State
    
    @Published var isFirstLaunch: Bool = false
    @Published var onboardingStep: OnboardingStep = .welcome
    @Published var recommendedModel: ModelRecommendation?
    
    // MARK: — Onboarding Steps
    
    enum OnboardingStep: Equatable {
        case welcome
        case modelDownload(modelId: String)
        case runtimeSetup
        case completed
    }
    
    // MARK: — Initialization
    
    init() {
        checkFirstLaunch()
        if isFirstLaunch {
            setupRecommendedModel()
        }
    }
    
    // MARK: — First Launch Detection
    
    private func checkFirstLaunch() {
        let hasLaunchedKey = "swiftbuddy.hasLaunched"
        isFirstLaunch = !UserDefaults.standard.bool(forKey: hasLaunchedKey)
        
        if isFirstLaunch {
            UserDefaults.standard.set(true, forKey: hasLaunchedKey)
        }
    }
    
    // MARK: — Model Recommendation
    
    private func setupRecommendedModel() {
        // Recommend Gemma 4 26B MoE as the bootstrap model
        recommendedModel = ModelRecommendation(
            modelId: "mlx-community/gemma-4-26b-moe-4bit",
            displayName: "Gemma 4 26B MoE",
            size: "16 GB",
            runtime: "mlx.streaming_moe",
            preset: "gemma4.moe.balanced",
            description: "Blazing fast 61.5 tok/s on Apple Silicon with MTP acceleration. Mixture-of-Experts architecture with 4-layer assistant model for excellent quality."
        )
    }
    
    // MARK: — Onboarding Messages
    
    /// Generate welcome message for first-time users.
    func getWelcomeMessage() -> OnboardingMessage {
        guard let model = recommendedModel else {
            return OnboardingMessage(
                text: "👋 Welcome to SwiftBuddy!\n\nI'm your native Swift AI assistant powered by MLX. Let's get started by downloading a model.",
                actions: []
            )
        }
        
        return OnboardingMessage(
            text: """
            👋 **Welcome to SwiftBuddy!**
            
            I'm your native Swift AI assistant running on Apple Silicon with MLX inference.
            
            To get started, I recommend downloading **\(model.displayName)** (\(model.size)):
            
            ✨ **Why this model?**
            • Blazing fast at 61.5 tokens/sec on Apple Silicon
            • Mixture-of-Experts architecture (26B params)
            • MTP (Multi-Token Prediction) acceleration
            • Excellent for coding, reasoning, and complex tasks
            • Optimized for high memory bandwidth on M-series chips
            
            **Requirements:** 24GB+ RAM recommended (works on M1 Max, M2 Pro/Max/Ultra, M3 Pro/Max/Ultra, M4 Pro/Max)
            
            Would you like me to download and set it up for you?
            """,
            actions: [
                .download(modelId: model.modelId, displayName: model.displayName),
                .skip,
                .browse
            ]
        )
    }
    
    /// Generate message after model download starts.
    func getDownloadingMessage(modelName: String) -> OnboardingMessage {
        OnboardingMessage(
            text: """
            📥 **Downloading \(modelName)...**
            
            This will take a few minutes depending on your connection.
            You can monitor the progress in the Models tab.
            
            While you wait, here's what SwiftBuddy can do:
            
            🚀 **Runtime Engines:**
            • **MLX Standard** - Balanced performance for most models
            • **DFlash** - Memory optimization for 40K-100K context
            • **Speculative** - Fast inference with draft models
            • **Streaming MoE** - Run 122B-397B models on 24GB RAM
            
            💡 **Preset Configurations:**
            Each engine has validated parameter presets like "Balanced", "Creative", "Precise" - no manual tuning needed!
            
            I'll let you know when the model is ready. 🎯
            """,
            actions: []
        )
    }
    
    /// Generate message after successful model download.
    func getModelReadyMessage(modelName: String) -> OnboardingMessage {
        OnboardingMessage(
            text: """
            ✅ **\(modelName) is ready!**
            
            Your model has been downloaded and configured with the **Balanced** preset:
            • Temperature: 0.7
            • Top-P: 0.9
            • Context window: 8K tokens
            
            🎉 **You're all set!**
            
            Try asking me:
            • "Explain how Swift actors work"
            • "Write a REST API with Vapor"
            • "Debug this code: [paste code]"
            • "What's the best way to handle async errors?"
            
            You can change the runtime engine and preset anytime in Settings. Let's chat! 💬
            """,
            actions: [
                .openSettings,
                .startChatting
            ]
        )
    }
    
    /// Generate message if user skips model download.
    func getSkipMessage() -> OnboardingMessage {
        OnboardingMessage(
            text: """
            👌 **No problem!**
            
            You can browse and download models anytime from the **Models** tab.
            
            🔍 **Quick tips:**
            • Use the search bar to find models on Hugging Face
            • Filter by size, quantization, and architecture
            • Check compatibility with your runtime engine
            
            📚 **Popular choices:**
            • **Qwen2.5** series - Excellent all-rounders
            • **Llama-3.2** - Meta's latest models
            • **Phi-3.5** - Small but powerful
            • **Gemma-2** - Google's instruction models
            
            Head to the Models tab when you're ready! 🚀
            """,
            actions: [
                .openModels
            ]
        )
    }
    
    // MARK: — State Management
    
    /// Mark onboarding as completed.
    func completeOnboarding() {
        onboardingStep = .completed
        UserDefaults.standard.set(true, forKey: "swiftbuddy.onboardingCompleted")
    }
    
    /// Reset onboarding state (for testing).
    func resetOnboarding() {
        UserDefaults.standard.removeObject(forKey: "swiftbuddy.hasLaunched")
        UserDefaults.standard.removeObject(forKey: "swiftbuddy.onboardingCompleted")
        isFirstLaunch = true
        onboardingStep = .welcome
    }
}

// MARK: — Supporting Types

struct ModelRecommendation {
    let modelId: String
    let displayName: String
    let size: String
    let runtime: String
    let preset: String
    let description: String
}

struct OnboardingMessage {
    let text: String
    let actions: [OnboardingAction]
}

enum OnboardingAction: Identifiable {
    case download(modelId: String, displayName: String)
    case skip
    case browse
    case openSettings
    case openModels
    case startChatting
    
    var id: String {
        switch self {
        case .download(let modelId, _): return "download-\(modelId)"
        case .skip: return "skip"
        case .browse: return "browse"
        case .openSettings: return "settings"
        case .openModels: return "models"
        case .startChatting: return "chat"
        }
    }
    
    var title: String {
        switch self {
        case .download(_, let displayName): return "📥 Download \(displayName)"
        case .skip: return "⏭️ Skip for Now"
        case .browse: return "🔍 Browse Models"
        case .openSettings: return "⚙️ Open Settings"
        case .openModels: return "📚 Browse Models"
        case .startChatting: return "💬 Start Chatting"
        }
    }
    
    var icon: String {
        switch self {
        case .download: return "arrow.down.circle.fill"
        case .skip: return "forward.fill"
        case .browse: return "magnifyingglass"
        case .openSettings: return "gearshape.fill"
        case .openModels: return "square.grid.2x2.fill"
        case .startChatting: return "message.fill"
        }
    }
}
