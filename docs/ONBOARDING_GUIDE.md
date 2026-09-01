# SwiftBuddy Onboarding System

## Overview

SwiftBuddy features a **conversational onboarding system** that guides first-time users through setup using the chat interface. Instead of traditional modal dialogs or setup wizards, onboarding happens naturally as system messages in the chat.

## Design Philosophy

### 1. **Chat-Native Experience**
- Onboarding messages appear as special system messages in the chat
- Users interact via action buttons embedded in the messages
- Feels natural and contextual rather than interrupting

### 2. **Intelligent Bootstrap**
- Recommends **Qwen2.5-7B-Instruct (4-bit)** as the default starter model
- Pre-validated runtime configuration (MLX Standard + Balanced preset)
- One-click download and setup

### 3. **Non-Blocking**
- Users can skip onboarding and explore freely
- No forced workflows or locked features
- Progressive disclosure of advanced features

## Recommended Bootstrap Model

**Model:** `mlx-community/Qwen2.5-7B-Instruct-4bit`

**Why Qwen2.5-7B?**
- ✅ **Size:** 4.5 GB (manageable download, fits on 8GB+ machines)
- ✅ **Quality:** Excellent instruction-following, great for demos
- ✅ **Speed:** Fast inference on M1/M2/M3/M4 chips
- ✅ **Compatibility:** Works perfectly with MLX Standard runtime
- ✅ **Versatility:** Coding, writing, reasoning, multilingual support
- ✅ **Reliability:** Well-tested, stable, popular in MLX community

**Alternative Recommendations:**
- **Smaller:** `mlx-community/Llama-3.2-3B-Instruct-4bit` (2.1 GB) - faster but less capable
- **Larger:** `mlx-community/Qwen2.5-14B-Instruct-4bit` (8.5 GB) - higher quality, needs 16GB+ RAM
- **Vision:** `mlx-community/Qwen2-VL-7B-Instruct-4bit` (4.8 GB) - includes vision capabilities

## Onboarding Flow

### Step 1: Welcome Message
**Trigger:** App first launch (detected via `UserDefaults`)

```
👋 Welcome to SwiftBuddy!

I'm your native Swift AI assistant running on Apple Silicon with MLX inference.

To get started, I recommend downloading Qwen2.5 7B Instruct (4.5 GB):

✨ Why this model?
• Excellent instruction-following capabilities
• Optimized for Apple Silicon with 4-bit quantization
• Fast inference on M-series chips
• Great for coding, writing, and general assistance

Would you like me to download and set it up for you?

[📥 Download Qwen2.5 7B] [⏭️ Skip for Now] [🔍 Browse Models]
```

**User Actions:**
- **Download** → Proceed to Step 2
- **Skip** → Show skip message, complete onboarding
- **Browse** → Navigate to Models tab

### Step 2: Downloading Message
**Trigger:** User clicks "Download"

```
📥 Downloading Qwen2.5 7B Instruct...

This will take a few minutes depending on your connection.
You can monitor the progress in the Models tab.

While you wait, here's what SwiftBuddy can do:

🚀 Runtime Engines:
• MLX Standard - Balanced performance for most models
• DFlash - Memory optimization for 40K-100K context
• Speculative - Fast inference with draft models
• Streaming MoE - Run 122B-397B models on 24GB RAM

💡 Preset Configurations:
Each engine has validated parameter presets like "Balanced", 
"Creative", "Precise" - no manual tuning needed!

I'll let you know when the model is ready. 🎯
```

**Background:** Model download starts automatically

### Step 3: Model Ready Message
**Trigger:** Download completes successfully

```
✅ Qwen2.5 7B is ready!

Your model has been downloaded and configured with the Balanced preset:
• Temperature: 0.7
• Top-P: 0.9
• Context window: 8K tokens

🎉 You're all set!

Try asking me:
• "Explain how Swift actors work"
• "Write a REST API with Vapor"
• "Debug this code: [paste code]"
• "What's the best way to handle async errors?"

You can change the runtime engine and preset anytime in Settings.
Let's chat! 💬

[⚙️ Open Settings] [💬 Start Chatting]
```

**User Actions:**
- **Open Settings** → Navigate to Settings tab
- **Start Chatting** → Clear onboarding, enable chat input

### Alternative: Skip Flow
**Trigger:** User clicks "Skip for Now"

```
👌 No problem!

You can browse and download models anytime from the Models tab.

🔍 Quick tips:
• Use the search bar to find models on Hugging Face
• Filter by size, quantization, and architecture
• Check compatibility with your runtime engine

📚 Popular choices:
• Qwen2.5 series - Excellent all-rounders
• Llama-3.2 - Meta's latest models
• Phi-3.5 - Small but powerful
• Gemma-2 - Google's instruction models

Head to the Models tab when you're ready! 🚀

[📚 Browse Models]
```

## Implementation Architecture

### Components

1. **OnboardingService.swift**
   - Manages onboarding state and progression
   - Generates context-aware messages
   - Tracks completion status

2. **OnboardingMessageView.swift**
   - SwiftUI component for rendering onboarding messages
   - Styled action buttons with icons
   - Responsive layout for iOS/macOS

3. **ChatViewModel.swift** (Extended)
   - Integrates OnboardingService
   - Handles onboarding actions
   - Manages message injection

4. **ChatMessage.swift** (Extended)
   - Added `isOnboarding: Bool` flag
   - Distinguishes onboarding from regular messages
   - Prevents onboarding messages from being sent to model

### State Management

**Persistence:**
```swift
UserDefaults.standard.bool(forKey: "swiftbuddy.hasLaunched")
UserDefaults.standard.bool(forKey: "swiftbuddy.onboardingCompleted")
```

**Onboarding Steps:**
```swift
enum OnboardingStep {
    case welcome
    case modelDownload(modelId: String)
    case runtimeSetup
    case completed
}
```

### Navigation Integration

**Notifications:**
```swift
extension Notification.Name {
    static let navigateToModels
    static let navigateToSettings
}
```

**Usage:** Onboarding actions can trigger tab/view navigation

## UI Design

### Message Styling
- **Background:** Accent color with 8% opacity
- **Border:** Accent color with 20% opacity
- **Text:** Markdown-rendered with full formatting support
- **Actions:** Prominent buttons with hover states

### Action Buttons
- **Primary (Download):** Accent color, emphasized
- **Secondary (Skip/Browse):** Neutral, subtle
- **Icons:** SF Symbols for consistency

### Responsive Layout
- **iOS:** Full-width cards with stacked buttons
- **macOS:** Flexible width with horizontal button layout

## Testing & Debugging

### Reset Onboarding
```swift
OnboardingService().resetOnboarding()
```

### Manual Triggers
```swift
// Force show welcome message
viewModel.showOnboardingIfNeeded()

// Trigger model ready notification
viewModel.notifyModelReady(modelName: "Qwen2.5 7B")
```

### State Inspection
```swift
print(onboardingService.isFirstLaunch)
print(onboardingService.onboardingStep)
print(onboardingService.recommendedModel)
```

## Best Practices

### For Users
1. **Follow the guided flow** for best first experience
2. **Try the recommended model** before exploring others
3. **Use preset configurations** instead of manual tuning
4. **Check Settings** to customize runtime/presets later

### For Developers
1. **Keep messages concise** - users should scan quickly
2. **Use emoji sparingly** - visual hierarchy, not decoration
3. **Test skip flow** - ensure graceful degradation
4. **Handle errors** - network issues, disk space, etc.
5. **Update model recommendation** as MLX ecosystem evolves

## Future Enhancements

### Potential Additions
- **Hardware detection:** Recommend models based on RAM/GPU
- **Use case selection:** "Coding" / "Writing" / "Research" flows
- **Multi-model onboarding:** Download complementary models (e.g., vision + text)
- **Interactive tutorial:** Guide through first chat interaction
- **Performance benchmarking:** Test model speed on user's hardware

### Analytics (Privacy-Preserving)
- Track onboarding completion rate
- Measure skip vs. download ratio
- Monitor recommended model success rate
- Identify common pain points

## Configuration Options

### Customize Bootstrap Model
```swift
// In OnboardingService.swift, modify:
recommendedModel = ModelRecommendation(
    modelId: "your-model-id",
    displayName: "Your Model Name",
    size: "X.X GB",
    runtime: "mlx.standard",
    preset: "mlx.standard.balanced",
    description: "Why this model is great..."
)
```

### Adjust Message Content
Edit message generation methods in `OnboardingService.swift`:
- `getWelcomeMessage()`
- `getDownloadingMessage(modelName:)`
- `getModelReadyMessage(modelName:)`
- `getSkipMessage()`

### Disable Onboarding
```swift
// Skip onboarding entirely
UserDefaults.standard.set(true, forKey: "swiftbuddy.hasLaunched")
UserDefaults.standard.set(true, forKey: "swiftbuddy.onboardingCompleted")
```

## Troubleshooting

### Onboarding doesn't appear
- **Check:** `OnboardingService.isFirstLaunch` should be `true`
- **Fix:** Reset UserDefaults or reinstall app
- **Debug:** Add breakpoint in `showOnboardingIfNeeded()`

### Model download fails
- **Check:** Network connection and Hugging Face availability
- **Fix:** Retry download or choose different model
- **Debug:** Check console logs for HTTP errors

### Action buttons don't work
- **Check:** `handleOnboardingAction(_:)` is being called
- **Fix:** Ensure ChatViewModel has reference to downloadManager
- **Debug:** Print action ID in button tap handler

## Support & Resources

- **Model Registry:** [mlx-community on Hugging Face](https://huggingface.co/mlx-community)
- **MLX Documentation:** [ml-explore/mlx](https://github.com/ml-explore/mlx)
- **SwiftBuddy Issues:** Report bugs via GitHub Issues
- **Community:** Join discussions in SwiftLM Discord/Forum

---

**Version:** 1.0  
**Last Updated:** May 11, 2026  
**Author:** SwiftBuddy Team
