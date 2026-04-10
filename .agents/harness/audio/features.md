# Audio Model — Feature Registry

## Scope
SwiftLM currently has zero audio support. This harness defines the TDD contract for building audio capabilities from scratch: mel spectrogram generation, audio token embedding, Whisper-class STT, multimodal audio fusion, and TTS output. Features are ordered by implementation dependency.

## Source Locations (Planned)

| Component | Location | Status |
|---|---|---|
| Audio CLI flag | `Sources/SwiftLM/SwiftLM.swift` | 🔲 Not implemented |
| Audio input parsing | `Sources/SwiftLM/Server.swift` (`extractAudio()`) | 🔲 Not implemented |
| Mel spectrogram | `Sources/SwiftLM/AudioProcessing.swift` | 🔲 Not created |
| Audio model registry | `mlx-swift-lm/Libraries/MLXALM/` | 🔲 Not created |
| Whisper encoder | `mlx-swift-lm/Libraries/MLXALM/Models/Whisper.swift` | 🔲 Not created |
| TTS vocoder | `Sources/SwiftLM/TTSVocoder.swift` | 🔲 Not created |

## Features

### Phase 1 — Audio Input Pipeline

| # | Feature | Status | Test | Last Verified |
|---|---------|--------|------|---------------|
| 1 | `--audio` CLI flag is accepted without crash | 🔲 TODO | `testAudio_AudioFlagAccepted` | — |
| 2 | Base64 WAV data URI extraction from API content | 🔲 TODO | `testAudio_Base64WAVExtraction` | — |
| 3 | WAV header parsing: extract sample rate, channels, bit depth | 🔲 TODO | `testAudio_WAVHeaderParsing` | — |
| 4 | PCM samples → mel spectrogram via FFT | 🔲 TODO | `testAudio_MelSpectrogramGeneration` | — |
| 5 | Mel spectrogram dimensions match Whisper's expected input (80 bins × N frames) | 🔲 TODO | `testAudio_MelDimensionsCorrect` | — |
| 6 | Audio longer than 30s is chunked into segments | 🔲 TODO | `testAudio_LongAudioChunking` | — |
| 7 | Empty/silent audio returns empty transcription (no crash) | 🔲 TODO | `testAudio_SilentAudioHandling` | — |

### Phase 2 — Speech-to-Text (STT)

| # | Feature | Status | Test | Last Verified |
|---|---------|--------|------|---------------|
| 8 | Whisper model type registered in ALM factory | 🔲 TODO | `testAudio_WhisperRegistered` | — |
| 9 | Whisper encoder produces valid hidden states from mel input | 🔲 TODO | `testAudio_WhisperEncoderOutput` | — |
| 10 | Whisper decoder generates token sequence from encoder output | 🔲 TODO | `testAudio_WhisperDecoderOutput` | — |
| 11 | `/v1/audio/transcriptions` endpoint returns JSON with text field | 🔲 TODO | `testAudio_TranscriptionEndpoint` | — |
| 12 | Transcription of known fixture WAV matches expected text | 🔲 TODO | `testAudio_TranscriptionAccuracy` | — |

### Phase 3 — Multimodal Audio Fusion

| # | Feature | Status | Test | Last Verified |
|---|---------|--------|------|---------------|
| 13 | Gemma 4 `audio_config` is parsed from config.json | 🔲 TODO | `testAudio_Gemma4ConfigParsed` | — |
| 14 | Audio tokens interleaved with text tokens at correct positions | 🔲 TODO | `testAudio_TokenInterleaving` | — |
| 15 | `boa_token_id` / `eoa_token_id` correctly bracket audio segments | 🔲 TODO | `testAudio_AudioTokenBoundaries` | — |
| 16 | Mixed text + audio + vision request processed without crash | 🔲 TODO | `testAudio_TrimodalRequest` | — |

### Phase 4 — Text-to-Speech (TTS) Output

| # | Feature | Status | Test | Last Verified |
|---|---------|--------|------|---------------|
| 17 | `/v1/audio/speech` endpoint accepts text input | 🔲 TODO | `testAudio_TTSEndpointAccepts` | — |
| 18 | TTS vocoder generates valid PCM waveform from tokens | 🔲 TODO | `testAudio_VocoderOutput` | — |
| 19 | Generated WAV has valid header and is playable | 🔲 TODO | `testAudio_ValidWAVOutput` | — |
| 20 | Streaming audio chunks sent as Server-Sent Events | 🔲 TODO | `testAudio_StreamingTTSOutput` | — |
