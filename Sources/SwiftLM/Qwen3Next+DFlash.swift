// Copyright 2026 SwiftLM Contributors
// MIT License — see LICENSE file
// Bridge: Qwen3Next models conform to DFlashTargetModel
//
// The dflash* methods are defined on Qwen3NextModel in the
// MLXLLM module. This file adds the DFlashTargetModel protocol conformance
// so the DFlash runtime can use them generically.

import DFlash
import MLX
import MLXLLM
import MLXLMCommon

// MARK: - Qwen3NextModel + DFlashTargetModel

extension Qwen3NextModel: DFlashTargetModel {}
