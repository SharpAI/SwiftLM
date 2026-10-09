#!/bin/bash
# Makes `swift test` runnable locally without CI's help.
#
# SwiftPM compiles the MLX kernels itself and embeds the metallib in each test bundle, so
# nothing needs to be copied around; the only prerequisite is the Metal Toolchain (Xcode 26+
# ships it as a separate download). Do NOT copy a metallib into <bundle>.xctest/Contents/MacOS:
# it invalidates the bundle signature and the next incremental build fails at CodeSign.
#
# Usage: scripts/bootstrap_local_tests.sh [--with-server]
#   --with-server  also build the release SwiftLM binary. A few SwiftBuddyTests (VLM, audio)
#                  launch it from .build/release/SwiftLM and fail with "Could not find
#                  SwiftLM executable" without it.
set -eo pipefail
cd "$(dirname "$0")/.."

bash scripts/check-metal-toolchain.sh

echo "=> Building test harness (swift build --build-tests)..."
swift build --build-tests

if [ "${1:-}" = "--with-server" ]; then
    echo "=> Building the SwiftLM server binary (swift build -c release)..."
    swift build -c release
fi

echo "=> Done. Run tests with: swift test --skip-build --filter SwiftLMTests --disable-swift-testing"
