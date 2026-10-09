#!/usr/bin/env bash
# check-metal-toolchain.sh — fail early, with the fix, when the Metal compiler is missing.
#
# SwiftPM compiles the Cmlx kernels itself and embeds `default.metallib` in
# mlx-swift_Cmlx.bundle inside every .xctest, so `swift build --build-tests` followed by
# `swift test --skip-build` needs no hand-copied metallib. Copying one into
# <bundle>.xctest/Contents/MacOS invalidates the bundle signature and the next incremental
# build then fails at CodeSign. The one thing the build does need is the Metal compiler,
# which Xcode 26+ ships as a separate download. See #128 Tier 3.
set -euo pipefail

if xcrun -find metal >/dev/null 2>&1; then
    exit 0
fi

cat >&2 <<'MSG'
error: the Metal Toolchain is not installed, so the MLX kernels cannot be compiled.
Install it with:

    xcodebuild -downloadComponent MetalToolchain
MSG
exit 1
