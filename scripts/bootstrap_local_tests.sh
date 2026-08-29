#!/bin/bash
# Makes `swift test` runnable locally without CI's help.
#
# A bare `swift test` aborts with "Failed to load the default metallib"
# because Package.swift links MLX but nothing on a local machine ever builds
# or installs mlx.metallib. CI works around this in .github/workflows/ci.yml
# ("Install MLX Metal library" step) by pip-installing the `mlx` wheel and
# copying its bundled metallib into every built .xctest bundle. This script
# does the same thing locally.
set -eo pipefail

VENV_DIR="${MLX_METALLIB_VENV:-/tmp/swiftlm_mlx_venv}"

echo "=> Building test harness (swift build --build-tests)..."
swift build --build-tests

echo "=> Installing MLX Metal library..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/pip" install --quiet --upgrade mlx

METALLIB=$(find "$VENV_DIR" -name "mlx.metallib" | head -1)
if [ -z "$METALLIB" ]; then
    echo "error: mlx.metallib not found after pip install mlx" >&2
    exit 1
fi

cp "$METALLIB" .build/debug/ 2>/dev/null || true
cp "$METALLIB" .build/release/ 2>/dev/null || true
find .build -type d -name "MacOS" -exec cp "$METALLIB" {}/ \;

echo "=> Done. Run tests with: swift test --skip-build"
