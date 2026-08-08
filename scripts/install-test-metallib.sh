#!/usr/bin/env bash
# install-test-metallib.sh — put mlx.metallib where the test bundles can find it.
#
# MLX aborts the process when it cannot load its Metal library, so without this a bare
# `swift test` dies on the first test that touches MLX — in practice the second test of
# the run, leaving ~250 tests unexecuted while the output still looks like a normal
# (short) pass. CI does this inline; this script is the same step for local use.
#
# Usage: scripts/install-test-metallib.sh [debug|release]   (default: both, if present)

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIGS=("${1:-debug release}")

find_metallib() {
    # 1. Already built by build.sh
    for candidate in .build/*/release/mlx.metallib .build/*/release/default.metallib default.metallib; do
        [ -f "$candidate" ] && { echo "$candidate"; return 0; }
    done
    # 2. From the mlx Python wheel, as CI does
    local venv="${TMPDIR:-/tmp}/swiftlm_mlx_venv"
    if [ ! -f "$venv"/lib/python*/site-packages/mlx/lib/mlx.metallib ]; then
        echo "Fetching mlx.metallib from the mlx wheel…" >&2
        python3 -m venv "$venv" >&2
        "$venv/bin/pip" install --quiet mlx >&2
    fi
    ls "$venv"/lib/python*/site-packages/mlx/lib/mlx.metallib 2>/dev/null | head -1
}

METALLIB="$(find_metallib)"
[ -n "$METALLIB" ] && [ -f "$METALLIB" ] || { echo "error: could not obtain mlx.metallib" >&2; exit 1; }
echo "Using $METALLIB"

INSTALLED=0
# Next to the built products, and inside every .xctest bundle — MLX looks for a metallib
# colocated with the running binary, which for tests is inside the bundle.
while IFS= read -r dir; do
    cp "$METALLIB" "$dir/mlx.metallib"
    cp "$METALLIB" "$dir/default.metallib"
    INSTALLED=$((INSTALLED + 1))
done < <(
    { for c in $CONFIGS; do ls -d .build/*/"$c" .build/"$c" 2>/dev/null; done
      find .build -type d -name "MacOS" 2>/dev/null; } | sort -u
)

echo "Installed into $INSTALLED location(s)."
[ "$INSTALLED" -gt 0 ] || { echo "error: no build directories found — run 'swift build --build-tests' first" >&2; exit 1; }
