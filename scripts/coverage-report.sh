#!/usr/bin/env bash
# coverage-report.sh — measure where the test suites are blind.
#
# Tier 4 of #128. The suites in this repository have never surfaced a defect: every
# bug in the #108/#110/#112 cycle was caught by loading a real checkpoint or by code
# review. Knowing *which* code they never execute turns that from an impression into
# a list, and tells you whether adding tests to what exists is worth more than adding
# a fixture for a shape that is missing.
#
# There are two suites and they cover different things, so both are measured:
#   - the SwiftLM package: server, inference core, the app
#   - the mlx-swift-lm submodule: the model architectures
# Reading either alone is misleading. MLXLLM/Models looks like 0% from the umbrella
# package because its tests live in the submodule, not because it is untested.
#
# Coverage is a map of what was executed, not evidence that behaviour is correct — a
# line can be run by a test that asserts nothing. Treat a low number as a question and
# a high number as no answer at all.
#
# Usage:
#   ./scripts/coverage-report.sh            # both suites
#   ./scripts/coverage-report.sh package    # SwiftLM package only
#   ./scripts/coverage-report.sh submodule  # mlx-swift-lm only

set -uo pipefail

WHICH="${1:-both}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
YELLOW='\033[1;33m'; NC='\033[0m'
log() { echo -e "${YELLOW}[coverage]${NC} $*"; }

# Aggregates an llvm-cov JSON export by area and lists the largest untouched files.
# Kept inline so the script is one file to copy and run.
summarise() {
    python3 - "$1" "$2" <<'PY'
import json, sys, collections

export_path, label = sys.argv[1], sys.argv[2]
with open(export_path) as fh:
    data = json.load(fh)["data"][0]

def area(path):
    if "/MLXLLM/Models/" in path:  return "model architectures (LLM)"
    if "/MLXVLM/Models/" in path:  return "model architectures (VLM)"
    if "/MLXLMCommon/" in path:    return "MLXLMCommon"
    if "/Sources/SwiftLM/" in path:         return "server"
    if "/Sources/MLXInferenceCore/" in path: return "inference core"
    if "/SwiftBuddy/" in path:     return "SwiftBuddy app"
    if "/Tests/" in path or "/tests/" in path: return "test code"
    if "/mlx-swift-lm/" in path:   return "submodule (other)"
    return None   # third-party dependencies; not ours to cover

agg = collections.defaultdict(lambda: [0, 0, 0])
untouched = []
for f in data["files"]:
    a = area(f["filename"])
    if a is None:
        continue
    lines = f["summary"]["lines"]
    agg[a][0] += lines["covered"]
    agg[a][1] += lines["count"]
    agg[a][2] += 1
    # Only flag files big enough that never executing them means something.
    if lines["percent"] == 0 and lines["count"] >= 200:
        untouched.append((lines["count"], f["filename"].split("/")[-1], a))

print(f"\n  ── {label} " + "─" * max(0, 56 - len(label)))
print(f"  {'AREA':<28} {'COVERED':>8} {'LINES':>8} {'PCT':>7}  FILES")
for a, (cov, tot, n) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
    pct = 100 * cov / tot if tot else 0.0
    print(f"  {a:<28} {cov:>8} {tot:>8} {pct:>6.1f}%  {n}")

if untouched:
    print(f"\n  largest files never executed by this suite:")
    for count, name, a in sorted(untouched, reverse=True)[:10]:
        print(f"    {count:>6} lines  {name:<28} ({a})")
PY
}

run_suite() {
    local dir="$1" label="$2"
    log "Running $label suite with coverage (this rebuilds in debug; give it a minute)"
    (
        cd "$dir" || exit 1
        swift test --enable-code-coverage > /tmp/coverage-$$.log 2>&1
        status=$?
        # Test failures still leave usable coverage data, so report and continue.
        if [ "$status" -ne 0 ]; then
            echo "  note: tests exited $status — coverage below reflects what ran"
            grep -aE "error:|failed" /tmp/coverage-$$.log | head -3
        fi
        grep -aE "Executed [0-9]+ tests" /tmp/coverage-$$.log | tail -1 | sed 's/^/  /'

        local profdata bundle binary
        profdata="$(dirname "$(swift test --show-codecov-path 2>/dev/null)")/default.profdata"
        bundle="$(ls -d .build/*/debug/*.xctest 2>/dev/null | head -1)"
        if [ ! -f "$profdata" ] || [ -z "$bundle" ]; then
            echo "  could not locate coverage artifacts for $label — skipping"
            exit 1
        fi
        binary="$bundle/Contents/MacOS/$(basename "$bundle" .xctest)"
        xcrun llvm-cov export -format=text -instr-profile "$profdata" "$binary" \
            > /tmp/coverage-export-$$.json 2>/dev/null
        summarise /tmp/coverage-export-$$.json "$label"
        rm -f /tmp/coverage-$$.log /tmp/coverage-export-$$.json
    )
}

# The package suite aborts on its second test without this; see #128 Tier 3.
if [ -x "$REPO_ROOT/scripts/install-test-metallib.sh" ]; then
    bash "$REPO_ROOT/scripts/install-test-metallib.sh" >/dev/null 2>&1
fi

case "$WHICH" in
    package)   run_suite "$REPO_ROOT" "SwiftLM package" ;;
    submodule) run_suite "$REPO_ROOT/mlx-swift-lm" "mlx-swift-lm submodule" ;;
    both)
        run_suite "$REPO_ROOT" "SwiftLM package"
        run_suite "$REPO_ROOT/mlx-swift-lm" "mlx-swift-lm submodule"
        ;;
    *) echo "usage: $0 [package|submodule|both]"; exit 2 ;;
esac

echo
log "Coverage measures execution, not correctness. A covered line may still be"
log "asserted on by nothing at all."
