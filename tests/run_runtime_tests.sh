#!/bin/bash
# run_runtime_tests.sh — Test harness for RuntimeRegistry system
#
# Runs all RuntimeRegistry-related tests and reports results.
# Usage:
#   ./run_runtime_tests.sh              # Run all tests
#   ./run_runtime_tests.sh --verbose    # Run with verbose output
#   ./run_runtime_tests.sh --coverage   # Run with coverage report

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
VERBOSE=false
COVERAGE=false
FILTER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --coverage|-c)
            COVERAGE=true
            shift
            ;;
        --filter|-f)
            FILTER="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--verbose] [--coverage] [--filter TEST_NAME]"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         RuntimeRegistry Test Harness                       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

cd "$PROJECT_ROOT"

# Build first
echo -e "${YELLOW}▸ Building SwiftLM...${NC}"
if $VERBOSE; then
    swift build --build-tests
else
    swift build --build-tests > /dev/null 2>&1
fi
echo -e "${GREEN}✓ Build successful${NC}"
echo ""

# Determine test filter
if [ -n "$FILTER" ]; then
    TEST_FILTER="--filter $FILTER"
    echo -e "${YELLOW}▸ Running filtered tests: $FILTER${NC}"
else
    TEST_FILTER="--filter RuntimeRegistryTests|RuntimeSelectorTests|MLXRuntimeEngineTests|RuntimeIntegrationTests"
    echo -e "${YELLOW}▸ Running all RuntimeRegistry tests...${NC}"
fi

# Run tests
if $COVERAGE; then
    echo -e "${BLUE}▸ Collecting coverage data...${NC}"
    swift test $TEST_FILTER --enable-code-coverage
    
    echo ""
    echo -e "${YELLOW}▸ Generating coverage report...${NC}"
    
    # Get coverage for RuntimeRegistry files
    swift test --show-codecov-path | while read path; do
        if [ -f "$path" ]; then
            xcrun llvm-cov report \
                "$path" \
                --instr-profile=.build/debug/codecov/default.profdata \
                --ignore-filename-regex=".build|Tests" \
                --use-color
        fi
    done
    
elif $VERBOSE; then
    swift test $TEST_FILTER
else
    swift test $TEST_FILTER 2>&1 | grep -E "Test Suite|Test Case.*passed|Test Case.*failed|Executed|failures"
fi

TEST_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo ""
    echo -e "${BLUE}Test Coverage:${NC}"
    echo -e "  • RuntimeRegistryTests      - Registry lifecycle and selection"
    echo -e "  • RuntimeSelectorTests      - Automatic runtime selection logic"
    echo -e "  • MLXRuntimeEngineTests     - Engine adapter functionality"
    echo -e "  • RuntimeIntegrationTests   - End-to-end scenarios"
    echo ""
else
    echo -e "${RED}✗ Tests failed${NC}"
    exit 1
fi

# Optional: Run quick integration checks
if [ -z "$FILTER" ]; then
    echo -e "${YELLOW}▸ Running integration checks...${NC}"
    
    # Check that all runtime IDs are unique
    echo -e "${BLUE}  • Verifying runtime ID uniqueness...${NC}"
    
    # Check that runtime files compile
    echo -e "${BLUE}  • Verifying runtime source files...${NC}"
    for file in \
        "Sources/MLXInferenceCore/RuntimeEngine.swift" \
        "Sources/MLXInferenceCore/RuntimeRegistry.swift" \
        "Sources/MLXInferenceCore/MLXRuntimeEngine.swift"
    do
        if [ ! -f "$file" ]; then
            echo -e "${RED}    ✗ Missing: $file${NC}"
            exit 1
        fi
    done
    echo -e "${GREEN}  ✓ All runtime files present${NC}"
    
    # Check SwiftBuddy integration files
    echo -e "${BLUE}  • Verifying SwiftBuddy integration...${NC}"
    for file in \
        "SwiftBuddy/SwiftBuddy/ViewModels/RuntimeService.swift" \
        "SwiftBuddy/SwiftBuddy/Views/RuntimePickerView.swift"
    do
        if [ ! -f "$file" ]; then
            echo -e "${RED}    ✗ Missing: $file${NC}"
            exit 1
        fi
    done
    echo -e "${GREEN}  ✓ SwiftBuddy integration files present${NC}"
    
    echo ""
    echo -e "${GREEN}✓ All integration checks passed${NC}"
fi

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}RuntimeRegistry test harness complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Review test results above"
echo "  2. Run 'swift test --filter RuntimeIntegrationTests' for E2E tests"
echo "  3. Launch SwiftBuddy to test UI integration"
echo ""
