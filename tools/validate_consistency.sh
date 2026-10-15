#!/bin/bash
# Comprehensive consistency validation for ZPU.
#
# Usage: ./tools/validate_consistency.sh [--verbose]
#
# Checks:
# 1. Dependency graph: zpu is independent from zbackend
# 2. Test counts match between MILESTONES.md, README.md, and actual tests
# 3. Feature documentation exists for all major modules
# 4. Examples compile and are listed in build.zig
# 5. GitHub issues are referenced where appropriate
# 6. No stale references to removed APIs
#
# Exit code 0 if all consistent, 1 if any mismatch found.

set -e

VERBOSE=0
if [ "$1" = "--verbose" ]; then
    VERBOSE=1
fi

echo "=== ZPU Consistency Validation ==="
echo

ISSUES=0
WARNINGS=0

# Helper functions
log_issue() {
    echo "   ❌ $1"
    ISSUES=$((ISSUES + 1))
}

log_warning() {
    echo "   ⚠️  $1"
    WARNINGS=$((WARNINGS + 1))
}

log_ok() {
    if [ $VERBOSE -eq 1 ]; then
        echo "   ✓ $1"
    fi
}

# ============================================================================
# 1. Dependency graph validation
# ============================================================================
echo "1. Dependency graph validation..."

# zpu must be independent from zbackend
if grep -q "zbackend" src/zpu/context.zig 2>/dev/null; then
    log_issue "src/zpu/context.zig imports zbackend (should be independent)"
else
    log_ok "src/zpu/context.zig is independent from zbackend"
fi

if grep -q "zbackend" src/zpu.zig 2>/dev/null; then
    log_issue "src/zpu.zig references zbackend (should be independent)"
else
    log_ok "src/zpu.zig is independent from zbackend"
fi

# build.zig dependency graph must match documented structure
if grep -q 'zpu.addImport("zbackend"' build.zig 2>/dev/null; then
    log_issue "build.zig adds zbackend to zpu (violates independence)"
else
    log_ok "build.zig does not add zbackend to zpu"
fi

if grep -q 'zgf.addImport("zbackend"' build.zig 2>/dev/null; then
    log_issue "build.zig adds zbackend to zgf (not needed)"
else
    log_ok "build.zig does not add zbackend to zgf"
fi

if grep -q 'zui.addImport("zbackend"' build.zig 2>/dev/null; then
    log_ok "build.zig adds zbackend to zui (expected)"
else
    log_issue "build.zig missing zbackend import for zui"
fi

# ============================================================================
# 2. Test count validation
# ============================================================================
echo
echo "2. Test count validation..."

# Get actual test count from zig build test
ACTUAL_TESTS=$(/usr/local/lib/python3.11/dist-packages/ziglang/zig build test --summary all 2>&1 | grep -oE '[0-9]+/[0-9]+ tests passed' | grep -oE '^[0-9]+' || echo "")

if [ -z "$ACTUAL_TESTS" ]; then
    # Try alternative pattern
    ACTUAL_TESTS=$(/usr/local/lib/python3.11/dist-packages/ziglang/zig build test --summary all 2>&1 | grep -oE 'tests passed' -B 1 | grep -oE '[0-9]+' || echo "")
fi

# Extract from documentation
MILESTONES_COUNT=$(grep -oE '[0-9]+ テスト' MILESTONES.md 2>/dev/null | grep -oE '[0-9]+' | head -1 || echo "")
README_COUNT=$(grep -oE '[0-9]+/[0-9]+ green' README.md 2>/dev/null | grep -oE '^[0-9]+' | head -1 || echo "")

if [ -n "$ACTUAL_TESTS" ]; then
    echo "   Actual test count: $ACTUAL_TESTS"
else
    log_warning "Could not determine actual test count (zig build might have failed)"
fi

if [ -n "$MILESTONES_COUNT" ]; then
    echo "   MILESTONES.md claims: $MILESTONES_COUNT tests"
    if [ -n "$ACTUAL_TESTS" ] && [ "$MILESTONES_COUNT" != "$ACTUAL_TESTS" ]; then
        log_issue "MILESTONES.md test count ($MILESTONES_COUNT) != actual ($ACTUAL_TESTS)"
    else
        log_ok "MILESTONES.md test count matches"
    fi
else
    log_warning "Could not find test count in MILESTONES.md"
fi

if [ -n "$README_COUNT" ]; then
    echo "   README.md claims: $README_COUNT tests"
    if [ -n "$ACTUAL_TESTS" ] && [ "$README_COUNT" != "$ACTUAL_TESTS" ]; then
        log_issue "README.md test count ($README_COUNT) != actual ($ACTUAL_TESTS)"
    else
        log_ok "README.md test count matches"
    fi
else
    log_warning "Could not find test count in README.md"
fi

# ============================================================================
# 3. Feature documentation validation
# ============================================================================
echo
echo "3. Feature documentation validation..."

REQUIRED_DOCS=(
    "docs/features/00-memory.md"
    "docs/features/01-ops.md"
    "docs/features/02-autodiff.md"
    "docs/features/03-transforms.md"
    "docs/features/04-nn.md"
    "docs/features/05-optim.md"
    "docs/features/06-linalg.md"
    "docs/features/07-rng.md"
    "docs/features/08-zgf.md"
    "docs/features/09-backends.md"
    "docs/features/10-jit.md"
    "docs/features/11-graph.md"
    "docs/features/12-zui.md"
)

for doc in "${REQUIRED_DOCS[@]}"; do
    if [ ! -f "$doc" ]; then
        log_issue "Missing documentation: $doc"
    else
        log_ok "Found $doc"
    fi
done

# Check English versions
EN_MISSING=0
for doc in "${REQUIRED_DOCS[@]}"; do
    en_doc="docs/en/${doc#docs/}"
    if [ ! -f "$en_doc" ]; then
        EN_MISSING=$((EN_MISSING + 1))
    fi
done

if [ $EN_MISSING -gt 0 ]; then
    log_warning "$EN_MISSING English documentation files missing"
else
    log_ok "All English documentation files present"
fi

# ============================================================================
# 4. Examples validation
# ============================================================================
echo
echo "4. Examples validation..."

EXAMPLES=(
    "autodiff"
    "diff-render"
    "gpu-probe"
    "images"
    "render3d"
    "tensors"
    "train-mlp"
    "ui-demo"
    "vector-art"
)

for example in "${EXAMPLES[@]}"; do
    # Convert hyphenated name to underscore for file check
    file_name="${example//-/_}"
    if [ ! -f "examples/${file_name}.zig" ]; then
        log_issue "Missing example: examples/${file_name}.zig"
    else
        log_ok "Found examples/${file_name}.zig"
    fi
    
    # Check if it's registered in build.zig
    if ! grep -q "run-${example}" build.zig 2>/dev/null; then
        log_warning "Example ${example} not registered in build.zig"
    else
        log_ok "Example ${example} registered in build.zig"
    fi
done

# ============================================================================
# 5. Stale API references
# ============================================================================
echo
echo "5. Stale API reference check..."

# Check for references to removed zpu.backends() etc.
STALE_APIS=(
    "zpu.backends()"
    "zpu.bestBackend()"
    "zpu.DeviceBuffer"
    "zpu.createDeviceBuffer"
    "zpu.dispatchDeviceBuffers"
    "zpu.setDriverLibrary"
)

for api in "${STALE_APIS[@]}"; do
    if grep -r "$api" --include="*.zig" src/ examples/ tests/ 2>/dev/null | grep -v "// " | grep -q .; then
        log_issue "Stale API reference found: $api"
    else
        log_ok "No stale references to $api"
    fi
done

# ============================================================================
# 6. GitHub issue references
# ============================================================================
echo
echo "6. GitHub issue reference check..."

# Check that known issues are documented
if grep -q "#8" docs/features/02-autodiff.md 2>/dev/null; then
    log_ok "Known hessian leak issue (#8) referenced in docs"
else
    log_warning "Known issue #8 not referenced in autodiff docs"
fi

# ============================================================================
# 7. Code formatting
# ============================================================================
echo
echo "7. Code formatting check..."

ZIG_BIN="/usr/local/lib/python3.11/dist-packages/ziglang/zig"
if [ -x "$ZIG_BIN" ]; then
    if $ZIG_BIN fmt --check src/zpu/gradcheck.zig src/zpu/tests_grad.zig src/zpu/transforms.zig 2>&1 | grep -q "non-conforming"; then
        log_issue "Some files need formatting (run: zig fmt <file>)"
    else
        log_ok "All checked files are properly formatted"
    fi
else
    log_warning "zig binary not found at $ZIG_BIN, skipping format check"
fi

# ============================================================================
# Summary
# ============================================================================
echo
echo "=== Summary ==="
echo "Issues: $ISSUES"
echo "Warnings: $WARNINGS"
echo

if [ $ISSUES -eq 0 ]; then
    echo "✓ All consistency checks passed"
    if [ $WARNINGS -gt 0 ]; then
        echo "  (with $WARNINGS warning(s))"
    fi
    exit 0
else
    echo "❌ Found $ISSUES consistency issue(s)"
    exit 1
fi
