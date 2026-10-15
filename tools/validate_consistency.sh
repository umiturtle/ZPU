#!/bin/bash
# Validate consistency between MILESTONES.md, README.md, and code.
#
# Usage: ./tools/validate_consistency.sh
#
# Checks:
# 1. zpu does not import zbackend (it should be independent)
# 2. build.zig dependency graph matches documented structure
# 3. Test counts in MILESTONES.md and README.md match
#
# Exit code 0 if all consistent, 1 if any mismatch found.

set -e

echo "=== Consistency Validation ==="
echo

ISSUES=0

# 1. Check that zpu does not import zbackend
echo "1. Checking zpu independence from zbackend..."

if grep -q "zbackend" src/zpu/context.zig; then
    echo "   ❌ src/zpu/context.zig still imports zbackend (should be independent)"
    ISSUES=$((ISSUES + 1))
else
    echo "   ✓ src/zpu/context.zig does not import zbackend"
fi

if grep -q "zbackend" src/zpu.zig; then
    echo "   ❌ src/zpu.zig still references zbackend (should be independent)"
    ISSUES=$((ISSUES + 1))
else
    echo "   ✓ src/zpu.zig does not reference zbackend"
fi

# 2. Check build.zig dependency graph
echo
echo "2. Checking build.zig dependency graph..."

if grep -q 'zpu.addImport("zbackend"' build.zig; then
    echo "   ❌ build.zig still adds zbackend import to zpu"
    ISSUES=$((ISSUES + 1))
else
    echo "   ✓ build.zig does not add zbackend to zpu"
fi

if grep -q 'zgf.addImport("zbackend"' build.zig; then
    echo "   ❌ build.zig still adds zbackend import to zgf"
    ISSUES=$((ISSUES + 1))
else
    echo "   ✓ build.zig does not add zbackend to zgf"
fi

if grep -q 'zui.addImport("zbackend"' build.zig; then
    echo "   ✓ build.zig adds zbackend to zui (expected)"
else
    echo "   ❌ build.zig missing zbackend import for zui"
    ISSUES=$((ISSUES + 1))
fi

# 3. Check test counts in docs
echo
echo "3. Checking test counts in documentation..."

MILESTONES_COUNT=$(grep -oE '[0-9]+ テスト' MILESTONES.md | grep -oE '[0-9]+' | head -1)
README_COUNT=$(grep -oE '[0-9]+/[0-9]+ green' README.md | grep -oE '^[0-9]+' | head -1)

if [ -n "$MILESTONES_COUNT" ]; then
    echo "   MILESTONES.md claims: $MILESTONES_COUNT tests"
else
    echo "   ⚠ Could not find test count in MILESTONES.md"
fi

if [ -n "$README_COUNT" ]; then
    echo "   README.md claims: $README_COUNT tests"
else
    echo "   ⚠ Could not find test count in README.md"
fi

if [ -n "$MILESTONES_COUNT" ] && [ -n "$README_COUNT" ]; then
    if [ "$MILESTONES_COUNT" != "$README_COUNT" ]; then
        echo "   ❌ Test counts do not match!"
        ISSUES=$((ISSUES + 1))
    else
        echo "   ✓ Test counts match"
    fi
fi

echo
echo "   Note: Run \`zig build test --summary all\` to verify actual test count"

# Summary
echo
echo "=== Summary ==="
if [ $ISSUES -eq 0 ]; then
    echo "✓ All consistency checks passed"
    exit 0
else
    echo "❌ Found $ISSUES consistency issue(s)"
    exit 1
fi
