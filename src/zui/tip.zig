//! Tip - a tooltip waiting to be painted.

const types = @import("types.zig");
const Rect = types.Rect;

/// A tooltip waiting to be painted at the end of the frame.
pub const Tip = struct {
    anchor: Rect,
    text: []const u8,
};

// ============================================================================
// Tests
// ============================================================================

const std = @import("std");

test "tip struct" {
    const tip = Tip{
        .anchor = Rect{ .x = 10, .y = 20, .w = 100, .h = 50 },
        .text = "tooltip text",
    };
    try std.testing.expect(tip.anchor.x == 10);
    try std.testing.expect(tip.anchor.y == 20);
    try std.testing.expectEqualStrings("tooltip text", tip.text);
}
