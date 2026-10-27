//! Separator widget - a horizontal line
//!
//! Draws a 1-pixel line using the theme's border color.

const std = @import("std");
const ui_mod = @import("../ui.zig");

const Ui = ui_mod.Ui;
const Size = @import("../types.zig").Size;

/// A horizontal separator line.
pub fn separator(self: *Ui) Ui.Error!void {
    const r = self.occupy(.{ .w = .full, .h = Size{ .px = 1 } }, .{ .x = 0, .y = 1 });
    if (self.measuring) return;
    try self.paintRect(r, self.theme.border);
}

// ============================================================================
// Tests
// ============================================================================

test "separator exists" {
    // Just verify the function exists and has correct signature
    const Func = @TypeOf(separator);
    const info = @typeInfo(Func);
    try std.testing.expect(info == .@"fn");
}
