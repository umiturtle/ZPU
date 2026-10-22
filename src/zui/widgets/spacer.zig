//! Spacer widget - empty space along the container's main axis.

const std = @import("std");
const ui_mod = @import("../ui.zig");
const types = @import("../types.zig");

const Ui = ui_mod.Ui;
const Size = types.Size;

/// Empty space along the container's main axis.
pub fn spacer(self: *Ui, extent: f32) void {
    _ = self.occupy(.{ .w = Size{ .px = extent }, .h = Size{ .px = extent } }, .{ .x = extent, .y = extent });
}

// ============================================================================
// Tests
// ============================================================================

test "spacer exists" {
    // Just verify the function exists and has correct signature
    const Func = @TypeOf(spacer);
    const info = @typeInfo(Func);
    try std.testing.expect(info == .@"fn");
}
