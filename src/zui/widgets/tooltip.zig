//! Tooltip widget - a popup that appears when hovering over an anchored rect.

const std = @import("std");
const ui_mod = @import("../ui.zig");
const types = @import("../types.zig");

const Ui = ui_mod.Ui;
const Id = types.Id;
const Rect = types.Rect;

/// Show a tooltip when the widget with `id` is hot.
/// Unlike a popup needs a second pass over the whole window, a tooltip does not.
pub fn tooltip(self: *Ui, id: Id, anchor: Rect, text: []const u8) Ui.Error!void {
    if (self.measuring or self.skipping) return;
    if (!self.ctx.hot.eql(id) or text.len == 0) return;
    self.tip = .{ .anchor = anchor, .text = text };
}

// ============================================================================
// Tests
// ============================================================================

test "tooltip exists" {
    // Just verify the function exists and has correct signature
    const Func = @TypeOf(tooltip);
    const info = @typeInfo(Func);
    try std.testing.expect(info == .@"fn");
}
