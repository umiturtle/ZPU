//! Progress bar widget - display only, no id and no retained state.

const std = @import("std");
const ui_mod = @import("../ui.zig");
const types = @import("../types.zig");

const Ui = ui_mod.Ui;
const Color = types.Color;
const Rect = types.Rect;
const Size = types.Size;

pub const ProgressOptions = struct {
    w: f32 = 160,
    h: f32 = 8,
};

/// A progress bar: display only, so no id and no retained state.
pub fn progress(self: *Ui, value: f32, opts: ProgressOptions) Ui.Error!void {
    const r = self.occupy(.{ .w = Size{ .px = opts.w }, .h = Size{ .px = opts.h } }, .{ .x = opts.w, .y = opts.h });
    if (self.measuring) return;
    const t = std.math.clamp(value, 0, 1);
    try self.paintRounded(r, r.h * 0.5, self.theme.track, 1, self.theme.border);
    if (t > 0) {
        try self.paintRounded(Rect.init(r.x, r.y, r.w * t, r.h), r.h * 0.5, self.theme.accent, 0, Color.transparent);
    }
}

// ============================================================================
// Tests
// ============================================================================

test "progress options defaults" {
    const opts = ProgressOptions{};
    try std.testing.expectEqual(@as(f32, 160), opts.w);
    try std.testing.expectEqual(@as(f32, 8), opts.h);
}
