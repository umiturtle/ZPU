//! Label widget - a run of text
//!
//! Wraps to the container's width unless told otherwise.

const std = @import("std");
const ui_mod = @import("../ui.zig");
const font = @import("../font.zig");

const Ui = ui_mod.Ui;
const Color = @import("../types.zig").Color;

pub const LabelOptions = struct {
    color: ?Color = null,
    size: ?f32 = null,
    wrap: bool = true,
};

/// A run of text. Wraps to the container's width unless told otherwise.
pub fn label(self: *Ui, content: []const u8, opts: LabelOptions) Ui.Error!void {
    const size = opts.size orelse self.theme.font_size;
    const color = opts.color orelse self.theme.text;
    const max_w = if (opts.wrap) self.contentWidth() else std.math.floatMax(f32);
    const lines = self.wrapText(content, size, max_w);
    const line_h = font.measure("", size).height;

    var widest: f32 = 0;
    for (lines) |line| widest = @max(widest, font.measure(content[line.start .. line.start + line.len], size).width);

    const r = self.occupy(.{}, .{ .x = widest, .y = line_h * @as(f32, @floatFromInt(lines.len)) });
    if (self.measuring) return;
    var baseline = r.y + font.measure("", size).ascent;
    for (lines) |line| {
        try self.paintText(r.x, baseline, size, color, content[line.start .. line.start + line.len]);
        baseline += line_h;
    }
}

// ============================================================================
// Tests
// ============================================================================

test "label options defaults" {
    const opts = LabelOptions{};
    try std.testing.expectEqual(@as(?Color, null), opts.color);
    try std.testing.expectEqual(@as(?f32, null), opts.size);
    try std.testing.expect(opts.wrap);
}
