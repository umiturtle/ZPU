//! Measure container helper - measures container content without laying it out.

const std = @import("std");
const ui_mod = @import("../ui.zig");
const types = @import("../types.zig");

const Ui = ui_mod.Ui;
const Style = ui_mod.Style;
const Rect = types.Rect;
const Vec2 = types.Vec2;
const Measured = Ui.Measured;

/// Measures container content without actually laying it out.
/// Returns the measured dimensions needed for layout.
pub fn measureContainer(self: *Ui, style: Style, avail: Vec2, body: anytype) Ui.Error!Measured {
    const was_measuring = self.measuring;
    self.measuring = true;
    self.pushLayout(.{
        .mode = .measure,
        .dir = style.dir,
        .content = Rect.init(0, 0, avail.x, avail.y),
        .gap = style.gap,
        .cross_align = style.cross_align,
        .cross = avail.x,
    });
    try body.run(self);
    const frame = self.layouts[self.depth - 1];
    self.popLayout();
    self.measuring = was_measuring;

    const main = frame.fixed_main + frame.fill_content;
    return .{
        .content = if (style.dir == .x)
            Vec2{ .x = main, .y = frame.max_cross }
        else
            Vec2{ .x = frame.max_cross, .y = main },
        .fixed_main = frame.fixed_main,
        .fill_content = frame.fill_content,
        .weight = frame.weight,
        .gaps = frame.gaps,
        .max_cross = frame.max_cross,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "measureContainer exists" {
    // Just verify the function exists and has correct signature
    const Func = @TypeOf(measureContainer);
    const info = @typeInfo(Func);
    try std.testing.expect(info == .@"fn");
}
