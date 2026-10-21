//! Container widget - a basic layout container.

const std = @import("std");
const ui_mod = @import("../ui.zig");
const types = @import("../types.zig");

const Ui = ui_mod.Ui;
const Style = ui_mod.Style;

/// A basic container that lays out its children according to the given style.
pub fn container(self: *Ui, id: []const u8, style: Style, body: anytype) Ui.Error!void {
    const avail = self.available();
    self.pushId(id);
    defer self.popId();

    const measured = try self.measureContainer(style, avail, body);
    const r = self.occupy(style, measured.content);
    if (self.measuring) return;
    if (r.isEmpty()) return;

    if (style.bg.a > 0 or (style.border > 0 and style.border_color.a > 0) or style.radius > 0) {
        try self.paintRounded(r, style.radius, style.bg, style.border, style.border_color);
    }

    const content_rect = r.inset(style.pad);
    const previous_clip = self.clip;
    if (style.clip) try self.pushClip(r);
    defer if (style.clip) {
        self.popClip(previous_clip) catch {};
    };

    var frame = Ui.drawFrame(style, content_rect, measured);
    if (frame.weight == 0 and style.justify != .start) {
        const slack = @max(0, frame.contentMain() - measured.fixed_main);
        const off: f32 = switch (style.justify) {
            .start, .stretch => 0,
            .center => slack * 0.5,
            .end => slack,
        };
        frame.cursor += off;
    }
    self.pushLayout(frame);
    defer self.popLayout();
    try body.run(self);
}

// ============================================================================
// Tests
// ============================================================================

test "container exists" {
    // Just verify the function exists and has correct signature
    const Func = @TypeOf(container);
    const info = @typeInfo(Func);
    try std.testing.expect(info == .@"fn");
}
