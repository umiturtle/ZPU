//! Checkbox widget - a toggleable checkbox with label
//!
//! Returns `true` on the frame it was toggled.

const std = @import("std");
const ui_mod = @import("../ui.zig");
const font = @import("../font.zig");
const types = @import("../types.zig");

const Ui = ui_mod.Ui;
const Color = types.Color;
const Rect = types.Rect;
const Size = types.Size;
const Vec2 = types.Vec2;

pub const ToggleOptions = struct {
    enabled: bool = true,
    w: Size = .auto,
};

/// A checkbox. `true` on the frame it was toggled.
pub fn checkbox(self: *Ui, content: []const u8, value: *bool, opts: ToggleOptions) Ui.Error!bool {
    const id = self.idOf(content);
    const size = self.theme.font_size;
    const m = font.measure(content, size);
    const box = @max(15, size);
    const wanted = Vec2{ .x = box + self.theme.pad + m.width, .y = @max(self.theme.item_h, m.height) };
    const r = self.occupy(.{ .w = opts.w, .h = Size{ .px = wanted.y }, .min_w = wanted.x }, wanted);
    const resp = self.respond(id, r, true, opts.enabled);
    if (self.measuring) return false;

    const bx = Rect.init(r.x, r.y + (r.h - box) * 0.5, box, box);
    var fill = Color.lerp(self.theme.widget, self.theme.widget_hover, resp.hover_t);
    if (value.*) fill = Color.lerp(fill, self.theme.accent, 1.0);
    try self.paintRounded(bx, self.theme.radius * 0.6, fill, 1, self.theme.border);
    if (value.*) {
        // A tick in two strokes: no glyph in a 95-character font is as
        // recognisable at 15 px.
        const c = bx.center();
        const w = box * 0.2;
        try self.paintLine(.{ .x = c.x - w * 1.4, .y = c.y }, .{ .x = c.x - w * 0.2, .y = c.y + w * 1.2 }, 2, self.theme.accent_text);
        try self.paintLine(.{ .x = c.x - w * 0.2, .y = c.y + w * 1.2 }, .{ .x = c.x + w * 1.5, .y = c.y - w * 1.1 }, 2, self.theme.accent_text);
    }
    const baseline = r.y + (r.h - m.height) * 0.5 + m.ascent;
    try self.paintText(r.x + box + self.theme.pad, baseline, size, if (opts.enabled) self.theme.text else self.theme.text_dim, content);
    if (resp.focused) try paintFocusRing(self, bx, self.theme.radius * 0.6);

    if (resp.clicked and opts.enabled) {
        value.* = !value.*;
        self.markDirty();
        return true;
    }
    return false;
}

fn paintFocusRing(self: *Ui, r: Rect, radius: f32) Ui.Error!void {
    try self.paintRounded(r.shrink(-1.5), radius + 1.5, Color.transparent, 1.5, self.theme.accent.withAlpha(0.9));
}

// ============================================================================
// Tests
// ============================================================================

test "toggle options defaults" {
    const opts = ToggleOptions{};
    try std.testing.expect(opts.enabled);
    try std.testing.expectEqual(Size.auto, opts.w);
}
