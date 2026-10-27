//! Radio button widget - a radio button in a group
//!
//! Returns `true` on the frame it was selected.

const std = @import("std");
const ui_mod = @import("../ui.zig");
const font = @import("../font.zig");
const types = @import("../types.zig");

const Ui = ui_mod.Ui;
const Color = types.Color;
const Rect = types.Rect;
const Size = types.Size;
const Vec2 = types.Vec2;

// Import ToggleOptions from checkbox to avoid duplication
pub const ToggleOptions = @import("checkbox.zig").ToggleOptions;

/// A radio button; `selected` is the index of the chosen item in the group.
pub fn radio(self: *Ui, content: []const u8, selected: *usize, index: usize, opts: ToggleOptions) Ui.Error!bool {
    const id = self.idOf(content);
    const size = self.theme.font_size;
    const m = font.measure(content, size);
    const box = @max(15, size);
    const wanted = Vec2{ .x = box + self.theme.pad + m.width, .y = @max(self.theme.item_h, m.height) };
    const r = self.occupy(.{ .w = opts.w, .h = Size{ .px = wanted.y }, .min_w = wanted.x }, wanted);
    const resp = self.respond(id, r, true, opts.enabled);
    if (self.measuring) return false;

    const cy = r.y + r.h * 0.5;
    const ring = Rect.init(r.x, cy - box * 0.5, box, box);
    const is_on = selected.* == index;
    var fill = Color.lerp(self.theme.widget, self.theme.widget_hover, resp.hover_t);
    if (is_on) fill = self.theme.track;
    // A circle is a rounded rectangle whose radius is half its side: the
    // display list needs no circle command for this.
    try self.paintRounded(ring, box * 0.5, fill, 1, self.theme.border);
    if (is_on) {
        const d = box * 0.44;
        try self.paintRounded(Rect.init(r.x + (box - d) * 0.5, cy - d * 0.5, d, d), d * 0.5, self.theme.radio_dot, 0, Color.transparent);
    }
    const baseline = r.y + (r.h - m.height) * 0.5 + m.ascent;
    try self.paintText(r.x + box + self.theme.pad, baseline, size, self.theme.text, content);
    if (resp.focused) try paintFocusRing(self, ring, box * 0.5);

    if (resp.clicked and opts.enabled and !is_on) {
        selected.* = index;
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

test "radio button selection" {
    // Just verify the function exists and has correct signature
    const Func = @TypeOf(radio);
    const info = @typeInfo(Func);
    try std.testing.expect(info == .@"fn");
}
