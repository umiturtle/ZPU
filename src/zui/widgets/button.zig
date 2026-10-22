//! Button widget - a push button
//!
//! `true` on the frame it was activated — by a click, or by
//! `enter`/`space` while focused.

const std = @import("std");
const ui_mod = @import("../ui.zig");
const font = @import("../font.zig");
const types = @import("../types.zig");

const Ui = ui_mod.Ui;
const Color = types.Color;
const Edges = types.Edges;
const Id = types.Id;
const Rect = types.Rect;
const Size = types.Size;
const Style = ui_mod.Style;

pub const ButtonOptions = struct {
    w: Size = .auto,
    h: ?f32 = null,
    pad: ?Edges = null,
    enabled: bool = true,
    radius: ?f32 = null,
    /// Draw a border.
    border: bool = true,
};

/// A push button. `true` on the frame it was activated.
pub fn button(self: *Ui, content: []const u8, opts: ButtonOptions) Ui.Error!bool {
    return buttonId(self, self.idOf(content), content, opts);
}

/// A button whose id is explicit: for labels that change (an id that moves
/// loses the press) or for rows of identical labels.
pub fn buttonId(self: *Ui, id: Id, content: []const u8, opts: ButtonOptions) Ui.Error!bool {
    const size = self.theme.font_size;
    const m = font.measure(content, size);
    const pad = opts.pad orelse Edges.xy(self.theme.pad + 2, (self.theme.item_h - m.height) * 0.5);
    const style = Style{
        .w = opts.w,
        .h = Size{ .px = opts.h orelse @max(self.theme.item_h, m.height + pad.sum(.y)) },
        .pad = pad,
        .min_w = m.width + pad.sum(.x),
    };
    const r = self.occupy(style, .{ .x = m.width, .y = m.height });
    const resp = self.respond(id, r, true, opts.enabled);
    if (self.measuring) return false;

    const radius = opts.radius orelse self.theme.radius;
    var fill = self.theme.widget;
    if (!opts.enabled) {
        fill = Color.lerp(self.theme.widget, self.theme.bg, 0.5);
    } else {
        fill = Color.lerp(self.theme.widget, self.theme.widget_hover, resp.hover_t);
        fill = Color.lerp(fill, self.theme.accent, resp.press_t * 0.85);
    }
    try self.paintRounded(r, radius, fill, if (opts.border) 1 else 0, self.theme.border);
    const text_color = if (!opts.enabled)
        self.theme.text_dim.scale(0.7)
    else if (resp.press_t > 0.5)
        self.theme.accent_text
    else
        self.theme.text;
    const baseline = r.y + (r.h - m.height) * 0.5 + m.ascent;
    try self.paintText(r.x + (r.w - m.width) * 0.5, baseline, size, text_color, content);
    if (resp.focused) try paintFocusRing(self, r, radius);
    return resp.clicked and opts.enabled;
}

fn paintFocusRing(self: *Ui, r: Rect, radius: f32) Ui.Error!void {
    try self.paintRounded(r.shrink(-1.5), radius + 1.5, Color.transparent, 1.5, self.theme.accent.withAlpha(0.9));
}

// ============================================================================
// Tests
// ============================================================================

test "button options defaults" {
    const opts = ButtonOptions{};
    try std.testing.expectEqual(Size.auto, opts.w);
    try std.testing.expectEqual(@as(?f32, null), opts.h);
    try std.testing.expect(opts.enabled);
    try std.testing.expect(opts.border);
}
