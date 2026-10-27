//! Slider widget - a draggable slider for numeric values
//!
//! Returns `true` on frames where the value changed.

const std = @import("std");
const ui_mod = @import("../ui.zig");
const font = @import("../font.zig");
const types = @import("../types.zig");

const Ui = ui_mod.Ui;
const Color = types.Color;
const Rect = types.Rect;
const Size = types.Size;
const Style = ui_mod.Style;

pub const SliderOptions = struct {
    min: f32 = 0,
    max: f32 = 1,
    w: Size = .{ .px = 160 },
    enabled: bool = true,
    /// Shown next to the label.
    show_value: bool = true,
};

/// A slider: dragging anywhere on the track moves the value (no "grab the
/// knob" friction), and the pixels map linearly onto the range.
pub fn slider(self: *Ui, content: []const u8, value: *f32, opts: SliderOptions) Ui.Error!bool {
    const id = self.idOf(content);
    const size = self.theme.font_size;
    const m = font.measure(content, size);
    var value_buf: [32]u8 = undefined;
    const value_text = if (opts.show_value)
        std.fmt.bufPrint(&value_buf, "{d:.2}", .{value.*}) catch ""
    else
        "";
    const vm = font.measure(value_text, size);
    const label_w = m.width + self.theme.pad;
    const min_w = label_w + @max(opts.w.resolve(0, 0), 40) + vm.width;

    const style = Style{ .w = opts.w, .h = Size{ .px = @max(self.theme.item_h, m.height) }, .min_w = min_w };
    const r = self.occupy(style, .{ .x = min_w, .y = m.height });
    const resp = self.respond(id, r, true, opts.enabled);
    if (self.measuring) return false;

    const range = if (opts.max > opts.min) opts.max - opts.min else 1;
    var changed = false;
    const track = Rect.init(
        r.x + label_w,
        r.y + r.h * 0.5 - 3,
        @max(20, r.w - label_w - (if (opts.show_value) vm.width + self.theme.pad else 0)),
        6,
    );
    if (resp.active and opts.enabled) {
        const t = std.math.clamp((self.input.pointer.x - track.x) / @max(1, track.w), 0, 1);
        const next = opts.min + t * range;
        if (next != value.*) {
            value.* = next;
            changed = true;
        }
    }
    if (resp.focused and opts.enabled) {
        const step = range / 100;
        if (self.input.keyPressed(.left)) {
            value.* = std.math.clamp(value.* - step, opts.min, opts.max);
            changed = true;
        }
        if (self.input.keyPressed(.right)) {
            value.* = std.math.clamp(value.* + step, opts.min, opts.max);
            changed = true;
        }
    }
    if (changed) self.markDirty();

    const t = std.math.clamp((value.* - opts.min) / range, 0, 1);
    try self.paintRounded(track, 3, self.theme.track, 1, self.theme.border);
    if (t > 0) {
        try self.paintRounded(Rect.init(track.x, track.y, track.w * t, track.h), 3, self.theme.accent, 0, Color.transparent);
    }
    const knob = Rect.init(track.x + track.w * t - 6, r.y + r.h * 0.5 - 6, 12, 12);
    try self.paintRounded(knob, 6, Color.lerp(self.theme.text, self.theme.accent_text, resp.hover_t * 0.5), 1, self.theme.border);

    const baseline = r.y + (r.h - m.height) * 0.5 + m.ascent;
    try self.paintText(r.x, baseline, size, self.theme.text_dim, content);
    if (opts.show_value) {
        const vx = r.x + r.w - vm.width;
        try self.paintText(vx, baseline, size, self.theme.text, value_text);
    }
    if (resp.focused) try paintFocusRing(self, track, 4);
    return changed;
}

fn paintFocusRing(self: *Ui, r: Rect, radius: f32) Ui.Error!void {
    try self.paintRounded(r.shrink(-1.5), radius + 1.5, Color.transparent, 1.5, self.theme.accent.withAlpha(0.9));
}

// ============================================================================
// Tests
// ============================================================================

test "slider options defaults" {
    const opts = SliderOptions{};
    try std.testing.expectEqual(@as(f32, 0), opts.min);
    try std.testing.expectEqual(@as(f32, 1), opts.max);
    try std.testing.expect(opts.enabled);
    try std.testing.expect(opts.show_value);
}
