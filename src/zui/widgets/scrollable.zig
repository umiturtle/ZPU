//! Scrollable container widget - a container that scrolls its children.

const std = @import("std");
const ui_mod = @import("../ui.zig");
const types = @import("../types.zig");

const Ui = ui_mod.Ui;
const Rect = types.Rect;
const Style = ui_mod.Style;
const LayoutFrame = ui_mod.LayoutFrame;

pub const ScrollOptions = struct {
    style: Style = .{},
    bar: f32 = 8,
};

/// A container that scrolls its children. The offset is retained (it cannot
/// be recomputed from a single frame's input), the clipping is the
/// painter's, and the bar only appears when the content does not fit.
pub fn scrollable(self: *Ui, id: []const u8, opts: ScrollOptions, body: anytype) Ui.Error!void {
    const avail = self.available();
    self.pushId(id);
    defer self.popId();
    const key = self.idOf("scroll");

    const measured = try self.measureContainer(opts.style, .{ .x = @max(0, avail.x - opts.bar), .y = std.math.floatMax(f32) }, body);
    const r = self.occupy(opts.style, measured.content);
    if (self.measuring) return;
    if (r.isEmpty()) return;

    // Unlike a plain container, a scroll region paints itself even with a
    // transparent style: it owns the pixels that the bar and the clipped
    // children are drawn against.
    try self.paintRounded(r, opts.style.radius, opts.style.bg, opts.style.border, opts.style.border_color);

    const view = r.inset(.{ .l = opts.style.pad.l, .t = opts.style.pad.t, .r = opts.style.pad.r + opts.bar, .b = opts.style.pad.b });
    const content_h = measured.content.y;
    const scroll = try self.scrollRegion(key, r, view, content_h, opts.bar, opts.style.pad);

    // Children are laid out in an unclipped column at `-scroll` and then
    // clipped: layout never has to know about scrolling.
    var frame = LayoutFrame{
        .mode = .draw,
        .dir = opts.style.dir,
        .content = Rect.init(view.x, view.y - scroll, view.w, content_h),
        .gap = opts.style.gap,
        .cross_align = opts.style.cross_align,
        .cursor = view.y - scroll,
        .cross = view.w,
        .fixed_main = measured.fixed_main,
        .fill_content = measured.fill_content,
        .weight = measured.weight,
        .gaps = measured.gaps,
        .max_cross = measured.max_cross,
    };
    frame.fill_extent = @max(0, content_h - measured.fixed_main);

    const previous_clip = self.clip;
    try self.pushClip(view);
    self.pushLayout(frame);
    defer {
        self.popLayout();
        self.popClip(previous_clip) catch {};
    }
    try body.run(self);
}

// ============================================================================
// Tests
// ============================================================================

test "scroll options defaults" {
    const opts = ScrollOptions{};
    try std.testing.expectEqual(@as(f32, 8), opts.bar);
}
